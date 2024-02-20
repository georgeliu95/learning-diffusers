# Copyright George L
# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import glob
from typing import List, Tuple, Union
from typing_extensions import Literal
import warnings

import pandas as pd

import torch
import torchvision
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torchmetrics.image.inception import InceptionScore as TorchInceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance as TorchFrechetInceptionDistance
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.multimodal.clip_score import CLIPScore as TorchClipScore

from transformers import CLIPModel
from transformers import CLIPProcessor


def dim_zero_cat(x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


class InceptionScore():
    def __init__(
            self,
            splits:int = 2,
            normalize: bool = True,
            use_torchmetrics: bool = False):
        self.inception_score = None, None
        self.feature_list = []
        self.splits = splits
        self.normalize = normalize
        # Load inception model
        self.use_torchmetrics = use_torchmetrics
        if self.use_torchmetrics:
            self.inception_score_fn = TorchInceptionScore(normalize=self.normalize, splits=self.splits)
            self.inception_score_fn.inception.cuda()
        else:
            self.inception_model = inception_v3(transform_input=False, weights=Inception_V3_Weights.DEFAULT).eval().cuda()
        return

    def feed_dataset(self, dataset: List):
        with torch.no_grad():
            for image_tensor in dataset:
                if self.use_torchmetrics:
                    self.inception_score_fn.update(image_tensor)
                else:
                    image_tensor = (image_tensor * 255).byte() if self.normalize else image_tensor
                    model_dtype = getattr(self.inception_model, "_dtype", torch.float)
                    self.feature_list.append(self.inception_model(image_tensor.to(model_dtype)))
        return

    def compute(self):
        if self.use_torchmetrics:
            self.inception_score = self.inception_score_fn.compute()
        else:
            features = dim_zero_cat(self.feature_list)
            # random permute the features
            features = features[torch.randperm(features.shape[0])]

            # calculate probs and logits
            prob = features.softmax(dim=1)
            log_prob = features.log_softmax(dim=1)

            # split into groups
            prob = prob.chunk(self.splits, dim=0)
            log_prob = log_prob.chunk(self.splits, dim=0)

            # calculate score per split
            mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
            kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
            kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
            kl = torch.stack(kl_)
            self.inception_score = kl.mean(), kl.std()
        return self.inception_score


class FrechetInceptionDistance():
    r"""This metric is known to be unstable in its calculatations, and we recommend for the best results using this metric 
        that you calculate using torch.float64 (default is torch.float32).
    """
    def __init__(
            self,
            feature: int = 2048,
            normalize: bool = True,
            data_type = torch.float64,
            use_torchmetrics: bool = False):
        valid_int_input = (
            64,     # First max pooling features
            192,    # Second max pooling featurs
            768,    # Pre-aux classifier features
            2048    # Final average pooling features
        )
        if feature not in valid_int_input:
            raise ValueError(f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}.")
        self.feature = feature
        self.normailze = normalize
        self.dtype = data_type
        self.use_torchmetrics = use_torchmetrics
        if self.use_torchmetrics:
            self.fid_score_fn = TorchFrechetInceptionDistance(feature=self.feature, normalize=self.normalize).set_dtype(self.dtype)
            self.fid_score_fn.inception.cuda()
        else:
            self.inception_model = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(self.feature)]).eval().cuda()
            self.feature_collection = {
                "real_features_sum": 0,
                "real_features_conv_sum": 0,
                "real_features_num_samples": 0,
                "artf_features_sum": 0,
                "artf_features_conv_sum": 0,
                "artf_features_num_samples": 0,
            }
        return

    def feed_dataset(self, dataset: List[Union[List, Tuple]]):
        r"""Sample from dataset should be (image_tensor: torch.Tensor, real_or_artf: bool)
        """
        with torch.no_grad():
            for image_tensor, real_or_artf in dataset:
                if self.use_torchmetrics:
                    self.fid_score_fn.update(image_tensor, real_or_artf)
                else:
                    image_tensor = (image_tensor * 255).byte() if self.normalize else image_tensor
                    features = self.inception_model(image_tensor)
                    self.orig_dtype = features.dtype
                    features = features.double()
                    if features.dim() == 1:
                        features = features.unsqueeze(0)
                    if real_or_artf:
                        self.feature_collection["real_features_sum"] += features.sum(dim=0)
                        self.feature_collection["real_features_cov_sum"] += features.t().mm(features)
                        self.feature_collection["real_features_num_samples"] += image_tensor.shape[0]
                    else:
                        self.feature_collection["artf_features_sum"] += features.sum(dim=0)
                        self.feature_collection["artf_features_cov_sum"] += features.t().mm(features)
                        self.feature_collection["artf_features_num_samples"] += image_tensor.shape[0]
        return
    
    def compute(self):
        def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
            a = (mu1 - mu2).square().sum(dim=-1)
            b = sigma1.trace() + sigma2.trace()
            c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
            return a + b - 2 * c
        
        if self.use_torchmetrics:
            self.fid_score = self.fid_score_fn.compute()
        else:
            mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
            mean_artf = (self.artf_features_sum / self.artf_features_num_samples).unsqueeze(0)

            cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            cov_artf_num = self.artf_features_cov_sum - self.artf_features_num_samples * mean_artf.t().mm(mean_artf)
            cov_artf = cov_artf_num / (self.artf_features_num_samples - 1)
            self.fid_score = _compute_fid(mean_real.squeeze(0), cov_real, mean_artf.squeeze(0), cov_artf).to(self.orig_dtype)
        return self.fid_score


class ClipScore():
    def __init__(
            self,
            clip_model_name_or_path: Literal[
                    "openai/clip-vit-base-patch16",
                    "openai/clip-vit-base-patch32",
                    "openai/clip-vit-large-patch14-336",
                    "openai/clip-vit-large-patch14",
                ] = "openai/clip-vit-base-patch16",
            use_torchmetrics: bool = False):
        self.score = 0
        self.n_samples = 0
        self.clip_score
        self.use_torchmetrics = use_torchmetrics
        if self.use_torchmetrics:
            torch.manual_seed(0)
            self.clip_score_fn = TorchClipScore(model_name_or_path=clip_model_name_or_path).eval().cuda()
        else:
            self.clip_model = CLIPModel.from_pretrained(clip_model_name_or_path)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name_or_path)
        return

    def feed_dataset(self, dataset: List[Union[List, Tuple]]):
        r"""Sample from dataset should be (image_tensor: torch.Tensor, prompt: str)
        """
        images = [sample[0].squeeze() for sample in dataset]
        prompts = [sample[1] for sample in dataset]
        if not all(i.ndim == 3 for i in images):
            raise ValueError("Expected all images to be 3d but found image that has either more or less")

        with torch.no_grad():
            if self.use_torchmetrics:
                self.clip_score_fn.update(images, prompts)
                self.score = self.clip_score_fn.score
                self.n_samples = self.clip_score_fn.n_sample
            else:
                device = images[0].device
                processed_input = self.clip_processor(text=prompts, images=[i.cpu() for i in images], return_tensors="pt", padding=True)

                img_features = self.clip_model.get_image_features(processed_input["pixel_values"].to(device))
                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

                max_position_embeddings = self.clip_model.config.text_config.max_position_embeddings
                if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
                    warnings.warn(
                        f"Encountered caption longer than {max_position_embeddings=}. Will truncate captions to this length."
                        "If longer captions are needed, initialize argument `model_name_or_path` with a model that supports"
                        "longer sequences", UserWarning)
                    processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
                    processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

                txt_features = self.clip_model.get_text_features(
                    processed_input["input_ids"].to(device), processed_input["attention_mask"].to(device)
                )
                txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

                # cosine similarity between feature vectors
                self.score = 100 * (img_features * txt_features).sum(axis=-1)
                self.n_samples = len(prompts)
        return

    def compute(self, image_tensor, prompts):
        if self.use_torchmetrics:
            self.clip_score = self.clip_score_fn.compute()
        else:
            self.clip_score = torch.max(self.score / self.n_samples, torch.zeros_like(self.score))
        return self.clip_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--eval-dataset', type=str, required=True, help="The directory of image dataset to be evaluated.")
    parser.add_argument('--fid-image-csv', type=str, default='', help="CSV file contains information of dataset for FID")
    parser.add_argument('--prompt-image-csv', type=str, default='', help="CSV file contains information about prompt-to-image pair")
    parser.add_argument('--torch-metrics', action='store_true', help="Whether to use torchmetrics for calculation")
    args = parser.parse_args()

    if os.path.isdir(args.eval_dataset):
        found_png_files = glob.glob(f"{args.eval_dataset}/*.png")
        print(f"Find {len(found_png_files)} png images")
        images = []
        for png_file in found_png_files:
            png_image = torchvision.io.read_image(png_file)
            png_image = torchvision.transforms.Resize((299,299))(png_image) ## InceptionV3 trained with 3x299x299 images
            images.append(png_image.unsqueeze(0).cuda() if png_image.ndim == 3 else png_image.cuda())
    else:
        raise RuntimeError(f"{args.eval_dataset} is not a valid directory.")

    ## Test inception score
    inception_score_metric = InceptionScore(use_torchmetrics=args.torch_metrics)
    inception_score_metric.feed_dataset(images)
    inception_score = inception_score_metric.compute()
    print(f"Inception score: {inception_score}")

    ## Test FID
    if args.fid_image_csv == '' or not os.path.isfile(args.fid_image_csv):
        raise RuntimeError(f"Please pass valid csv file by `--fid-image-csv csv_filename` (current: {args.fid_image_csv})")
    fid_image_dataset = [
        (torchvision.io.read_image(image_path), is_real) for image_path, is_real in pd.read_csv(args.fid_image_csv).values.tolist()]
    fid_score_metric = FrechetInceptionDistance(use_torchmetrics=args.torch_metrics)
    fid_score_metric.feed_dataset()
    fid_score = fid_score_metric.compute()
    print(f"FID score: {fid_score}")

    ## Test CLIP score
    if args.prompt_image_csv == '' or not os.path.isfile(args.prompt_image_csv):
        raise RuntimeError(f"Please pass valid csv file by `--prompt-image-csv csv_filename` (current: {args.prompt_image_csv})")
    prompt_image_pairs = pd.read_csv(args.prompt_image_csv).values.tolist()
    prompt_image_dataset = [(torchvision.io.read_image(image_path), prompt) for image_path, prompt in prompt_image_pairs]
    clip_score_metric = ClipScore(use_torchmetrics=args.torch_metrics)
    clip_score_metric.feed_dataset(prompt_image_dataset)
    clip_score = clip_score_metric.compute()
    print(f"CLIP score: {clip_score}")
