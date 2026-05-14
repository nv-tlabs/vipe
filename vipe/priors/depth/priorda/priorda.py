# This file includes code originally from the PriorDA repository:
# https://github.com/SpatialVision/Prior-Depth-Anything
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

import os
import time
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .dav2 import build_backbone
from .depth_completion import DepthCompletion
from .sparse_sampler import SparseSampler
from .utils import Arguments, depth2disparity, disparity2depth


class PriorDepthAnything(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        fmde_dir=None,
        cmde_dir=None,
        ckpt_dir=None,
        frozen_model_size=None,
        conditioned_model_size=None,
        coarse_only=False,
    ):
        super(PriorDepthAnything, self).__init__()

        self.args = Arguments()
        self.device = device
        """ 
        For inference stability, we set the output coarse/fine globally. 
        TODO : You can easily modify the code to specify the model to output coarse/fine depth sample-wisely.
        """
        self.coarse_only = coarse_only
        if frozen_model_size:
            self.args.frozen_model_size = frozen_model_size
        if conditioned_model_size:
            self.args.conditioned_model_size = conditioned_model_size

        ## Frozon MDE loading.
        if self.args.frozen_model_size in ["vitg"]:
            raise ValueError(f"{self.args.frozen_model_size} coming soon...")
        fmde_name = f"depth_anything_v2_{self.args.frozen_model_size}.pth"  # Download model checkpoints
        if fmde_dir is None:
            fmde_path = hf_hub_download(repo_id=self.args.repo_name, filename=fmde_name)
        else:
            fmde_path = os.path.join(fmde_dir, fmde_name)

        # Initialize Frozon-MDE.
        self.completion = DepthCompletion.build(args=self.args, fmde_path=fmde_path, device=device)

        ## Conditioned MDE loading.
        if not coarse_only:
            if self.args.conditioned_model_size in ["vitl", "vitg"]:
                raise ValueError(f"{self.args.conditioned_model_size} coming soon...")
            cmde_name = f"depth_anything_v2_{self.args.conditioned_model_size}.pth"  # Download model checkpoints
            if cmde_dir is None:
                cmde_path = hf_hub_download(repo_id=self.args.repo_name, filename=cmde_name)
            else:
                cmde_path = os.path.join(cmde_dir, cmde_name)

            # Initialize and load preptrained `prior-depth-anything` models.
            model = build_backbone(
                depth_size=self.args.conditioned_model_size,
                encoder_cond_dim=3,
                model_path=cmde_path,
            ).eval()
            self.model = self.load_checkpoints(model, ckpt_dir, self.device)

        self.sampler = SparseSampler(device=device)

    def load_checkpoints(self, model, ckpt_dir, device="cuda:0"):
        ckpt_name = f"prior_depth_anything_{self.args.conditioned_model_size}.pth"
        if ckpt_dir is None:
            ckpt_path = hf_hub_download(repo_id=self.args.repo_name, filename=ckpt_name)
        else:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        state_dict = torch.load(ckpt_path, map_location="cpu")

        new_state_dict = OrderedDict()
        for key, value in state_dict["model"].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        model = model.to(device)
        return model

    def forward(
        self,
        images,
        sparse_depths,
        sparse_masks,
        cover_masks=None,
        prior_depths=None,
        geometric_depths=None,
        pattern=None,
    ):
        """To facilitate further research, we batchify the forward process."""
        ##### Coarse stage. #####
        completed_maps = self.completion(
            images=images,
            sparse_depths=sparse_depths,
            sparse_masks=sparse_masks,
            cover_masks=cover_masks,
            prior_depths=prior_depths,
            pattern=pattern,
            geometric_depths=geometric_depths,
        )

        # knn-aligned depths
        comp_cond = completed_maps["scaled_preds"].unsqueeze(1)
        if self.coarse_only:
            coarse_depths = disparity2depth(comp_cond)
            return coarse_depths
        # Global Scale-Shift aligned depths.
        global_cond = completed_maps["global_preds"].unsqueeze(1)

        ##### Fine stage. #####
        if self.args.normalize_depth:
            # Obtain the value of norm params.
            masked_min, denom = self.zero_one_normalize(sparse_depths, sparse_masks, affine_only=True)

            global_depths = (disparity2depth(global_cond) - masked_min) / denom
            global_cond = depth2disparity(global_depths)

            comp_depths = (disparity2depth(comp_cond) - masked_min) / denom
            comp_cond = depth2disparity(comp_depths)
        condition = torch.cat([global_cond, comp_cond], dim=1)

        if self.args.err_condition:
            uctns = completed_maps["uncertainties"].unsqueeze(1)
            condition = torch.cat([uctns, condition], dim=1)

        # heit = sparse_depths.shape[-2] // 14 * 14
        heit = 518
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t0 = time.time()
        metric_disparities = self.model(images, heit, condition=condition, device=self.device)
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t1 = time.time()
            self.timer.append(t1 - t0)

        metric_depths = disparity2depth(metric_disparities)
        if self.args.normalize_depth:
            metric_depths = metric_depths * denom + masked_min
        return metric_depths

    def zero_one_normalize(self, depth_maps, valid_masks=None, affine_only=False):
        if valid_masks is not None:
            masked_min = (
                depth_maps.masked_fill(~valid_masks, float("inf")).min(dim=-1).values.min(dim=-1).values
            )  # (B, 1)
            masked_max = (
                depth_maps.masked_fill(~valid_masks, float("-inf")).max(dim=-1).values.max(dim=-1).values
            )  # (B, 1)
        else:
            masked_min = depth_maps.min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.max(dim=-1).values.max(dim=-1).values  # (B, 1)

        denom = masked_max - masked_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        masked_min = masked_min.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        denom = denom.view(-1, 1, 1, 1)

        if not affine_only:
            normalized = (depth_maps - masked_min) / denom
            return normalized, (masked_min, denom)
        else:
            return masked_min, denom

    @torch.no_grad()
    def infer_one_sample(
        self,
        image: Union[str, torch.Tensor, np.ndarray] = None,
        prior: Union[str, torch.Tensor, np.ndarray] = None,
        geometric: Union[str, torch.Tensor, np.ndarray] = None,
        pattern: str = None,
        double_global=False,
        prior_cover=False,
        visualize=False,
    ) -> torch.Tensor:
        """Perform inference. Return the refined/completed depth.

        Args:
            image:
                1. RGB in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Image path of RGB
            prior:
                1. Prior depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of prior depth map. (with scale)
            geometric:
                1. Geometric depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of geometric depth map. (with geometry)
            pattern: The mode of prior-based additional sampling. It could be None.
            double_global: Whether to condition with two estimated depths or estimated + knn-map.
            prior_cover: Whether to keep all prior areas in knn-map, it functions when 'pattern' is not None.
            visualize: Save results.


            Example1:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-2/rgb.jpg'
                >>> prior_path = 'assets/sample-2/prior_depth.png'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)

            Example2:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-6/rgb.npy'
                >>> prior_path = 'assets/sample-6/prior_depth.npy'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
        """

        # For each inference, params below should be reset.
        self.args.double_global = double_global
        assert image is not None and prior is not None

        ### Load and preprocess example images
        # We implement preprocess with batch size of 1, but our model works for multi-images naturally.
        data = self.sampler(
            image=image,
            prior=prior,
            geometric=geometric,
            pattern=pattern,
            K=self.args.K,
            prior_cover=prior_cover,
        )
        rgb, prior_depth, sparse_depth = (
            data["rgb"],
            data["prior_depth"],
            data["sparse_depth"],
        )  # Shape: [B, C, H, W]
        cover_mask, sparse_mask = (
            data["cover_mask"],
            data["sparse_mask"],
        )  # Shape: [B, 1, H, W]
        geometric_depth = data["geometric_depth"] if geometric is not None else None
        if (sparse_mask.view(sparse_mask.shape[0], -1).sum(dim=1) < self.args.K).any():
            raise ValueError("There are not enough known points in at least one of samples")

        ### The core inference stage.
        """ If you want to input multiple samples at once, just stack samples at dim=0, s.t. [B, C, H, W] """
        pred_depth = self.forward(
            images=rgb,
            sparse_depths=sparse_depth,
            prior_depths=prior_depth,
            sparse_masks=sparse_mask,
            cover_masks=cover_mask,
            pattern=pattern,
            geometric_depths=geometric_depth,
        )  # (B, 1, H, W)

        return pred_depth.squeeze()
