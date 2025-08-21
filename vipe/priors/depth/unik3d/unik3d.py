"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import importlib
import warnings
from copy import deepcopy
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from .models import encoder as mod
from .models.decoder import Decoder
from .utils.camera import BatchCamera, Camera
from .utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
from .utils.misc import get_params, last_stack, match_gt


def orthonormal_init(num_tokens, dims):
    pe = torch.randn(num_tokens, dims)
    # use Gram-Schmidt process to make the matrix orthonormal
    for i in range(num_tokens):
        for j in range(i):
            pe[i] -= torch.dot(pe[i], pe[j]) * pe[j]
        pe[i] = F.normalize(pe[i], p=2, dim=0)
    return pe


def get_paddings(original_shape, aspect_ratio_range):
    # Original dimensions
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    # Determine the closest aspect ratio within the range
    min_ratio, max_ratio = aspect_ratio_range
    target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape, pixels_range, shape_multiplier=14):
    # Original dimensions
    H_ori, W_ori = original_shape
    n_pixels_ori = W_ori * H_ori

    # Determine the closest number of pixels within the range
    min_pixels, max_pixels = pixels_range
    target_pixels = min(max_pixels, max(min_pixels, n_pixels_ori))

    # Calculate the resize factor
    resize_factor = (target_pixels / n_pixels_ori) ** 0.5
    new_width = int(W_ori * resize_factor)
    new_height = int(H_ori * resize_factor)
    new_height = ceil(new_height / shape_multiplier) * shape_multiplier
    new_width = ceil(new_width / shape_multiplier) * shape_multiplier

    return resize_factor, (new_height, new_width)


def _postprocess(tensor, shapes, paddings, interpolation_mode="bilinear"):

    # interpolate to original size
    tensor = F.interpolate(
        tensor, size=shapes, mode=interpolation_mode, align_corners=False
    )

    # remove paddings
    pad1_l, pad1_r, pad1_t, pad1_b = paddings
    tensor = tensor[..., pad1_t : shapes[0] - pad1_b, pad1_l : shapes[1] - pad1_r]
    return tensor


class UniK3D(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="UniK3D",
    repo_url="https://github.com/lpiccinelli-eth/UniK3D",
    tags=["monocular-metric-3D-estimation"],
):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.build(config)
        # self.build_losses(config)

    def pack_sequence(
        self,
        inputs: dict[str, torch.Tensor],
    ):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.reshape(-1, *value.shape[2:])
            elif isinstance(value, BatchCamera):
                inputs[key] = value.reshape(-1)
        return inputs

    def unpack_sequence(self, inputs: dict[str, torch.Tensor], B: int, T: int):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.reshape(B, T, *value.shape[1:])
            elif isinstance(value, BatchCamera):
                inputs[key] = value.reshape(B, T)
        return inputs

    def forward_train(self, inputs, image_metas):
        losses = {"opt": {}, "stat": {}}
        B, T = inputs["image"].shape[:2]
        image_metas[0]["B"], image_metas[0]["T"] = B, T
        inputs = self.pack_sequence(inputs)  # move from  B, T, ... -> B*T, ...

        inputs, outputs = self.encode_decode(inputs, image_metas)
        validity_mask = inputs["validity_mask"]

        # be careful on possible NaNs in reconstruced 3D (unprojection out-of-bound)
        pts_gt = inputs["camera"].reconstruct(inputs["depth"]) * validity_mask.float()
        pts_gt = torch.where(pts_gt.isnan().any(dim=1, keepdim=True), 0.0, pts_gt)
        mask_pts_gt_nan = ~pts_gt.isnan().any(dim=1, keepdim=True)
        mask = (
            inputs["depth_mask"].bool() & validity_mask.bool() & mask_pts_gt_nan.bool()
        )

        # compute loss!
        inputs["distance"] = torch.norm(pts_gt, dim=1, keepdim=True)
        inputs["points"] = pts_gt
        inputs["depth_mask"] = mask
        losses = self.compute_losses(outputs, inputs, image_metas)

        outputs = self.unpack_sequence(outputs, B, T)
        return (
            outputs,
            losses,
        )

    def forward_test(self, inputs, image_metas):
        B, T = inputs["image"].shape[:2]
        image_metas[0]["B"], image_metas[0]["T"] = B, T
        # move from  B, T, ... -> B*T, ...
        inputs = self.pack_sequence(inputs)
        inputs, outputs = self.encode_decode(inputs, image_metas)

        # you can add a dummy tensor with the actual output shape
        depth_gt = inputs["depth"]

        outs = {}
        outs["points"] = match_gt(
            outputs["points"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        outs["confidence"] = match_gt(
            outputs["confidence"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        outs["distance"] = outs["points"].norm(dim=1, keepdim=True)
        outs["depth"] = outs["points"][:, -1:]
        outs["rays"] = outs["points"] / torch.norm(
            outs["points"], dim=1, keepdim=True
        ).clip(min=1e-5)

        outs = self.unpack_sequence(outs, B, T)
        return outs

    def forward(self, inputs, image_metas):
        if self.training:
            return self.forward_train(inputs, image_metas)
        else:
            return self.forward_test(inputs, image_metas)

    def encode_decode(self, inputs, image_metas=[]):
        B, _, H, W = inputs["image"].shape

        # shortcut eval should avoid errors
        if len(image_metas) and "paddings" in image_metas[0]:
            # lrtb
            inputs["paddings"] = torch.tensor(
                [image_meta["paddings"] for image_meta in image_metas],
                device=self.device,
            )[..., [0, 2, 1, 3]]
            inputs["depth_paddings"] = torch.tensor(
                [image_meta["depth_paddings"] for image_meta in image_metas],
                device=self.device,
            )
            # at inference we do not have image paddings on top of depth ones (we have not "crop" on gt in ContextCrop)
            if self.training:
                inputs["depth_paddings"] = inputs["depth_paddings"] + inputs["paddings"]
            else:
                inputs["paddings"] = inputs["paddings"].squeeze(0)
                inputs["depth_paddings"] = inputs["depth_paddings"].squeeze(0)

        if inputs.get("camera", None) is not None:
            inputs["rays"] = inputs["camera"].get_rays(shapes=(B, H, W))

        features, tokens = self.pixel_encoder(inputs["image"])
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]

        outputs = self.pixel_decoder(inputs, image_metas)
        outputs["rays"] = rearrange(outputs["rays"], "b (h w) c -> b c h w", h=H, w=W)
        pts_3d = outputs["rays"] * outputs["distance"]
        outputs.update({"points": pts_3d, "depth": pts_3d[:, -1:]})

        return inputs, outputs

    def compute_losses(self, outputs, inputs, image_metas):
        B, _, H, W = inputs["image"].shape
        losses = {"opt": {}, "stat": {}}
        losses_to_be_computed = list(self.losses.keys())

        # depth loss
        si = torch.tensor(
            [x.get("si", False) for x in image_metas], device=self.device
        ).reshape(B)
        loss = self.losses["depth"]
        depth_losses = loss(
            outputs["distance"],
            target=inputs["distance"],
            mask=inputs["depth_mask"].clone(),
            si=si,
        )
        losses["opt"][loss.name] = loss.weight * depth_losses.mean()
        losses_to_be_computed.remove("depth")

        loss = self.losses["camera"]
        camera_losses = loss(
            outputs["rays"], target=inputs["rays"], mask=inputs["validity_mask"].bool()
        )
        losses["opt"][loss.name] = loss.weight * camera_losses.mean()
        losses_to_be_computed.remove("camera")

        # remaining losses, we expect no more losses to be computed
        loss = self.losses["confidence"]
        conf_losses = loss(
            outputs["confidence"],
            target_gt=inputs["depth"],
            target_pred=outputs["depth"],
            mask=inputs["depth_mask"].clone(),
        )
        # print(conf_losses, camera_losses, depth_losses)
        losses["opt"][loss.name + "_conf"] = loss.weight * conf_losses.mean()
        losses_to_be_computed.remove("confidence")

        assert (
            not losses_to_be_computed
        ), f"Losses {losses_to_be_computed} not computed, revise `compute_loss` method"

        return losses

    @torch.no_grad()
    def infer(
        self,
        rgb: torch.Tensor,
        camera: torch.Tensor | Camera | None = None,
        rays=None,
        normalize=True,
    ):
        ratio_bounds = self.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            self.shape_constraints["pixels_min"],
            self.shape_constraints["pixels_max"],
        ]
        if hasattr(self, "resolution_level"):
            assert (
                self.resolution_level >= 0 and self.resolution_level < 10
            ), "resolution_level should be in [0, 10)"
            pixels_range = pixels_bounds[1] - pixels_bounds[0]
            interval = pixels_range / 10
            new_lowbound = self.resolution_level * interval + pixels_bounds[0]
            new_upbound = (self.resolution_level + 1) * interval + pixels_bounds[0]
            pixels_bounds = (new_lowbound, new_upbound)
        else:
            warnings.warn("!! self.resolution_level not set, using default bounds !!")

        # houskeeping on cpu/cuda and batchify
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if camera is not None:
            camera = BatchCamera.from_camera(camera)
            camera = camera.to(self.device)

        B, _, H, W = rgb.shape
        rgb = rgb.to(self.device)

        # preprocess
        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor(
            (padded_H, padded_W), pixels_bounds
        )
        # -> rgb preprocess (input std-ized and resized)
        if normalize:
            rgb = TF.normalize(
                rgb.float() / 255.0,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(
            rgb, size=(new_H, new_W), mode="bilinear", align_corners=False
        )
        # -> camera preprocess
        if camera is not None:
            camera = camera.crop(
                left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom
            )
            camera = camera.resize(resize_factor)

        # prepare inputs
        inputs = {"image": rgb}
        if camera is not None:
            inputs["camera"] = camera
            rays = camera.get_rays(shapes=(B, new_H, new_W), noisy=False).reshape(
                B, 3, new_H, new_W
            )
            inputs["rays"] = rays

        if rays is not None:
            rays = rays.to(self.device)
            if rays.ndim == 3:
                rays = rays.unsqueeze(0)
            rays = F.pad(
                rays,
                (
                    max(0, pad_left),
                    max(0, pad_right),
                    max(0, pad_top),
                    max(0, pad_bottom),
                ),
                value=0.0,
            )
            rays = F.interpolate(
                rays, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
            inputs["rays"] = rays

        # run model
        _, model_outputs = self.encode_decode(inputs, image_metas={})

        # collect outputs
        out = {}
        out["confidence"] = _postprocess(
            model_outputs["confidence"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        points = _postprocess(
            model_outputs["points"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        rays = _postprocess(
            model_outputs["rays"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )

        out["distance"] = points.norm(dim=1, keepdim=True)
        out["depth"] = points[:, -1:]
        out["points"] = points
        out["rays"] = rays / torch.norm(rays, dim=1, keepdim=True).clip(min=1e-5)
        out["lowres_features"] = model_outputs["lowres_features"]
        return out

    def load_pretrained(self, model_file):
        dict_model = torch.load(model_file, map_location="cpu", weights_only=False)
        if "model" in dict_model:
            dict_model = dict_model["model"]
        info = self.load_state_dict(dict_model, strict=False)
        # if is_main_process():
        #     print(
        #         f"Loaded from {model_file} for {self.__class__.__name__} results in:",
        #         info,
        #     )

    def build(self, config):
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["model"]["pixel_encoder"],
            **config["data"],
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths
        config["model"]["pixel_encoder"]["cls_token_embed_dims"] = getattr(
            pixel_encoder, "cls_token_embed_dims", pixel_encoder_embed_dims
        )

        pixel_decoder = Decoder(config)

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = pixel_decoder
        # print("pixel_decoder", pixel_decoder)

        self.slices_encoder_range = list(
            zip([0, *self.pixel_encoder.depths[:-1]], self.pixel_encoder.depths)
        )
        self.stacking_fn = last_stack
        self.shape_constraints = config["data"]["shape_constraints"]
        self.interpolation_mode = "bilinear"

    def build_losses(self, config):
        self.losses = {}
        for loss_name, loss_config in config["training"]["losses"].items():
            mod = importlib.import_module("unik3d.ops.losses")
            loss_factory = getattr(mod, loss_config["name"])
            self.losses[loss_name] = loss_factory.build(loss_config)

    def get_params(self, config):
        if hasattr(self.pixel_encoder, "get_params"):
            encoder_p, _ = self.pixel_encoder.get_params(
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
                config["training"]["ld"],
            )
        else:
            encoder_p, _ = get_params(
                self.pixel_encoder,
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
            )
        decoder_p = self.pixel_decoder.get_params(
            config["training"]["lr"], config["training"]["wd"]
        )
        return [*encoder_p, *decoder_p]

    def step(self):
        self.pixel_decoder.steps += 1

    def parameters_grad(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    @property
    def device(self):
        return next(self.parameters()).device
