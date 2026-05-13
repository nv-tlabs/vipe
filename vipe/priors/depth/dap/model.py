# SPDX-FileCopyrightText: Copyright (c) 2025 Insta360 Research Team
# SPDX-License-Identifier: MIT
#
# This file contains the DAP inference modules needed by ViPE. It is adapted
# from https://github.com/Insta360-Research-Team/DAP with training, dataset,
# visualization, and command-line code removed.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import DINOv3Adapter


def _make_scratch(in_shape: list[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool) -> None:
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        del deconv
        super().__init__()
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features // 2 if self.expand else features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))

        output = self.resConfUnit2(output)
        modifier = {"size": size if size is not None else self.size} if (size or self.size) else {"scale_factor": 2}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _make_fusion_block(features: int, use_bn: bool, size: tuple[int, int] | None = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: list[int] | None = None,
        use_clstoken: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or [256, 512, 1024, 1024]
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(
        self,
        out_features: tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], ...],
        patch_h: int,
        patch_w: int,
        patch_size: int = 16,
    ) -> torch.Tensor:
        out = []
        for idx, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]  # type: ignore[index]
                readout = cls_token.unsqueeze(1).expand_as(x) if x.dim() == 3 else None
                if readout is not None:
                    x = self.readout_projects[idx](torch.cat((x, readout), -1))
            else:
                x = x[0] if isinstance(x, (tuple, list)) else x

            if x.dim() == 3:
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            elif x.dim() == 4:
                if x.shape[-2] != patch_h or x.shape[-1] != patch_w:
                    x = F.interpolate(x, size=(patch_h, patch_w), mode="bilinear", align_corners=True)
            else:
                raise RuntimeError(f"Unexpected feature shape {x.shape}, expected 3D or 4D")

            x = self.projects[idx](x)
            x = self.resize_layers[idx](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * patch_size), int(patch_w * patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        return self.scratch.output_conv2(out)


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: list[int] | None = None,
        use_bn: bool = False,
        use_clstoken: bool = False,
        max_depth: float = 20.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or [256, 512, 1024, 1024]
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        self.max_depth = max_depth
        self.encoder = encoder
        self.pretrained = DINOv3Adapter(model_name=encoder)
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )
        self.mask_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )
        self.patch_size = int(self.pretrained.patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_size = getattr(self.pretrained, "patch_size", 16)
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        depth = self.depth_head(features, patch_h, patch_w, patch_size) * self.max_depth
        mask = self.mask_head(features, patch_h, patch_w, patch_size)
        return depth.squeeze(1), mask.squeeze(1)


class DAP(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        midas_model_type = args.midas_model_type
        self.max_depth = args.max_depth

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if midas_model_type not in model_configs:
            raise ValueError(f"Unsupported DAP model type '{midas_model_type}'. Expected one of {list(model_configs)}.")

        self.core = DepthAnythingV2(**{**model_configs[midas_model_type], "max_depth": 1.0})
        for param in self.core.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        erp_pred, mask_pred = self.core(image)
        erp_pred = erp_pred.unsqueeze(1)
        erp_pred[erp_pred < 0] = 0
        mask_pred = mask_pred.unsqueeze(1)
        return {
            "pred_depth": erp_pred * self.max_depth,
            "pred_mask": mask_pred,
        }

    def get_encoder_decoder_params(self) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        encoder_params = list(self.core.pretrained.parameters())
        decoder_params = list(self.core.depth_head.parameters())
        mask_params = list(self.core.mask_head.parameters())
        return encoder_params, decoder_params, mask_params


def make_dap_model(
    midas_model_type: str = "vitl",
    fine_tune_type: str = "none",
    min_depth: float = 0.001,
    max_depth: float = 1.0,
    train_decoder: bool = True,
) -> DAP:
    del fine_tune_type, min_depth, train_decoder
    args = Namespace()
    args.midas_model_type = midas_model_type
    args.max_depth = max_depth
    return DAP(args)
