# This file includes code originally from the PriorDA repository:
# https://github.com/SpatialVision/Prior-Depth-Anything
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from vipe.priors.depth.dav2.util.blocks import FeatureFusionBlock, _make_scratch

from .dinov2 import DINOv2
from .transform import NormalizeImage, Resize


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        encoder_cond_dim=-1,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.encoder_cond_dim = encoder_cond_dim

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

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        hido_feature = head_features_1 // 2
        hidi_feature = hido_feature

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, hido_feature, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(hidi_feature, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w, condition=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

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
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        encoder_cond_dim=-1,
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.encoder = encoder
        self.encoder_cond_dim = encoder_cond_dim
        self.pretrained = DINOv2(model_name=encoder)
        self.out_channels = features // 2

        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            encoder_cond_dim=encoder_cond_dim,
        )

    def forward(self, image, input_size=518, condition=None, device="cuda:0"):
        x, (h, w) = self.raw2input(image, input_size, device)

        rh, rw = x.shape[-2:]
        patch_h, patch_w = rh // 14, rw // 14

        if self.encoder_cond_dim > 0:
            condition = F.interpolate(condition, (rh, rw), mode="bilinear", align_corners=True)
        else:
            condition = None

        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
            condition=condition,
        )
        disparity = self.depth_head(features, patch_h, patch_w, condition=condition)

        disparity = F.relu(disparity).squeeze(1)
        disparity = F.interpolate(disparity[:, None], (h, w), mode="bilinear", align_corners=True)

        return disparity

    def freeze_network(self, names: dict):
        trainable_params = {"encoder": self.pretrained, "decoder": self.depth_head}

        for name in names:
            if name in trainable_params:
                for param in trainable_params[name].parameters():
                    param.requires_grad = False

    def init_state_dict(self, state_dict, **kwargs):
        missing, unexpected = super().load_state_dict(state_dict=state_dict, strict=True)

        self.depth_head.scratch.output_conv2 = nn.Sequential(
            self.depth_head.scratch.output_conv2, nn.ReLU(), nn.Identity()
        )
        if self.encoder_cond_dim > 0:
            self.pretrained.patch_embed.init_alpha_conv(cond_channels=self.encoder_cond_dim)

        if hasattr(self.depth_head.scratch.refinenet4, "resConfUnit1"):
            del self.depth_head.scratch.refinenet4.resConfUnit1
        if hasattr(self.pretrained, "mask_token"):
            del self.pretrained.mask_token

        return missing, unexpected

    def raw2input(self, raw_image, input_size=518, device="cuda"):
        assert isinstance(raw_image, torch.Tensor)
        assert raw_image.dtype == torch.uint8
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method="bicubic",
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device),
            ]
        )
        raw_image = raw_image.to(device)

        h, w = raw_image.shape[-2:]
        raw_image = raw_image / 255.0
        images = transform({"image": raw_image})["image"]
        return images, (h, w)
