"""Implementation of MSCAN from SegNeXt: Rethinking Convolutional Attention Design for Semantic
Segmentation (NeurIPS 2022) adapted from

https://github.com/Visual-Attention-Network/SegNeXt/blob/main/mmseg/models/backbones/mscan.py


Light Hamburger Decoder adapted from:

https://github.com/Visual-Attention-Network/SegNeXt/blob/main/mmseg/models/decode_heads/ham_head.py
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

# flake8: noqa: E266
# mypy: ignore-errors


class ConvModule(nn.Module):
    """Replacement for mmcv.cnn.ConvModule to avoid mmcv dependency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        use_norm: bool = False,
        bias: bool = True,
    ):
        """Simple convolution block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size.
            padding (int, optional): Padding. Defaults to 0.
            use_norm (bool, optional): Whether to use normalization. Defaults to False.
            bias (bool, optional): Whether to use bias. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Simple residual convolution block.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features: int, unit2only=False, upsample=True):
        """Feature fusion block.

        Args:
            features (int): Number of features.
            unit2only (bool, optional): Whether to use only the second unit. Defaults to False.
            upsample (bool, optional): Whether to upsample. Defaults to True.
        """
        super().__init__()
        self.upsample = upsample

        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = xs[0]

        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)

        return output


###################################################
########### Light Hamburger Decoder ###############
###################################################


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization (NMF) for 2D data."""

    def __init__(self):
        """Non-negative Matrix Factorization (NMF) for 2D data."""
        super().__init__()
        self.S, self.D, self.R = 1, 512, 64
        self.train_steps = 6
        self.eval_steps = 7
        self.inv_t = 1

    def _build_bases(self, B: int, S: int, D: int, R: int, device: str = "cpu") -> torch.Tensor:
        if self.training or not torch.are_deterministic_algorithms_enabled():
            # Preserve GeoCalib's original RNG stream and values: sample on
            # CPU first, then transfer to the requested device.
            bases = torch.rand((B * S, D, R)).to(device)
        else:
            # GeoCalib's NMF decoder otherwise samples a different basis on
            # every inference call, making the estimated intrinsics depend on
            # unrelated global RNG state.  Use a private fixed generator in
            # eval mode without consuming the caller's RNG stream.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
            bases = torch.rand((B * S, D, R), generator=generator).to(device)
        return F.normalize(bases, dim=1)

    def local_step(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)
        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        return coef * numerator / (denominator + 1e-6)

    def local_inference(self, x: torch.Tensor, bases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local inference."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)

        # (S, D, R) -> (B * S, D, R)
        bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)
        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))
        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)
        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class Hamburger(nn.Module):
    """Hamburger Module."""

    def __init__(self, ham_channels: int = 512):
        """Hambuger Module.

        Args:
            ham_channels (int, optional): Number of channels in the hamburger module. Defaults to
            512.
        """
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1)
        self.ham = NMF2D()
        self.ham_out = ConvModule(ham_channels, ham_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=False)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=False)
        return ham


class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?

    This head is the implementation of `HamNet <https://arxiv.org/abs/2109.04553>`.
    """

    def __init__(self):
        """Light hamburger decoder head."""
        super().__init__()
        self.in_index = [0, 1, 2, 3]
        self.in_channels = [64, 128, 320, 512]
        self.out_channels = 64
        self.ham_channels = 512
        self.align_corners = False

        self.squeeze = ConvModule(sum(self.in_channels), self.ham_channels, 1)

        self.hamburger = Hamburger(self.ham_channels)

        self.align = ConvModule(self.ham_channels, self.out_channels, 1)

        self.linear_pred_uncertainty = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
        )

        self.out_conv = ConvModule(self.out_channels, self.out_channels, 3, padding=1, bias=False)
        self.ll_fusion = FeatureFusionBlock(self.out_channels, upsample=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        inputs = [features["hl"][i] for i in self.in_index]

        inputs = [
            F.interpolate(
                level,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        feats = self.align(x)

        assert "ll" in features, "Low-level features are required for this model"
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.out_conv(feats)
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.ll_fusion(feats, features["ll"].clone())

        uncertainty = self.linear_pred_uncertainty(feats).squeeze(1)

        return feats, uncertainty


###################################################
###########          MSCAN         ################
###################################################


class DWConv(nn.Module):
    """Depthwise convolution."""

    def __init__(self, dim: int = 768):
        """Depthwise convolution.

        Args:
            dim (int, optional): Number of features. Defaults to 768.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.dwconv(x)


class Mlp(nn.Module):
    """MLP module."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """Initialize the MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    """Simple stem convolution module."""

    def __init__(self, in_channels: int, out_channels: int):
        """Simple stem convolution module.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
        """
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    """Attention module."""

    def __init__(self, dim: int):
        """Attention module.

        Args:
            dim (int): Number of features.
        """
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        return attn * u


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, dim: int):
        """Spatial attention module.

        Args:
            dim (int): Number of features.
        """
        super().__init__()
        self.d_model = dim
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    """MSCAN block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
    ):
        """MSCAN block.

        Args:
            dim (int): Number of features.
            mlp_ratio (float, optional): Ratio of the hidden features in the MLP. Defaults to 4.0.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        """
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = nn.Identity()  # only used in training
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1[..., None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[..., None, None] * self.mlp(self.norm2(x)))
        return x.view(B, C, N).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """Image to Patch Embedding.

        Args:
            patch_size (int, optional): Image patch size. Defaults to 7.
            stride (int, optional): Stride. Defaults to 4.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 768.
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Forward pass."""
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN(nn.Module):
    """Multi-scale convolutional attention network."""

    def __init__(self):
        """Multi-scale convolutional attention network."""
        super().__init__()
        self.in_channels = 3
        self.embed_dims = [64, 128, 320, 512]
        self.mlp_ratios = [8, 8, 4, 4]
        self.drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.depths = [3, 3, 12, 3]
        self.num_stages = 4

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(3, self.embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=self.in_chans if i == 0 else self.embed_dims[i - 1],
                    embed_dim=self.embed_dims[i],
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=self.embed_dims[i],
                        mlp_ratio=self.mlp_ratios[i],
                        drop=self.drop_rate,
                    )
                    for _ in range(self.depths[i])
                ]
            )
            norm = nn.LayerNorm(self.embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, data):
        """Forward pass."""
        # rgb -> bgr and from [0, 1] to [0, 255]
        x = data["image"][:, [2, 1, 0], :, :] * 255.0
        B = x.shape[0]

        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return {"features": outs}
