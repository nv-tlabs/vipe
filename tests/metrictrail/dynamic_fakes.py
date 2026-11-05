# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU stand-ins for the dynamic-mask and depth-fusion tests.

- ``fake_net_module``/``fake_altcorr_module``: the vendored DPVO ``Patchifier``
  (``vipe/metrictrail/dpvo/net.py``) line for line, with ``device="cuda"`` replaced
  by the input's device and tiny convolutions for the two encoders, and a
  pure-torch ``altcorr.patchify`` (bilinear patches, as the CUDA kernel computes).
- :class:`MatrixSE3`: ViPE's lietorch ``SE3`` reduced to what the output stream
  and ``AdaptiveDepthProcessor`` call, on 4x4 camera-to-world matrices.
- :class:`MaskedStream`: :class:`metrictrail_fakes.FakeStream` frames carrying a
  ViPE ``frame.mask``.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrictrail_fakes import FakeStream

from vipe.streams.base import FrameAttribute, VideoFrame

FAKE_DIM = 16


def cpu_patchify(net: torch.Tensor, coords: torch.Tensor, radius: int, mode: str = "bilinear") -> torch.Tensor:
    """``altcorr.patchify`` on CPU: (n,C,H,W) features, (n,M,2) xy -> (n,M,C,2r+1,2r+1), zero outside."""

    n, channels, height, width = net.shape
    d = 2 * radius + 1
    floor = coords.floor()
    offsets = torch.arange(-radius, radius + 2, dtype=torch.long)
    x0 = floor[..., 0].long()[..., None] + offsets  # (n,M,d+1)
    y0 = floor[..., 1].long()[..., None] + offsets
    inside = (
        (x0[..., None, :] >= 0) & (x0[..., None, :] < width) & (y0[..., :, None] >= 0) & (y0[..., :, None] < height)
    )
    xs = x0.clamp(0, width - 1)
    ys = y0.clamp(0, height - 1)
    frames = torch.arange(n)[:, None, None, None]
    window = net.permute(0, 2, 3, 1)[frames, ys[..., :, None], xs[..., None, :]]  # (n,M,d+1,d+1,C)
    window = (window * inside[..., None]).permute(0, 1, 4, 2, 3)  # (n,M,C,d+1,d+1)
    if mode != "bilinear":
        return window[..., :d, :d]
    offset = coords - floor
    dx, dy = offset[:, :, None, None, None].unbind(dim=-1)
    return (
        (1 - dy) * (1 - dx) * window[..., :d, :d]
        + (1 - dy) * dx * window[..., :d, 1:]
        + dy * (1 - dx) * window[..., 1:, :d]
        + dy * dx * window[..., 1:, 1:]
    )


def coords_grid_with_index(d, **kwargs):
    """coordinate grid with frame index (vendored ``dpvo/utils.py``)"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)  # noqa: F841 - vendored line
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


class _Encoder4(nn.Module):
    """(b,n,3,H,W) -> (b,n,C,H/4,W/4), like DPVO's ``BasicEncoder4``."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, output_dim, kernel_size=4, stride=4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n = images.shape[:2]
        out = self.conv(images.reshape(b * n, *images.shape[2:]))
        return out.reshape(b, n, *out.shape[1:])


altcorr = types.SimpleNamespace(patchify=cpu_patchify)
DIM = FAKE_DIM


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = _Encoder4(128)
        self.inet = _Encoder4(DIM)

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat="RANDOM", return_color=False):
        """extract patches from input images"""
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size
        device = images.device

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == "GRADIENT_BIAS":
            g = self.__image_gradient(images)
            x = torch.randint(1, w - 1, size=[n, 3 * patches_per_image], device=device)
            y = torch.randint(1, h - 1, size=[n, 3 * patches_per_image], device=device)

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0, :, None], coords, 0).view(n, 3 * patches_per_image)

            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == "RANDOM":
            x = torch.randint(1, w - 1, size=[n, patches_per_image], device=device)
            y = torch.randint(1, h - 1, size=[n, patches_per_image], device=device)

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4 * (coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device=device)

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P // 2).view(b, -1, 3, P, P)

        index = torch.arange(n, device=device).view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


fake_net_module = types.SimpleNamespace(Patchifier=Patchifier, DIM=DIM, coords_grid_with_index=coords_grid_with_index)
fake_altcorr_module = altcorr


class FakeDino(nn.Module):
    """``FrozenDinoV3``'s interface: (B,N,3,H,W) -> (B,N,C,ceil(H/16),ceil(W/16))."""

    def __init__(self, feature_dim: int = 8, patch_size: int = 16) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, feature_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n, _, height, width = images.shape
        flat = images.reshape(b * n, 3, height, width)
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
        out = self.proj(F.avg_pool2d(flat, self.patch_size))
        return out.reshape(b, n, *out.shape[1:])


def make_patchifier(seed: int = 0, adapter_dim: int = 4) -> nn.Module:
    """A DINO-fused patchifier on the fake DPVO modules, with nonzero (seeded) fusion weights."""

    from vipe.metrictrail.dino_fusion import make_dino_patchifier_class

    cls = make_dino_patchifier_class(fake_net_module, fake_altcorr_module)
    generator = torch.Generator().manual_seed(seed)
    patchifier = cls(FakeDino(), patch_size=3, adapter_dim=adapter_dim)
    with torch.no_grad():
        for parameter in patchifier.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return patchifier.eval()


def dpvo_images(height: int = 32, width: int = 48, seed: int = 3) -> torch.Tensor:
    """(1,1,3,H,W) in DPVO's normalization ``2 * image / 255 - 0.5``."""

    generator = torch.Generator().manual_seed(seed)
    u8 = torch.randint(0, 256, (1, 1, 3, height, width), generator=generator).float()
    return 2 * (u8 / 255.0) - 0.5


class MatrixSE3:
    """ViPE's ``SE3`` reduced to what the output stream and ``AdaptiveDepthProcessor`` call (4x4 matrices, CPU)."""

    def __init__(self, matrices: torch.Tensor) -> None:
        self.m = matrices

    @property
    def shape(self) -> torch.Size:
        return self.m.shape[:-2]

    def __len__(self) -> int:
        return int(self.m.shape[0])

    def __getitem__(self, index: Any) -> MatrixSE3:
        return MatrixSE3(self.m[index])

    def __mul__(self, other: MatrixSE3) -> MatrixSE3:
        return MatrixSE3(self.m @ other.m)

    def inv(self) -> MatrixSE3:
        return MatrixSE3(torch.linalg.inv(self.m))

    def matrix(self) -> torch.Tensor:
        return self.m

    def cpu(self) -> MatrixSE3:
        return self

    def cuda(self) -> MatrixSE3:
        return self


def dynamic_box(hw: tuple[int, int], index: int) -> torch.Tensor:
    """ViPE mask (True = static) with a dynamic box that moves with the frame index."""

    mask = torch.ones(hw, dtype=torch.bool)
    top = 4 + index % 3
    mask[top : top + hw[0] // 3, 8 + 2 * index : 8 + 2 * index + hw[1] // 3] = False
    return mask


class MaskedStream(FakeStream):
    """FakeStream frames carrying ``frame.mask`` (``mask_fn(hw, index)``) and the MASK attribute."""

    def __init__(self, count: int, *, hw: tuple[int, int], mask_fn=dynamic_box, **kwargs: Any) -> None:
        super().__init__(count, hw=hw, attributes={FrameAttribute.MASK}, **kwargs)
        self.mask_fn = mask_fn

    def __iter__(self):
        for index, frame in enumerate(super().__iter__()):
            yield VideoFrame(raw_frame_idx=frame.raw_frame_idx, rgb=frame.rgb, mask=self.mask_fn(self.hw, index))


class CpuTorch:
    """``torch`` with ``device=`` dropped from ``ones``/``zeros`` (ViPE's ``project_map`` allocates on "cuda")."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def ones(*args: Any, device: Any = None, **kwargs: Any) -> torch.Tensor:
        return torch.ones(*args, **kwargs)

    @staticmethod
    def zeros(*args: Any, device: Any = None, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(*args, **kwargs)
