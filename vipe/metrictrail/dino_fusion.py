# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen DINOv3 features fused into DPVO's patchifier (DINO-DPVO).

The frozen DINOv3 ViT-S/16 is ViPE's in-tree DINOv3
(``vipe.priors.depth.dap.dino.make_dinov3_backbone("vits")``), not a DINOv3 hub
checkout. Its ``get_intermediate_layers(x, n=1, reshape=False,
norm=True)[0]`` equals the hub model's ``forward_features(x)["x_norm_patchtokens"]``
bit for bit with the same weights (checked on CPU, float32). The fusion
(residual concat/1x1 fusion of matching and context features) is built on the
vendored DPVO's ``Patchifier``, with the optional centroid restriction of ViPE's
dynamic masking (``centroid_valid``, :mod:`vipe.metrictrail.dynamic`); unset, the
forward is DPVO's own, with the same random draws.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: DINOv3 hub model names the DINO-DPVO checkpoint may name -> ViPE backbone name.
VIPE_DINOV3_BACKBONES = {"dinov3_vits16": "vits"}


def resize_dino_to_dpvo_raster(
    features: torch.Tensor,
    *,
    original_hw: tuple[int, int],
    target_hw: tuple[int, int],
    foundation_patch_size: int,
    dpvo_stride: int = 4,
) -> torch.Tensor:
    """Align padded foundation tokens to the unpadded DPVO feature raster.

    DINO sees a right/bottom-padded image while DPVO sees the original image.
    Resizing the full token map directly to the smaller DPVO raster would
    compress the padded region into valid pixels.  Instead, first resize in
    the padded coordinate system and then crop away the padded rows/columns.
    """

    if features.ndim != 4:
        raise ValueError("DINO features must have shape [BN,C,H,W]")
    if foundation_patch_size <= 0 or dpvo_stride <= 0:
        raise ValueError("Feature strides must be positive")
    if foundation_patch_size % dpvo_stride:
        raise ValueError("DINO patch size must be divisible by DPVO stride")
    original_height, original_width = map(int, original_hw)
    target_height, target_width = map(int, target_hw)
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError("Raster dimensions must be positive")

    padded_height = features.shape[-2] * foundation_patch_size
    padded_width = features.shape[-1] * foundation_patch_size
    expected_padded_height = ((original_height + foundation_patch_size - 1) // foundation_patch_size) * (
        foundation_patch_size
    )
    expected_padded_width = ((original_width + foundation_patch_size - 1) // foundation_patch_size) * (
        foundation_patch_size
    )
    if (padded_height, padded_width) != (expected_padded_height, expected_padded_width):
        raise ValueError("DINO token raster does not match the padded image size")

    padded_target = (padded_height // dpvo_stride, padded_width // dpvo_stride)
    if target_height > padded_target[0] or target_width > padded_target[1]:
        raise ValueError("DPVO raster is larger than the padded DINO raster")
    resized = F.interpolate(features, size=padded_target, mode="bilinear", align_corners=False)
    return resized[..., :target_height, :target_width]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DINOv3PatchTokens(nn.Module):
    """ViPE's DINOv3 ViT with the hub model's ``forward_features`` patch-token output."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = int(getattr(backbone, "embed_dim"))
        self.patch_size = int(getattr(backbone, "patch_size"))

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.backbone.get_intermediate_layers(images, n=1, reshape=False, norm=True)[0]  # type: ignore[operator]
        return {"x_norm_patchtokens": tokens}


def make_vipe_dinov3_backbone(model_name: str) -> DINOv3PatchTokens:
    """ViPE's in-tree DINOv3 for the hub model ``model_name`` (randomly initialised; load weights next)."""

    from vipe.priors.depth.dap.dino import make_dinov3_backbone

    name = VIPE_DINOV3_BACKBONES.get(model_name)
    if name is None:
        raise ValueError(f"no ViPE DINOv3 backbone for {model_name!r}; supported: {sorted(VIPE_DINOV3_BACKBONES)}")
    return DINOv3PatchTokens(make_dinov3_backbone(name))


class FrozenDinoV3(nn.Module):
    """Frozen, dense DINOv3 feature extractor on ViPE's in-tree DINOv3.

    DPVO's training readers expose BGR images while the official DINOv3
    LVD-1689M weights use RGB and the ImageNet evaluation normalization.  The
    input to this module is DPVO-normalized (the range produced by
    ``2 * image / 255 - 0.5``); conversion is deliberately kept here so the
    two frontends cannot silently disagree about channel order or scale.
    """

    rgb_mean: torch.Tensor
    rgb_std: torch.Tensor

    def __init__(
        self,
        *,
        weights: str | Path,
        model_name: str = "dinov3_vits16",
        feature_dim: int = 384,
        patch_size: int = 16,
        frame_chunk_size: int = 3,
        amp_dtype: str = "bfloat16",
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights are missing: {weights}")
        if expected_sha256 and sha256_file(weights) != expected_sha256:
            raise ValueError("DINOv3 checkpoint SHA256 mismatch")
        if frame_chunk_size <= 0:
            raise ValueError("frame_chunk_size must be positive")
        if amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported DINO amp_dtype: {amp_dtype}")

        self.backbone = make_vipe_dinov3_backbone(model_name)
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only support.
            state = torch.load(weights, map_location="cpu")
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise TypeError("DINOv3 checkpoint must be a plain state_dict")
        self.backbone.backbone.load_state_dict(state, strict=True)
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.frame_chunk_size = int(frame_chunk_size)
        self.amp_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[amp_dtype]
        actual_dim = self.backbone.embed_dim
        actual_patch = self.backbone.patch_size
        if actual_dim != self.feature_dim:
            raise ValueError(f"DINOv3 feature_dim mismatch: model={actual_dim}, config={self.feature_dim}")
        if actual_patch != self.patch_size:
            raise ValueError(f"DINOv3 patch_size mismatch: model={actual_patch}, config={self.patch_size}")
        self.register_buffer(
            "rgb_mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "rgb_std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenDinoV3":
        super().train(mode)
        # The adapter is trainable, but the foundation backbone is not.
        self.backbone.eval()
        return self

    def _extract_chunk(self, images: torch.Tensor) -> torch.Tensor:
        output: Any = self.backbone.forward_features(images)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise TypeError("The configured DINOv3 backbone must expose forward_features()['x_norm_patchtokens']")
        tokens = output["x_norm_patchtokens"]
        rows = images.shape[-2] // self.patch_size
        cols = images.shape[-1] // self.patch_size
        if tokens.shape != (images.shape[0], rows * cols, self.feature_dim):
            raise ValueError(
                "Unexpected DINOv3 patch-token shape: "
                f"{tuple(tokens.shape)} != {(images.shape[0], rows * cols, self.feature_dim)}"
            )
        return tokens.transpose(1, 2).reshape(images.shape[0], self.feature_dim, rows, cols)

    def forward(self, dpvo_images: torch.Tensor) -> torch.Tensor:
        """Return dense features as ``[B,N,C,H/patch,W/patch]``."""

        if dpvo_images.ndim != 5 or dpvo_images.shape[2] != 3:
            raise ValueError("DINOv3 expects DPVO images [B,N,3,H,W]")
        batch, frames, _, height, width = dpvo_images.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("Images are smaller than the DINOv3 patch size")

        # Undo DPVO's affine normalization, then BGR -> RGB.
        images = ((dpvo_images + 0.5) * 0.5).flip(2).clamp_(0.0, 1.0)
        images = images.reshape(batch * frames, 3, height, width)
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        if pad_height or pad_width:
            # Padding is isolated to DINO; DPVO's image raster and K are never
            # resized or cropped by the feature augmentation.
            images = F.pad(images, (0, pad_width, 0, pad_height), mode="replicate")
        images = (images - self.rgb_mean) / self.rgb_std
        chunks: list[torch.Tensor] = []
        autocast_enabled = images.is_cuda and self.amp_dtype != torch.float32
        with torch.no_grad():
            for chunk in images.split(self.frame_chunk_size, dim=0):
                with torch.autocast(device_type=chunk.device.type, dtype=self.amp_dtype, enabled=autocast_enabled):
                    chunks.append(self._extract_chunk(chunk))
        features = torch.cat(chunks, dim=0).float()
        return features.reshape(
            batch,
            frames,
            self.feature_dim,
            (height + pad_height) // self.patch_size,
            (width + pad_width) // self.patch_size,
        )


def make_dino_patchifier_class(net_module: Any | None = None, altcorr_module: Any | None = None) -> type:
    """Build the DINO-fused Patchifier on top of the vendored DPVO.

    ``net_module``/``altcorr_module`` default to ``vipe.metrictrail.dpvo.net`` and
    ``vipe.metrictrail.dpvo.altcorr``. The class body is the trained DINO-DPVO
    fusion, unchanged.
    """

    if net_module is None:
        net_module = importlib.import_module("vipe.metrictrail.dpvo.net")
    if altcorr_module is None:
        altcorr_module = importlib.import_module("vipe.metrictrail.dpvo.altcorr")
    Patchifier = net_module.Patchifier
    DIM = net_module.DIM
    altcorr = altcorr_module
    coords_grid_with_index = net_module.coords_grid_with_index

    class DinoAugmentedPatchifier(Patchifier):  # type: ignore[misc,valid-type]
        """Official DPVO patchifier with residual concat/1x1 DINOv3 fusion.

        Both matching and context paths keep their original output dimensions, so
        the official CorrBlock and Update checkpoint remain usable.  Zero-initial
        residual projections make step zero exactly equivalent to pretrained
        DPVO.  The base-feature columns of the conceptual concat/1x1 operator are
        a fixed identity, so Phase A cannot improve via a DINO-independent adapter.
        """

        def __init__(
            self,
            dino: FrozenDinoV3,
            patch_size: int = 3,
            adapter_dim: int = 32,
            fuse_matching: bool = True,
            fuse_context: bool = True,
        ) -> None:
            super().__init__(patch_size=patch_size)
            if not (fuse_matching or fuse_context):
                raise ValueError("At least one DINOv3 fusion path must be enabled")
            self.dino = dino
            dino_dim = dino.feature_dim
            self.adapter_dim = int(adapter_dim)
            if self.adapter_dim <= 0:
                raise ValueError("adapter_dim must be positive")
            self.fuse_matching = bool(fuse_matching)
            self.fuse_context = bool(fuse_context)
            output_dim = self.adapter_dim * (int(fuse_matching) + int(fuse_context))
            self.dino_projection = nn.Conv2d(dino_dim, output_dim, kernel_size=1)
            # ``base + Conv1x1(dino)`` is exactly a concat/1x1 fusion whose base
            # columns are fixed to identity.  Keeping those columns non-trainable
            # makes a branch-only warmup attributable to the DINO features.
            self.match_fusion = nn.Conv2d(self.adapter_dim, 128, kernel_size=1) if fuse_matching else None
            self.context_fusion = nn.Linear(self.adapter_dim, DIM) if fuse_context else None
            if self.match_fusion is not None:
                nn.init.zeros_(self.match_fusion.weight)
                nn.init.zeros_(self.match_fusion.bias)  # type: ignore[arg-type]
            if self.context_fusion is not None:
                nn.init.zeros_(self.context_fusion.weight)
                nn.init.zeros_(self.context_fusion.bias)

        @staticmethod
        def _snapshot_patch_coordinates(
            patches: torch.Tensor,
            *,
            frames: int,
            patches_per_image: int,
            center: int,
        ) -> torch.Tensor:
            """Return immutable XY sampling coordinates for CUDA patchify.

            ``patches`` is returned to the DPVO forward, whose official
            ``set_depth()`` implementation updates the disparity channel in place.
            A plain XY slice still shares the parent tensor's version counter, so
            that later disparity-only update invalidates coordinates saved by
            ``PatchLayer`` for backward. Coordinates are not differentiable in the
            CUDA operator, therefore an owning detached snapshot preserves the
            exact forward values while isolating autograd bookkeeping.
            """

            coordinates = patches[0, :, :2, center, center]
            expected = frames * patches_per_image
            if coordinates.shape != (expected, 2):
                raise ValueError(
                    f"Unexpected DPVO patch-coordinate shape: {tuple(coordinates.shape)} != {(expected, 2)}"
                )
            return coordinates.reshape(frames, patches_per_image, 2).detach().clone()

        #: Centroid validity (h,w bool on the 1/4 feature grid) for the current call, set per frame by
        #: MetricTrail's dynamic masking (:mod:`vipe.metrictrail.dynamic`). None: DPVO's own draw, unchanged.
        centroid_valid: torch.Tensor | None = None

        def _masked_base_forward(
            self,
            images: torch.Tensor,
            patches_per_image: int,
            disps: torch.Tensor | None,
            centroid_sel_strat: str,
            return_color: bool,
            valid: torch.Tensor,
        ):
            """The vendored ``Patchifier.forward`` with every centroid drawn on a ``valid`` cell.

            Line for line the vendored DPVO forward, except that the uniform
            ``torch.randint`` centroid draws are rejection-sampled to ``valid``
            (:func:`vipe.metrictrail.dynamic.sample_valid_centroids`).
            """

            from .dynamic import sample_valid_centroids

            fmap = self.fnet(images) / 4.0
            imap = self.inet(images) / 4.0

            b, n, c, h, w = fmap.shape
            P = self.patch_size
            if tuple(valid.shape) != (h, w):
                raise ValueError(f"centroid validity {tuple(valid.shape)} does not match the DPVO grid {(h, w)}")
            valid = valid.to(device=fmap.device)

            if centroid_sel_strat == "GRADIENT_BIAS":
                g = getattr(self, "_Patchifier__image_gradient")(images)
                x, y = sample_valid_centroids(valid, n, 3 * patches_per_image)

                coords = torch.stack([x, y], dim=-1).float()
                g = altcorr.patchify(g[0, :, None], coords, 0).view(n, 3 * patches_per_image)

                ix = torch.argsort(g, dim=1)
                x = torch.gather(x, 1, ix[:, -patches_per_image:])
                y = torch.gather(y, 1, ix[:, -patches_per_image:])

            elif centroid_sel_strat == "RANDOM":
                x, y = sample_valid_centroids(valid, n, patches_per_image)

            else:
                raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

            coords = torch.stack([x, y], dim=-1).float()
            imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
            gmap = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)

            if return_color:
                clr = altcorr.patchify(images[0], 4 * (coords + 0.5), 0).view(b, -1, 3)

            if disps is None:
                disps = torch.ones(b, n, h, w, device=fmap.device)

            grid, _ = coords_grid_with_index(disps, device=fmap.device)
            patches = altcorr.patchify(grid[0], coords, P // 2).view(b, -1, 3, P, P)

            index = torch.arange(n, device=fmap.device).view(n, 1)
            index = index.repeat(1, patches_per_image).reshape(-1)

            if return_color:
                return fmap, gmap, imap, patches, index, clr

            return fmap, gmap, imap, patches, index

        def forward(
            self,
            images: torch.Tensor,
            patches_per_image: int = 80,
            disps: torch.Tensor | None = None,
            centroid_sel_strat: str = "RANDOM",
            return_color: bool = False,
        ):
            if self.centroid_valid is None:
                base = super().forward(
                    images,
                    patches_per_image=patches_per_image,
                    disps=disps,
                    centroid_sel_strat=centroid_sel_strat,
                    return_color=return_color,
                )
            else:
                base = self._masked_base_forward(
                    images, patches_per_image, disps, centroid_sel_strat, return_color, self.centroid_valid
                )
            if return_color:
                fmap, base_gmap, imap, patches, index, color = base
            else:
                fmap, base_gmap, imap, patches, index = base
            # gmap must be resampled after matching features are augmented.
            del base_gmap

            batch, frames, _, height, width = fmap.shape
            dino_raw = self.dino(images)
            dino_low = self.dino_projection(
                dino_raw.reshape(-1, dino_raw.shape[2], dino_raw.shape[3], dino_raw.shape[4])
            )
            dino_low = resize_dino_to_dpvo_raster(
                dino_low,
                original_hw=(images.shape[-2], images.shape[-1]),
                target_hw=(height, width),
                foundation_patch_size=self.dino.patch_size,
            ).reshape(batch, frames, -1, height, width)
            split = list(dino_low.split(self.adapter_dim, dim=2))
            dino_match = split.pop(0) if self.fuse_matching else None
            dino_context = split.pop(0) if self.fuse_context else None

            if self.match_fusion is not None:
                assert dino_match is not None
                match_delta = self.match_fusion(dino_match.reshape(batch * frames, -1, height, width)).reshape(
                    batch, frames, 128, height, width
                )
                fmap = fmap + match_delta

            center = self.patch_size // 2
            coordinates = self._snapshot_patch_coordinates(
                patches,
                frames=frames,
                patches_per_image=patches_per_image,
                center=center,
            )
            gmap = altcorr.patchify(fmap[0], coordinates, center).view(batch, -1, 128, self.patch_size, self.patch_size)
            if self.context_fusion is not None:
                assert dino_context is not None
                dino_patches = altcorr.patchify(dino_context[0], coordinates, 0).view(batch, -1, self.adapter_dim)
                context = imap[..., 0, 0]
                context_delta = self.context_fusion(dino_patches)
                imap = (context + context_delta).unsqueeze(-1).unsqueeze(-1)

            if return_color:
                return fmap, gmap, imap, patches, index, color
            return fmap, gmap, imap, patches, index

    DinoAugmentedPatchifier.__module__ = __name__
    DinoAugmentedPatchifier.__qualname__ = "DinoAugmentedPatchifier"
    return DinoAugmentedPatchifier


def dino_from_config(config: dict[str, Any]) -> FrozenDinoV3:
    if not config.get("weights"):
        raise ValueError("DINOv3 config is missing: ['weights']")
    return FrozenDinoV3(
        weights=config["weights"],
        model_name=str(config.get("model_name", "dinov3_vits16")),
        feature_dim=int(config.get("feature_dim", 384)),
        patch_size=int(config.get("patch_size", 16)),
        frame_chunk_size=int(config.get("frame_chunk_size", 3)),
        amp_dtype=str(config.get("amp_dtype", "bfloat16")),
        expected_sha256=config.get("weights_sha256"),
    )
