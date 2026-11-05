# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only symmetric DINOv3 matching features for stereo DPVO.

The trained DINO-fused Patchifier augments the left-image correlation feature
raster in ``forward()``. Stereo DPVO obtains the right-image correlation target
through ``feature_only()``; because the DINO class does not override that method,
the right raster would use only the base DPVO encoder. This module installs a
class-method override that adds the same trained matching residual to the right
raster without adding parameters or changing checkpoint keys. The context branch
is intentionally absent because the right image is a correlation target only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

PATCH_CONTRACT_VERSION = 1
PATCH_OWNER = "metrictrail_dino_stereo_feature_only_v1"


def _matching_augmented_feature_only(
    base_feature_only: Callable[[Any, torch.Tensor], torch.Tensor],
    resize_fn: Callable[..., torch.Tensor],
) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Build the checkpoint-compatible symmetric ``feature_only`` method.

    Dependency injection keeps the numerical contract CPU-testable without
    importing DPVO's CUDA extensions. Production passes the exact base
    ``Patchifier.feature_only`` and DINO raster-alignment function.
    """

    def feature_only(self: Any, images: torch.Tensor) -> torch.Tensor:
        base = base_feature_only(self, images)
        if base.ndim != 5 or base.shape[2] != 128:
            raise ValueError(f"base DPVO feature_only must return [B,N,128,H,W], got {tuple(base.shape)}")
        if not bool(getattr(self, "fuse_matching", False)):
            raise RuntimeError("DINO stereo repair requires fuse_matching=True")
        match_fusion = getattr(self, "match_fusion", None)
        if match_fusion is None:
            raise RuntimeError("DINO stereo repair requires the trained match_fusion")

        batch, frames, _, height, width = base.shape
        dino_raw = self.dino(images)
        if dino_raw.ndim != 5 or dino_raw.shape[:2] != (batch, frames):
            raise ValueError(f"DINO must return [B,N,C,H,W] aligned with right images, got {tuple(dino_raw.shape)}")
        dino_low = self.dino_projection(
            dino_raw.reshape(
                batch * frames,
                dino_raw.shape[2],
                dino_raw.shape[3],
                dino_raw.shape[4],
            )
        )
        dino_low = resize_fn(
            dino_low,
            original_hw=(int(images.shape[-2]), int(images.shape[-1])),
            target_hw=(height, width),
            foundation_patch_size=int(self.dino.patch_size),
        ).reshape(batch, frames, -1, height, width)

        adapter_dim = int(self.adapter_dim)
        if adapter_dim <= 0 or dino_low.shape[2] < adapter_dim:
            raise ValueError("projected DINO raster cannot supply the matching slice")
        # ``forward`` assigns the first adapter slice to matching whenever
        # fuse_matching=True.  Keep exactly that ordering here.
        dino_match = dino_low[:, :, :adapter_dim]
        delta = match_fusion(dino_match.reshape(batch * frames, adapter_dim, height, width)).reshape(
            batch, frames, 128, height, width
        )
        if delta.dtype != base.dtype:
            delta = delta.to(dtype=base.dtype)
        return base + delta

    feature_only.__name__ = "dinov3_symmetric_stereo_feature_only"
    feature_only.__qualname__ = "DinoAugmentedPatchifier.dinov3_symmetric_stereo_feature_only"
    setattr(feature_only, "_dinov3_stereo_patch_owner", PATCH_OWNER)
    return feature_only


def install_symmetric_stereo_feature_only(
    *,
    dino_patchifier_class: type,
    base_patchifier_class: Any,
    resize_fn: Callable[..., torch.Tensor],
) -> type:
    """Install the method override on the loaded DINO class and return it."""

    current = getattr(dino_patchifier_class, "feature_only", None)
    if getattr(current, "_dinov3_stereo_patch_owner", None) == PATCH_OWNER:
        return dino_patchifier_class
    # Fail closed if somebody has already installed a different instance-level
    # repair.  An inherited base implementation is the expected precondition.
    if "feature_only" in dino_patchifier_class.__dict__:
        raise RuntimeError("DinoAugmentedPatchifier already defines an unreviewed feature_only override")
    method = _matching_augmented_feature_only(
        base_patchifier_class.feature_only,
        resize_fn,
    )
    setattr(dino_patchifier_class, "feature_only", method)
    setattr(dino_patchifier_class, "_symmetric_stereo_patch_version", PATCH_CONTRACT_VERSION)
    return dino_patchifier_class


def install_stereo_patch(
    network: Any,
    *,
    base_patchifier_class: type | None = None,
    resize_fn: Callable[..., torch.Tensor] | None = None,
) -> type:
    """Give the loaded DINO Patchifier a matching-augmented ``feature_only``.

    ``base_patchifier_class`` defaults to the vendored stereo DPVO's ``Patchifier``
    and ``resize_fn`` to :func:`vipe.metrictrail.dino_fusion.resize_dino_to_dpvo_raster`
    (the function the left raster uses). Returns the patched class.
    """

    if base_patchifier_class is None:
        from .dpvo_stereo import net as stereo_net

        base_patchifier_class = stereo_net.Patchifier
        if base_patchifier_class.__module__ != stereo_net.__name__:
            raise RuntimeError(
                f"stereo Patchifier came from {base_patchifier_class.__module__}, expected {stereo_net.__name__}"
            )
    if resize_fn is None:
        from vipe.metrictrail.dino_fusion import resize_dino_to_dpvo_raster

        resize_fn = resize_dino_to_dpvo_raster
    augmented_class = type(network.patchify)
    if base_patchifier_class not in augmented_class.__mro__:
        raise RuntimeError("loaded DINO Patchifier is not based on configured DPVO Patchifier")
    installed = install_symmetric_stereo_feature_only(
        dino_patchifier_class=augmented_class,
        base_patchifier_class=base_patchifier_class,
        resize_fn=resize_fn,
    )
    if installed is not augmented_class:
        raise RuntimeError("stereo patch did not target the loaded augmented Patchifier")
    owner = getattr(augmented_class.feature_only, "_dinov3_stereo_patch_owner", None)
    if owner != PATCH_OWNER:
        raise RuntimeError("loaded DINO Patchifier lacks the stereo feature repair")
    return augmented_class


__all__ = [
    "PATCH_CONTRACT_VERSION",
    "PATCH_OWNER",
    "install_stereo_patch",
    "install_symmetric_stereo_feature_only",
]
