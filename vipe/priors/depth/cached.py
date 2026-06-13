import logging

import numpy as np
import torch
import torch.nn.functional as F

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

log = logging.getLogger(__name__)


class CachedDepthModel(DepthEstimationModel):
    """
    Pre-computed per-frame metric depths loaded from .npz cache.
    Feeds Pi3X+MoGe-2 fused depths directly into SLAM BA loop via frame_idx lookup.
    Cache format: depths (T, H_orig, W_orig) float32 in metres.
    """

    def __init__(self, cache_path: str):
        d = np.load(cache_path)
        self._depths = d["depths"].astype(np.float32)  # (T, H, W)
        log.info(f"[CachedDepthModel] Loaded {len(self._depths)} frames from {cache_path}")

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        idx = src.frame_idx
        if idx is None or idx < 0 or idx >= len(self._depths):
            raise ValueError(f"[CachedDepthModel] Invalid frame_idx={idx}, cache has {len(self._depths)} frames")

        depth_np = self._depths[idx]  # (H_orig, W_orig)
        depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        target_h, target_w = src.rgb.shape[-3], src.rgb.shape[-2]
        if depth.shape[-2] != target_h or depth.shape[-1] != target_w:
            depth = F.interpolate(depth, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Return (1, H, W) to match VIPE's expected metric_depth shape
        return DepthEstimationResult(metric_depth=depth.squeeze(0))
