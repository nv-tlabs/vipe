"""VIPE-compatible depth backend: Pi3X (consistent) + MoGe-2 (metric scale).

Paper App. B.1 protocol:
  1. Pi3X infers (T,H,W) long-sequence-consistent relative depth.
  2. MoGe-2 infers (T,H,W) per-frame metric depth.
  3. EMA-momentum-0.99 closed-form scale fusion (App. B.1) gives per-frame
     scale s_t that converts Pi3X depth to metric.

Integration: drop into vipe/priors/depth/ and register in vipe/priors/depth/__init__.py
as make_depth_model("pi3x_moge2") -> Pi3XMoGe2DepthModel().

Install prerequisites:
  pip install git+https://github.com/yyfz/Pi3.git   # Pi3X code
  hf download yyfz233/Pi3X --local-dir <weights>    # Pi3X weights
  pip install git+https://github.com/microsoft/MoGe.git  # MoGe code
  hf download Ruicheng/moge-2-vitl-normal --local-dir <weights>  # MoGe-2 weights
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def _fuse_ema(d_consistent: np.ndarray, d_metric: np.ndarray,
              momentum: float = 0.99) -> np.ndarray:
    """Closed-form EMA scale fusion (App. B.1).

    s_t = Σ(w · a · b) / Σ(w · a²),  w = 1/a  (inverse-depth weighting)
    where a = d_consistent[t], b = d_metric[t].
    """
    T = d_consistent.shape[0]
    scale = np.ones(T, dtype=np.float32)
    ema_s = None
    for t in range(T):
        a = d_consistent[t].flatten().astype(np.float64)
        b = d_metric[t].flatten().astype(np.float64)
        valid = (a > 1e-6) & (b > 1e-6) & np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 16:
            s_t = ema_s if ema_s is not None else 1.0
        else:
            av, bv = a[valid], b[valid]
            w = 1.0 / np.clip(av, 1e-6, None)
            s_t = float(np.sum(w * av * bv) / np.sum(w * av * av))
        ema_s = s_t if ema_s is None else momentum * ema_s + (1 - momentum) * s_t
        scale[t] = float(ema_s)
    return scale


class Pi3XMoGe2DepthModel(DepthEstimationModel):
    """Pi3X + MoGe-2 fused depth for SANA-WM (App. B.1).

    Environment variables:
      SANA_WM_PI3X_WEIGHTS  — path to yyfz233/Pi3X local_dir (required)
      SANA_WM_MOGE2_WEIGHTS — path to Ruicheng/moge-2-vitl-normal local_dir (required)
    """

    def __init__(self, device: str = "cuda", ema_momentum: float = 0.99) -> None:
        super().__init__()
        self.device = device
        self.ema_momentum = ema_momentum
        self._pi3x: Optional[object] = None
        self._moge2: Optional[object] = None
        self._video_buffer: list[np.ndarray] = []

    def _lazy_load(self) -> None:
        if self._pi3x is not None:
            return

        pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS")
        moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS")

        if pi3x_weights is None or moge2_weights is None:
            raise RuntimeError(
                "Set SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS env vars "
                "to the local weight directories before using Pi3XMoGe2DepthModel."
            )

        try:
            from pi3 import Pi3X  # type: ignore
            self._pi3x = Pi3X.from_pretrained(pi3x_weights).to(self.device).eval()
        except ImportError as e:
            raise RuntimeError(
                "Pi3X (pip install git+https://github.com/yyfz/Pi3.git) not found."
            ) from e

        try:
            from moge.model.v2 import MoGeModel  # type: ignore
            import pathlib
            moge2_path = pathlib.Path(moge2_weights)
            moge2_ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
            self._moge2 = MoGeModel.from_pretrained(str(moge2_ckpt)).to(self.device).eval()
        except ImportError as e:
            raise RuntimeError(
                "MoGe (pip install git+https://github.com/microsoft/MoGe.git) not found."
            ) from e

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Process a single frame (VIPE calls this per-frame during SLAM).

        Pi3X requires the full video context for sequence-consistent depth.
        We buffer all frames and run Pi3X in batch on the first call after the
        video is complete (video_frame_list is not None).

        For per-frame SLAM keyframe calls (rgb only), we fall back to MoGe-2
        metric depth directly (Pi3X sequence context not available yet).
        """
        self._lazy_load()

        # Video-sequence mode (post-SLAM depth alignment): process full clip.
        if src.video_frame_list is not None:
            return self._estimate_video(src)

        # Per-frame mode (SLAM keyframe): use MoGe-2 only.
        return self._estimate_single(src)

    @torch.no_grad()
    def _estimate_single(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = unpack_optional(src.rgb).to(self.device)
        if rgb.dim() == 3:
            rgb = rgb[None]
            was_unbatched = True
        else:
            was_unbatched = False
        # MoGe-2 inference — rgb is (B, H, W, 3) here
        inp = rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)
        fov_x = None
        if src.intrinsics is not None:
            fx = src.intrinsics[0].item()
            w = rgb.shape[2]
            import math
            fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
        out = self._moge2.infer(inp, fov_x=fov_x)  # type: ignore[union-attr]
        depth = out["depth"]  # (B, H, W)
        if was_unbatched:
            depth = depth.squeeze(0)  # (H, W) only if we added the batch dim ourselves
        return DepthEstimationResult(metric_depth=depth)

    @torch.no_grad()
    def _estimate_video(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Run Pi3X (chunked) + MoGe-2 (per-frame) and fuse."""
        frames = src.video_frame_list  # list of (H,W,3) float32 [0,1]
        assert frames is not None
        T = len(frames)

        import torch.nn.functional as F_nn

        frames_np = np.stack(frames, axis=0)  # (T,H,W,3)
        frames_t = torch.from_numpy(frames_np).to(self.device).permute(0, 3, 1, 2)  # (T,3,H,W)
        H_img, W_img = frames_np.shape[1], frames_np.shape[2]

        # Pi3X requires H and W to be multiples of patch size 14
        H_r = (H_img // 14) * 14
        W_r = (W_img // 14) * 14
        if H_r != H_img or W_r != W_img:
            frames_pi3x = F_nn.interpolate(frames_t, size=(H_r, W_r), mode='bilinear', align_corners=False)
        else:
            frames_pi3x = frames_t

        # Pi3X forward: (B, N, 3, H, W) -> local_points (B, N, H, W, 3), depth = z-component
        # Process in chunks of 16 (ViT attention is O(N^2 x patches^2) — full 600+ frames OOM)
        CHUNK, STRIDE = 16, 8
        d_pi3x_accum = np.zeros((T, H_r, W_r), dtype=np.float32)
        count = np.zeros(T, dtype=np.float32)
        starts = list(range(0, max(T - CHUNK + 1, 1), STRIDE))
        if not starts or starts[-1] + CHUNK < T:
            starts.append(max(0, T - CHUNK))
        for s in starts:
            e = min(s + CHUNK, T)
            chunk = frames_pi3x[s:e].unsqueeze(0)  # (1, n, 3, H_r, W_r)
            out = self._pi3x(chunk)  # type: ignore[union-attr]
            d_chunk = out["local_points"][0, :e-s, :, :, 2].cpu().numpy()  # (n, H_r, W_r)
            d_pi3x_accum[s:e] += d_chunk
            count[s:e] += 1
        d_pi3x_r = d_pi3x_accum / np.maximum(count[:, None, None], 1.0)  # (T, H_r, W_r)

        # Upsample Pi3X depth back to original resolution
        if H_r != H_img or W_r != W_img:
            d_pi3x = F_nn.interpolate(
                torch.from_numpy(d_pi3x_r).unsqueeze(1).to(self.device),
                size=(H_img, W_img), mode='bilinear', align_corners=False
            ).squeeze(1).cpu().numpy()
        else:
            d_pi3x = d_pi3x_r

        # MoGe-2: per-frame metric depth
        fov_x = None
        if src.intrinsics is not None:
            fx = src.intrinsics[0].item()
            w = frames_np.shape[2]
            import math
            fov_x = math.degrees(2 * math.atan(w / (2 * fx)))

        d_moge_list = []
        for i in range(T):
            f = frames_t[i:i+1]
            out = self._moge2.infer(f, fov_x=fov_x)  # type: ignore[union-attr]
            d_moge_list.append(out["depth"].squeeze(0).cpu().numpy())
        d_moge = np.stack(d_moge_list, axis=0)  # (T,H,W)

        # EMA scale fusion
        scale = _fuse_ema(d_pi3x, d_moge, self.ema_momentum)  # (T,)
        metric_depth = d_pi3x * scale[:, None, None]

        return DepthEstimationResult(
            metric_depth=torch.from_numpy(metric_depth.astype(np.float32)).to(self.device)
        )
