# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StereoDPVOSystem: MetricTrail's stereo tracker on a ViPE left/right stream pair with a known K.

The stereo tracking flow:

1. DPVO configuration: the stereo preset (``dpvo``) merged over the stereo DPVO
   tree's defaults, then ``BUFFER_SIZE = buffer_size``
   (:func:`~vipe.metrictrail.stereo.config.resolve_stereo_dpvo_config`).
2. Models: the depth model exactly as the mono tracker builds it
   (``DPVOSLAMSystem.depth_model``, MoGe-2 in FP16) and the DINO-DPVO checkpoint
   (step 2's checks) on the stereo tree's ``VONet``, whose DINO Patchifier gets the
   symmetric stereo ``feature_only`` (:mod:`~vipe.metrictrail.stereo.features`).
3. Per sequence: seed = ``seed`` + the sequence name; RNGs reset; a fresh stereo
   DPVO at the input raster (a raster that is not divisible by 4 is refused: the
   reflect-padded transport cannot carry the right image); the left frames go
   through step 2's producer (:func:`vipe.metrictrail.depth_producer.track_sequence`,
   ``depth_queue``) and the per-frame preparation of the mono tracker
   (``DPVOSLAMSystem._prepare``); :class:`~vipe.metrictrail.stereo.transport.StereoSlamProxy`
   adds the right frame and the baseline to every DPVO call. A trajectory with a
   non-finite pose row is re-tracked by the mono tracker's FP32 recovery
   (``nonfinite_recovery``, on by default;
   :func:`vipe.metrictrail.slam.track_with_nonfinite_recovery`: precision only, the cached
   network released and a freshly loaded one used, at the per-sequence seed + 0, 1, 2, 3,
   the first all-finite trajectory kept, every pass recorded).
4. Output: step 2's ``DPVOSLAMOutput`` (left-camera c2w, the known K, identity rig,
   DPVO's patch points as the ``SLAMMap``) with the stereo tracking record, including
   its ``runtime`` block (:mod:`vipe.metrictrail.runtime_record`).

Poses are OpenCV camera-to-world of the left camera in DPVO's world frame (the
first frame is the identity).
"""

from __future__ import annotations

import gc
import inspect
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vipe.metrictrail.depth_producer import CudaOps, PreparedFrame, poses7_xyzw_to_c2w, track_sequence
from vipe.metrictrail.runtime_record import TrackingRuntime
from vipe.metrictrail.slam import (
    DPVOSLAMOutput,
    DPVOSLAMSystem,
    NetworkSlot,
    TrackingAttempt,
    apply_tf32_policy,
    attempt_valid,
    case_seed,
    reset_random_state,
    slam_map_from_keyframe_points,
)
from vipe.metrictrail.transport import dpvo_transport_padding
from vipe.slam.interface import SLAMMap
from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils.model_cache import ModelCache

from .config import StereoRunnerConfig, resolve_stereo_dpvo_config
from .features import PATCH_OWNER, install_stereo_patch
from .transport import RightFrames, StereoSlamProxy

logger = logging.getLogger(__name__)

TRACKING_SCHEMA = "metrictrail_vipe_stereo_tracking_v1"
#: Arguments the stereo DPVO must accept (metric depth API + rectified stereo).
STEREO_DPVO_API = ("metric_depth", "metric_mask", "right_image", "stereo_baseline_m")


def bind_cuda_device(device: torch.device) -> None:
    """Make ``device`` the current CUDA device (``torch.cuda.set_device``).

    DPVO allocates its state on the current device (``device="cuda"``, ``network.cuda()``),
    so ``--device cuda:1`` must also make cuda:1 current; a bare ``cuda`` keeps the current one.
    """

    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)


def load_stereo_dino_dpvo_network(path: Path, *, dinov3_weights: Path) -> torch.nn.Module:
    """The stereo tree's ``VONet`` with the DINO-fused Patchifier and the DINO-DPVO weights.

    Step 2's :func:`vipe.metrictrail.checkpoint.load_dino_dpvo_network` on the stereo
    tree: the same checkpoint checks (``checkpoint_payload``) and the same fusion class
    (``make_dino_patchifier_class``), built on the stereo ``Patchifier`` and ``altcorr``.
    """

    from vipe.metrictrail.checkpoint import checkpoint_payload
    from vipe.metrictrail.dino_fusion import dino_from_config, make_dino_patchifier_class

    from .dpvo_stereo import altcorr as stereo_altcorr
    from .dpvo_stereo import net as stereo_net

    value, dino = checkpoint_payload(path, dinov3_weights=dinov3_weights)
    patchifier_class = make_dino_patchifier_class(net_module=stereo_net, altcorr_module=stereo_altcorr)
    network: Any = stereo_net.VONet()
    network.patchify = patchifier_class(
        dino_from_config(dino),
        patch_size=int(network.P),
        adapter_dim=int(dino.get("adapter_dim", 32)),
        fuse_matching=True,
        fuse_context=True,
    )
    state = OrderedDict((str(key).removeprefix("module."), tensor) for key, tensor in value["model"].items())
    missing, unexpected = network.load_state_dict(state, strict=False)
    invalid_missing = [name for name in missing if not name.startswith("patchify.dino.backbone.")]
    if invalid_missing or unexpected:
        raise RuntimeError(f"DINO-DPVO state mismatch: missing={invalid_missing}, unexpected={unexpected}")
    network._metrictrail_dino_enabled = True
    network._metrictrail_dino_matching = True
    network._metrictrail_dino_context = True
    return network


def stereo_slam_map(dpvo: Any) -> SLAMMap | None:
    """Step 2's ``dpvo_slam_map`` on the stereo tree (its own ``projective_ops`` and lietorch ``SE3``)."""

    from .dpvo_stereo import projective_ops as pops
    from .dpvo_stereo.lietorch import SE3 as DPVOSE3

    n, m, patches_per_frame = int(dpvo.n), int(dpvo.m), int(dpvo.M)
    if n == 0 or m != n * patches_per_frame:
        return None
    with torch.inference_mode():
        points = pops.point_cloud(DPVOSE3(dpvo.poses), dpvo.patches[:, :m], dpvo.intrinsics, dpvo.ix[:m])
        disparity = points[..., 1, 1, 3:]
        xyz = (points[..., 1, 1, :3] / disparity).reshape(n, patches_per_frame, 3).float()
        valid = (torch.isfinite(xyz).all(dim=-1) & (disparity.reshape(n, patches_per_frame) > 0)).reshape(
            n, patches_per_frame
        )
        rgb = dpvo.pg.colors_[:n].float().div(255.0)
    ordinals = np.asarray(dpvo.pg.tstamps_[:n], dtype=np.int64)
    return slam_map_from_keyframe_points(xyz.clone(), rgb.clone(), valid.clone(), ordinals)


class StereoDPVOSystem(DPVOSLAMSystem):
    """MetricTrail's stereo tracker: left/right ViPE streams, a known pinhole K and a metric baseline.

    The per-frame left preparation (``_prepare``) and the depth model
    (``depth_model``) are the mono tracker's, unchanged: they read only the config
    fields that ``StereoRunnerConfig`` shares with ``MetricTrailSLAMConfig``.
    """

    config: StereoRunnerConfig  # type: ignore[assignment]
    recovery_option = "nonfinite_recovery"

    def __init__(self, device: torch.device, config: Any, model_cache: ModelCache | None = None) -> None:
        # DPVOSLAMSystem.__init__ validates the mono slam schema; the stereo runner has its own.
        self.device = torch.device(device)
        self.config = config if isinstance(config, StereoRunnerConfig) else StereoRunnerConfig.model_validate(config)
        self.model_cache = model_cache if model_cache is not None else ModelCache()
        self.ops = CudaOps(self.device)

    # ------------------------------------------------------------------ models

    def dpvo_network_cache_key(self) -> str:
        checkpoint, dinov3 = self._network_files()
        return f"metrictrail/stereo/dino_dpvo/{checkpoint.resolve()}|{dinov3.resolve()}"

    def fresh_dpvo_network(self) -> torch.nn.Module:
        """The stereo DINO-DPVO network loaded anew from its checkpoint (the model cache's builder)."""

        checkpoint, dinov3 = self._network_files()
        network = load_stereo_dino_dpvo_network(checkpoint, dinov3_weights=dinov3)
        return network.to(self.device).eval()

    def recovery_dpvo_network(self) -> torch.nn.Module:
        """The FP32 recovery's network: loaded anew, outside the model cache, with the stereo feature patch."""

        network = self.fresh_dpvo_network()
        install_stereo_patch(network)
        return network

    def dpvo_network(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        from vipe.metrictrail.checkpoint import checkpoint_identity

        checkpoint, dinov3 = self._network_files()
        network = self.model_cache.get(self.dpvo_network_cache_key(), self.fresh_dpvo_network)
        install_stereo_patch(network)
        identity = {"dpvo": checkpoint_identity(checkpoint), "dinov3": checkpoint_identity(dinov3)}
        return network, identity

    def dpvo_class(self) -> Any:
        from .dpvo_stereo.dpvo import DPVO

        return DPVO

    # ------------------------------------------------------------ per sequence

    def run(  # type: ignore[override]
        self,
        left: VideoStream,
        right: VideoStream,
        *,
        intrinsics: np.ndarray,
        baseline_m: float,
        name: str | None = None,
    ) -> DPVOSLAMOutput:
        """Track ``left`` with stereo factors from ``right`` (every frame, in order) and the known K."""

        from vipe.ext.lietorch import SE3
        from vipe.utils.geometry import se3_matrix_to_se3

        from ..depth import MetricTrailMoGe2

        started = time.perf_counter()
        name = left.name() if name is None else name
        count = len(left)
        if count <= 0:
            raise ValueError(f"stream {name!r} has no frames")
        if len(right) != count:
            raise ValueError(f"stereo needs one right frame per left frame: left={count}, right={len(right)}")
        image_hw = (int(left.frame_size()[0]), int(left.frame_size()[1]))
        if tuple(int(value) for value in right.frame_size()) != image_hw:
            raise ValueError(f"rectified stereo images must have the same size: {image_hw} vs {right.frame_size()}")
        baseline = float(baseline_m)
        if not np.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("stereo requires positive baseline and right frames")
        k4 = np.asarray(intrinsics, dtype=np.float32).reshape(4)
        if not np.isfinite(k4).all() or np.any(k4[:2] <= 0):
            raise ValueError(f"DPVO intrinsics must be finite [fx,fy,cx,cy] with positive focal: {k4}")
        if any(dpvo_transport_padding(image_hw)):
            raise ValueError(f"stereo tracking needs a raster divisible by 4, got {image_hw}")
        dpvo_class = self.dpvo_class()
        missing = set(STEREO_DPVO_API) - set(inspect.signature(dpvo_class.__call__).parameters)
        if missing:
            raise RuntimeError(f"vendored stereo DPVO does not expose {sorted(missing)}")
        cfg, resolved = resolve_stereo_dpvo_config(self.config, frame_count=count)
        tf32 = apply_tf32_policy(self.config.tf32)
        bind_cuda_device(self.device)
        depth_model = self.depth_model()
        network, identities = self.dpvo_network()
        # This frame keeps no reference of its own: the FP32 recovery must be able to free it.
        canonical_network = NetworkSlot(network, evict=self.evict_dpvo_network)
        del network
        base_seed = case_seed(name, int(self.config.seed)) if self.config.seed is not None else None
        k_device = torch.from_numpy(k4.copy()).to(self.device)

        def attempt(attempt_cfg: Any, attempt_network: Any, seed: int | None) -> TrackingAttempt:
            """Track the whole sequence once with a fresh stereo DPVO (the right stream is read again)."""

            if seed is not None:
                reset_random_state(seed)
            source_ids: list[int] = []
            dpvo = dpvo_class(attempt_cfg, attempt_network, ht=image_hw[0], wd=image_hw[1], viz=False)
            slam: Any = StereoSlamProxy(dpvo, RightFrames(right, image_hw), baseline_m=baseline, device=self.device)

            def prepare(index: int, frame: VideoFrame, pinned: bool, _ids: list[int] = source_ids) -> PreparedFrame:
                return self._prepare(
                    index,
                    frame,
                    pinned,
                    depth_model=depth_model,
                    k4=k4,
                    k_device=k_device,
                    image_hw=image_hw,
                    source_ids=_ids,
                )

            try:
                candidate = track_sequence(
                    iter(left),
                    count,
                    prepare,
                    slam,
                    k4,
                    ops=self.ops,
                    depth_queue=self.config.depth_queue,
                    metric_prior_warmup=int(attempt_cfg.METRIC_DEPTH_PRIOR_WARMUP),
                )
                valid, bad_rows = attempt_valid(candidate, count=count)
                result = TrackingAttempt(
                    seed=seed, tracked=candidate, valid=valid, bad_rows=bad_rows, source_ids=source_ids
                )
                if valid:
                    result.slam_map = self.slam_map(dpvo)
                return result
            finally:
                del slam, dpvo
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Tracking stage only: the models are loaded; the CUDA peak statistics restart here.
        runtime = TrackingRuntime(self.device, frames=count)
        runtime.start()
        selected, nonfinite_recovery = self.track_passes(
            attempt,
            cfg=cfg,
            canonical_network=canonical_network,
            base_seed=base_seed,
            runtime=runtime,
            name=name,
        )
        runtime_block = runtime.stop(passes=1 + len(nonfinite_recovery.get("attempts", ())))
        tracked = selected.tracked
        slam_map = selected.slam_map
        source_ids = selected.source_ids

        c2w = poses7_xyzw_to_c2w(tracked["poses7_xyzw"])
        if c2w.shape != (count, 4, 4) or not np.isfinite(c2w).all():
            raise ValueError("DINO-DPVO returned an invalid trajectory")
        describe = getattr(depth_model, "describe", None)
        record: dict[str, Any] = {
            "schema": TRACKING_SCHEMA,
            "status": "success",
            "case": name,
            "num_frames": count,
            "input_hw": list(image_hw),
            "intrinsics_fx_fy_cx_cy": k4.tolist(),
            "intrinsics_source": "known",
            "depth_provider_K_equals_dpvo_K": True,
            "tracking_mode": "stereo",
            "stereo_factors_used": True,
            "stereo_baseline_m": baseline,
            "stereo_feature_patch": PATCH_OWNER,
            "pose_camera_frame": "cam0",
            "dinov3_matching": True,
            "dinov3_context": True,
            "metric_depth_fastba": True,
            "runtime_resize_or_stretch": False,
            "dpvo_transport": {"source_hw": list(image_hw), "padding_ltrb": [0, 0, 0, 0]},
            "rgb_normalization": self.config.rgb_normalization,
            "depth_provider": describe()
            if callable(describe) and isinstance(depth_model, MetricTrailMoGe2)
            else {"registry_name": self.config.keyframe_depth, "depth_type": depth_model.depth_type.name},
            "dpvo_buffer_size": int(cfg.BUFFER_SIZE),
            "dpvo_buffer_size_config": self.config.buffer_size,
            "tf32_flags": tf32,
            "tracking_seed": self.config.seed,
            "case_seed": base_seed,
            "nonfinite_recovery": nonfinite_recovery,
            "tracking_status": tracked["tracking_status"],
            "dpvo_initialized": tracked["dpvo_initialized"],
            "metric_prior_status": tracked["metric_prior_status"],
            "metric_prior_diagnostics": tracked["metric_prior_diagnostics"],
            "timing": tracked["timing"],
            "execution": tracked["execution"],
            "valid_depth_fractions": tracked["valid_depth_fractions"],
            "resolved_dpvo_cfg": resolved,
            "checkpoint_identities": identities,
            "loop_closure": False,
            "global_ba": False,
            "dpvo_native_loop_disabled": True,
            "runtime": runtime_block,
            "runtime_seconds": time.perf_counter() - started,
        }
        record["end_to_end_fps"] = count / record["runtime_seconds"] if record["runtime_seconds"] > 0 else None

        trajectory = se3_matrix_to_se3(torch.from_numpy(c2w).float().to(self.device))
        return DPVOSLAMOutput(
            trajectory=trajectory,
            intrinsics=torch.from_numpy(k4)[None].to(self.device),
            rig=SE3.Identity(1, device=self.device),
            slam_map=slam_map,
            ba_residual=0.0,
            c2w=c2w,
            source_frame_ids=np.asarray(source_ids, dtype=np.int64),
            record=record,
        )

    def slam_map(self, dpvo: Any) -> SLAMMap | None:
        return stereo_slam_map(dpvo)
