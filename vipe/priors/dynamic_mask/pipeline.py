"""End-to-end dynamic-mask estimation: flow -> consistency -> Sampson -> per-instance scoring."""

from dataclasses import dataclass, field

import numpy as np
import torch

from tqdm import tqdm

from ..flow.base import OpticalFlowModel
from .flow_consistency import forward_backward_consistency
from .sampson import sampson_error_for_frame


@dataclass
class DynamicMaskOutput:
    masks: torch.Tensor  # (S, H, W) bool — final dynamic mask per frame
    sampson: torch.Tensor  # (S, H, W) float — Sampson error per pixel per frame
    fwd_consistency: torch.Tensor  # (S, H, W) bool — forward consistency mask
    bwd_consistency: torch.Tensor  # (S, H, W) bool — backward consistency mask
    instance_motion_scores: dict[int, float] = field(default_factory=dict)
    selected_instances: list[int] = field(default_factory=list)


class DynamicMaskPipeline:
    """Computes per-frame dynamic-object masks from RGB frames + per-frame
    instance segmentations (with consistent IDs across frames).

    Args:
        flow_model: an `OpticalFlowModel` to compute pairwise optical flow.
        flow_alpha, flow_beta: parameters of the fwd/bwd consistency check.
        ransac_threshold: RANSAC reprojection threshold for the fundamental
            matrix fit (in normalized [-1, 1] coords).
        keep_motion_ratio: an instance is dynamic iff its motion score exceeds
            `keep_motion_ratio * max(motion_scores)`.
        min_area_ratio: drop instances smaller than this fraction of the image.
        filter_time_thres: per-frame Sampson errors below this are zeroed before
            averaging across time, so frames where an object is momentarily
            static do not lower its score.
        sampson_max_points: cap on points fed to RANSAC.
        device: torch device for flow + tensors.
    """

    def __init__(
        self,
        flow_model: OpticalFlowModel,
        flow_alpha: float = 0.5,
        flow_beta: float = 0.5,
        ransac_threshold: float = 0.01,
        keep_motion_ratio: float = 0.25,
        min_area_ratio: float = 1e-4,
        filter_time_thres: float = 1e-4,
        sampson_max_points: int = 10000,
        device: torch.device | str = "cuda",
    ) -> None:
        self.flow_model = flow_model
        self.flow_alpha = flow_alpha
        self.flow_beta = flow_beta
        self.ransac_threshold = ransac_threshold
        self.keep_motion_ratio = keep_motion_ratio
        self.min_area_ratio = min_area_ratio
        self.filter_time_thres = filter_time_thres
        self.sampson_max_points = sampson_max_points
        self.device = torch.device(device)

    @torch.no_grad()
    def _compute_pairwise_flow(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: (S, 3, H, W) in [0, 1]. Returns fwd, bwd flow each of shape
        (S - 1, 2, H, W)."""
        S = frames.shape[0]
        fwds = []
        bwds = []
        for i in tqdm(range(S - 1), desc="SEA-RAFT pairwise flow"):
            a = frames[i : i + 1].to(self.device)
            b = frames[i + 1 : i + 2].to(self.device)
            fwds.append(self.flow_model.estimate(a, b).cpu())
            bwds.append(self.flow_model.estimate(b, a).cpu())
        fwd = torch.cat(fwds, dim=0)
        bwd = torch.cat(bwds, dim=0)
        return fwd, bwd

    @torch.no_grad()
    def _compute_consistency_masks(
        self, fwd_flow: torch.Tensor, bwd_flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fwd_flow, bwd_flow: (S - 1, 2, H, W). Returns:

        per_frame_fwd_mask: (S, H, W) bool — pixels in frame i whose forward
            flow to frame i+1 is consistent (last frame = all True).
        per_frame_bwd_mask: (S, H, W) bool — pixels in frame i whose backward
            flow to frame i-1 is consistent (first frame = all True).
        """
        # Process in chunks to bound peak GPU mem.
        S_minus_1 = fwd_flow.shape[0]
        H, W = fwd_flow.shape[-2:]
        fwd_masks = torch.empty((S_minus_1, H, W), dtype=torch.bool)
        bwd_masks_pair = torch.empty((S_minus_1, H, W), dtype=torch.bool)
        chunk = 8
        for s in range(0, S_minus_1, chunk):
            e = min(s + chunk, S_minus_1)
            f_a, f_b = forward_backward_consistency(
                fwd_flow[s:e].to(self.device),
                bwd_flow[s:e].to(self.device),
                alpha=self.flow_alpha,
                beta=self.flow_beta,
            )
            fwd_masks[s:e] = f_a.cpu()
            bwd_masks_pair[s:e] = f_b.cpu()

        # Pad to (S, H, W).
        per_frame_fwd_mask = torch.cat(
            [fwd_masks, torch.ones((1, H, W), dtype=torch.bool)], dim=0
        )
        per_frame_bwd_mask = torch.cat(
            [torch.ones((1, H, W), dtype=torch.bool), bwd_masks_pair], dim=0
        )
        return per_frame_fwd_mask, per_frame_bwd_mask

    @torch.no_grad()
    def _compute_sampson_errors(
        self,
        fwd_flow: torch.Tensor,
        bwd_flow: torch.Tensor,
        per_frame_fwd_mask: torch.Tensor,
        per_frame_bwd_mask: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Returns (S, H, W) Sampson-error tensor on CPU (float32)."""
        S = per_frame_fwd_mask.shape[0]
        fwd_flow_np = fwd_flow.permute(0, 2, 3, 1).numpy()  # (S-1, H, W, 2)
        bwd_flow_np = bwd_flow.permute(0, 2, 3, 1).numpy()
        fwd_mask_np = per_frame_fwd_mask.numpy()
        bwd_mask_np = per_frame_bwd_mask.numpy()

        out = np.zeros((S, H, W), dtype=np.float32)
        for i in tqdm(range(S), desc="Sampson per frame"):
            fwd_f = fwd_flow_np[i] if i < S - 1 else None
            fwd_m = fwd_mask_np[i] if i < S - 1 else None
            bwd_f = bwd_flow_np[i - 1] if i > 0 else None
            bwd_m = bwd_mask_np[i] if i > 0 else None
            out[i] = sampson_error_for_frame(
                fwd_flow=fwd_f,
                fwd_mask=fwd_m,
                bwd_flow=bwd_f,
                bwd_mask=bwd_m,
                H=H,
                W=W,
                max_points=self.sampson_max_points,
                use_ransac=True,
                ransac_threshold=self.ransac_threshold,
            )
        return torch.from_numpy(out)

    def _compute_motion_scores(
        self,
        sampson: torch.Tensor,
        per_frame_fwd_mask: torch.Tensor,
        per_frame_bwd_mask: torch.Tensor,
        instances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate the per-pixel Sampson error to a per-instance motion score.

        Args:
            sampson: (S, H, W) float CPU.
            per_frame_fwd_mask, per_frame_bwd_mask: (S, H, W) bool CPU.
            instances: (S, H, W) int CPU. 0 = background.

        Returns:
            obj_ids: (O,) long, the unique non-zero instance ids.
            motion_scores: (O,) float, average Sampson error per object.
        """
        unique_ids = torch.unique(instances)
        unique_ids = unique_ids[unique_ids != 0]

        if unique_ids.numel() == 0:
            return unique_ids, torch.zeros(0)

        flow_mask = per_frame_fwd_mask & per_frame_bwd_mask  # (S, H, W)

        s, H, W = sampson.shape
        device = self.device

        # We chunk over objects (and over time) to bound memory.
        scores = torch.zeros(len(unique_ids), dtype=torch.float32)

        sampson_g = sampson.to(device)
        flow_mask_g = flow_mask.to(device)
        instances_g = instances.to(device)

        for k, obj_id in enumerate(unique_ids.tolist()):
            obj_mask = (instances_g == obj_id) & flow_mask_g  # (S, H, W) bool
            denom = obj_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
            per_frame_err = (sampson_g * obj_mask.float()).sum(dim=(1, 2)) / denom  # (S,)
            valid = per_frame_err >= self.filter_time_thres
            n_valid = valid.float().sum().clamp(min=1.0)
            scores[k] = (per_frame_err * valid.float()).sum().item() / n_valid.item()

        return unique_ids, scores

    @torch.no_grad()
    def compute(
        self,
        frames: torch.Tensor,
        instances: torch.Tensor,
    ) -> DynamicMaskOutput:
        """Run the full pipeline.

        Args:
            frames: (S, 3, H, W) float in [0, 1].
            instances: (S, H, W) int — instance map per frame, 0 = background,
                instance ids must be consistent across frames.

        Returns:
            `DynamicMaskOutput`.
        """
        S, _, H, W = frames.shape
        assert instances.shape == (S, H, W), f"instances shape mismatch: {instances.shape}"

        fwd_flow, bwd_flow = self._compute_pairwise_flow(frames)
        per_frame_fwd_mask, per_frame_bwd_mask = self._compute_consistency_masks(fwd_flow, bwd_flow)
        sampson = self._compute_sampson_errors(
            fwd_flow, bwd_flow, per_frame_fwd_mask, per_frame_bwd_mask, H, W
        )

        # Drop tiny instances by area before scoring.
        unique_ids = torch.unique(instances)
        unique_ids = unique_ids[unique_ids != 0]
        keep_ids = []
        total_pixels = float(S * H * W)
        for obj_id in unique_ids.tolist():
            area = float((instances == obj_id).sum().item())
            if area / total_pixels >= self.min_area_ratio:
                keep_ids.append(obj_id)
        if not keep_ids:
            return DynamicMaskOutput(
                masks=torch.zeros((S, H, W), dtype=torch.bool),
                sampson=sampson,
                fwd_consistency=per_frame_fwd_mask,
                bwd_consistency=per_frame_bwd_mask,
            )

        keep_instances = torch.zeros_like(instances)
        for new_id, old_id in enumerate(keep_ids, start=1):
            keep_instances[instances == old_id] = new_id

        obj_ids, motion_scores = self._compute_motion_scores(
            sampson, per_frame_fwd_mask, per_frame_bwd_mask, keep_instances
        )

        # Map obj_ids (re-numbered 1..K) back to original ids.
        new_to_old = {new_id: old_id for new_id, old_id in enumerate(keep_ids, start=1)}
        score_dict = {new_to_old[int(i)]: float(s) for i, s in zip(obj_ids.tolist(), motion_scores.tolist())}

        # Threshold-based selection.
        if motion_scores.numel() == 0 or motion_scores.max().item() <= 0.0:
            selected = []
        else:
            thr = motion_scores.max().item() * self.keep_motion_ratio
            selected_new_ids = [int(obj_ids[i].item()) for i in range(len(obj_ids)) if motion_scores[i].item() > thr]
            selected = [new_to_old[i] for i in selected_new_ids]

        if selected:
            sel_set = set(selected)
            sel_keep_idx = [i for i, oid in enumerate(keep_ids, start=1) if oid in sel_set]
            sel_mask = torch.zeros((S, H, W), dtype=torch.bool)
            for idx in sel_keep_idx:
                sel_mask |= keep_instances == idx
        else:
            sel_mask = torch.zeros((S, H, W), dtype=torch.bool)

        return DynamicMaskOutput(
            masks=sel_mask,
            sampson=sampson,
            fwd_consistency=per_frame_fwd_mask,
            bwd_consistency=per_frame_bwd_mask,
            instance_motion_scores=score_dict,
            selected_instances=selected,
        )
