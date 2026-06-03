# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import os
import logging
from pathlib import Path

import gdown
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from vipe.streams.base import VideoFrame
from vipe.utils.nvtx import nvtx_range

from .seg_tracker import SegTracker

LOGGER = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _resolve_checkpoint(
    *,
    cache_subdir: str,
    default_filename: str,
    default_url: str | None = None,
    default_gdrive_id: str | None = None,
    env_prefix: str,
) -> Path:
    explicit_path = os.environ.get(f"{env_prefix}_CHECKPOINT")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_prefix}_CHECKPOINT does not exist: {path}")
        return path

    hf_repo = os.environ.get(f"{env_prefix}_HF_REPO")
    hf_filename = os.environ.get(f"{env_prefix}_HF_FILENAME")
    if hf_repo or hf_filename:
        if not hf_repo or not hf_filename:
            raise ValueError(f"{env_prefix}_HF_REPO and {env_prefix}_HF_FILENAME must be set together")
        return Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                revision=os.environ.get(f"{env_prefix}_HF_REVISION"),
            )
        )

    url = os.environ.get(f"{env_prefix}_URL")
    if url:
        filename = Path(url.split("?", 1)[0]).name or default_filename
        path = Path(torch.hub.get_dir()) / cache_subdir / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, dst=str(path))
        return path

    path = Path(torch.hub.get_dir()) / cache_subdir / default_filename
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if default_url is not None:
        torch.hub.download_url_to_file(default_url, dst=str(path))
    elif default_gdrive_id is not None:
        gdown.download(id=default_gdrive_id, output=str(path))
    else:
        raise FileNotFoundError(
            f"No default checkpoint source is configured for {default_filename}. "
            f"Set {env_prefix}_CHECKPOINT or {env_prefix}_HF_REPO/{env_prefix}_HF_FILENAME."
        )
    return path


DEFAULT_AOT_CHECKPOINTS = {
    "r50_deaotl": ("R50_DeAOTL_PRE_YTB_DAV.pth", "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"),
    "deaotl": ("DeAOTL_PRE_YTB_DAV.pth", "18elNz_wi9JyVBcIUYKhRdL08MA-FqHD5"),
    "deaotb": ("DeAOTB_PRE_YTB_DAV.pth", "1BHxsonnvJXylqHlZ1zJHHc-ymKyq-CFf"),
    "deaots": ("DeAOTS_PRE_YTB_DAV.pth", "1YwIAV5tBtn5spSFxKLBQBEQGwPHyQlHi"),
    "deaott": ("DeAOTT_PRE_YTB_DAV.pth", "1ThWIZQS03cYWx1EKNN8MIMnJS5eRowzr"),
}


class TrackAnythingPipeline:
    def __init__(
        self,
        mask_phrases: list[str],
        sam_points_per_side: int = 30,
        sam_run_gap: int = 10,
    ) -> None:
        # Prepare checkpoints.
        with nvtx_range("track_anything.init.resolve_sam_checkpoint"):
            sam_model_type = _env_str("VIPE_TRACK_ANYTHING_SAM_MODEL_TYPE", "vit_b")
            default_sam_filename = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_t": "mobile_sam.pt",
            }.get(sam_model_type)
            default_sam_url = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_t": "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt",
            }.get(sam_model_type)
            if default_sam_filename is None:
                default_sam_filename = f"{sam_model_type}.pth"
            sam_ckpt_path = _resolve_checkpoint(
                cache_subdir="sam",
                default_filename=default_sam_filename,
                default_url=default_sam_url,
                env_prefix="VIPE_TRACK_ANYTHING_SAM",
            )

        with nvtx_range("track_anything.init.resolve_aot_checkpoint"):
            aot_model = _env_str("VIPE_TRACK_ANYTHING_AOT_MODEL", "r50_deaotl").lower()
            if aot_model not in DEFAULT_AOT_CHECKPOINTS:
                supported = ", ".join(sorted(DEFAULT_AOT_CHECKPOINTS))
                raise ValueError(f"unsupported VIPE_TRACK_ANYTHING_AOT_MODEL={aot_model!r}; expected one of: {supported}")
            aot_filename, aot_gdrive_id = DEFAULT_AOT_CHECKPOINTS[aot_model]
            aot_ckpt_path = _resolve_checkpoint(
                cache_subdir="aot",
                default_filename=aot_filename,
                default_gdrive_id=aot_gdrive_id,
                env_prefix="VIPE_TRACK_ANYTHING_AOT",
            )
        LOGGER.info(
            "Track Anything model selection: SAM model_type=%s checkpoint=%s; AOT model=%s checkpoint=%s",
            sam_model_type,
            sam_ckpt_path.name,
            aot_model,
            aot_ckpt_path.name,
        )

        self.threshold_args = {
            "box_threshold": 0.35,
            "text_threshold": 0.5,  # Not useful now!
            "box_size_threshold": 1.0,
            "reset_image": True,
        }
        self.frame_idx = 0
        self.caption = "".join([m + "." for m in mask_phrases])
        self.sam_run_gap = sam_run_gap
        self.input_scale = _env_float("VIPE_TRACK_ANYTHING_INPUT_SCALE", 1.0)
        if self.input_scale <= 0:
            raise ValueError("VIPE_TRACK_ANYTHING_INPUT_SCALE must be positive")
        self.aot_stride = max(1, _env_int("VIPE_TRACK_ANYTHING_AOT_STRIDE", 1))
        self._last_pred_mask_tensor: torch.Tensor | None = None
        self._last_pred_phrase: dict[int, str] = {}
        with nvtx_range("track_anything.init.build_segtracker"):
            self.segtracker = SegTracker(
                segtracker_args={
                    "sam_gap": sam_run_gap,  # the interval to run sam to segment new objects
                    "min_area": 200,  # minimal mask area to add a new mask as a new object
                    "max_obj_num": 255,  # maximal object number to track in a video
                    "min_new_obj_iou": 0.8,  # the background area ratio of a new object should > 80%
                },
                sam_args={
                    "sam_checkpoint": str(sam_ckpt_path),
                    "model_type": sam_model_type,
                    "generator_args": {
                        "points_per_side": sam_points_per_side,
                        "pred_iou_thresh": 0.8,
                        "stability_score_thresh": 0.9,
                        "crop_n_layers": 1,
                        "crop_n_points_downscale_factor": 2,
                        "min_mask_region_area": 200,
                    },
                    "gpu_id": 0,
                },
                aot_args={
                    "phase": "PRE_YTB_DAV",
                    "model": aot_model,
                    "model_path": str(aot_ckpt_path),
                    "long_term_mem_gap": 9999,
                    "max_len_long_term": 9999,
                    "gpu_id": 0,
                },
            )
        with nvtx_range("track_anything.init.restart_tracker"):
            self.segtracker.restart_tracker()
        self.instance_phrase = {0: "background"}

    def track(self, frame_data: VideoFrame) -> tuple[torch.Tensor, dict[int, str]]:
        """
        Detect new and track existing objects in the frame.

        Args:
            frame_data (VideoFrame): The frame data to track.

        Returns:
            torch.Tensor: The mask of the tracked objects (H, W) uint8 tensor.
                0 is background, >0 is object id.
            dict[int, str]: The phrases associated with each object id.
        """

        should_periodic_sam = self.frame_idx > 0 and self.frame_idx % self.sam_run_gap == 0
        if self.aot_stride > 1 and not should_periodic_sam and self._last_pred_mask_tensor is not None:
            if self.frame_idx % self.aot_stride != 0:
                self.frame_idx += 1
                with nvtx_range("track_anything.frame.reuse_previous_mask"):
                    return self._last_pred_mask_tensor.clone(), dict(self._last_pred_phrase)

        with nvtx_range("track_anything.frame.prepare_gpu"):
            if not frame_data.rgb.is_cuda:
                raise RuntimeError("GPU Track Anything path requires frame.rgb to be a CUDA tensor")

            rgb_frame = frame_data.rgb
            original_shape = tuple(rgb_frame.shape[:2])
            if self.input_scale != 1.0:
                scaled_h = max(1, int(round(original_shape[0] * self.input_scale)))
                scaled_w = max(1, int(round(original_shape[1] * self.input_scale)))
                rgb_bchw = rgb_frame.permute(2, 0, 1).unsqueeze(0)
                if self.input_scale < 1.0:
                    rgb_bchw = F.interpolate(rgb_bchw, size=(scaled_h, scaled_w), mode="area")
                else:
                    rgb_bchw = F.interpolate(rgb_bchw, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
                rgb_frame = rgb_bchw.squeeze(0).permute(1, 2, 0).contiguous()

        if self.frame_idx == 0:
            with nvtx_range("track_anything.frame.initial_detect_and_seg"):
                pred_mask, _, pred_phrase = self.segtracker.detect_and_seg(
                    rgb_frame, self.caption, **self.threshold_args
                )
            with nvtx_range("track_anything.frame.initial_add_reference"):
                self.segtracker.add_reference(rgb_frame, pred_mask)
            self.instance_phrase.update(pred_phrase)

        elif self.frame_idx % self.sam_run_gap == 0:
            with nvtx_range("track_anything.frame.periodic_detect_and_seg"):
                seg_mask, _, pred_phrase = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            with nvtx_range("track_anything.frame.periodic_aot_track"):
                track_mask = self.segtracker.track(rgb_frame)
            with nvtx_range("track_anything.frame.find_new_objs"):
                new_obj_mask, seg_to_new_mapping = self.segtracker.find_new_objs(track_mask, seg_mask)
            with nvtx_range("track_anything.frame.merge_new_objs"):
                if torch.sum(new_obj_mask > 0).item() > rgb_frame.shape[0] * rgb_frame.shape[1] * 0.4:
                    new_obj_mask = torch.zeros_like(new_obj_mask)
                    seg_to_new_mapping = {}
                pred_mask = track_mask + new_obj_mask
                pred_phrase = {seg_to_new_mapping[k]: v for k, v in pred_phrase.items() if k in seg_to_new_mapping}
            self.instance_phrase.update(pred_phrase)
            with nvtx_range("track_anything.frame.periodic_add_reference"):
                self.segtracker.add_reference(rgb_frame, pred_mask)

        else:
            with nvtx_range("track_anything.frame.aot_track_update_memory"):
                pred_mask = self.segtracker.track(rgb_frame, update_memory=True)

        self.frame_idx += 1

        with nvtx_range("track_anything.frame.phrase_lookup"):
            pred_mask_unique = torch.unique(pred_mask).detach().cpu().tolist()
            pred_phrase = {
                int(k): self.instance_phrase[int(k)] for k in pred_mask_unique if int(k) in self.instance_phrase
            }

        if self.input_scale != 1.0:
            with nvtx_range("track_anything.frame.resize_mask_to_input"):
                pred_mask = F.interpolate(
                    pred_mask[None, None].float(),
                    size=original_shape,
                    mode="nearest",
                )[0, 0].to(torch.uint8)

        with nvtx_range("track_anything.frame.mask_ready_gpu"):
            pred_mask = pred_mask.to(dtype=torch.uint8, device=rgb_frame.device)
        self._last_pred_mask_tensor = pred_mask
        self._last_pred_phrase = dict(pred_phrase)
        return pred_mask, pred_phrase
