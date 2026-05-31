# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

from pathlib import Path

import gdown
import torch

from vipe.streams.base import VideoFrame
from vipe.utils.model_cache import ModelCache

from .seg_tracker import SegTracker


class TrackAnythingPipeline:
    def __init__(
        self,
        mask_phrases: list[str],
        sam_points_per_side: int = 30,
        sam_run_gap: int = 10,
        model_cache: ModelCache | None = None,
    ) -> None:
        # Prepare checkpoints.
        sam_ckpt_path = Path(torch.hub.get_dir()) / "sam" / "sam_vit_b_01ec64.pth"
        if not sam_ckpt_path.exists():
            sam_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                dst=str(sam_ckpt_path),
            )

        aot_ckpt_path = Path(torch.hub.get_dir()) / "aot" / "R50_DeAOTL_PRE_YTB_DAV.pth"
        if not aot_ckpt_path.exists():
            aot_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                id="1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ",
                output=str(aot_ckpt_path),
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
        self.segtracker = SegTracker(
            segtracker_args={
                "sam_gap": sam_run_gap,  # the interval to run sam to segment new objects
                "min_area": 200,  # minimal mask area to add a new mask as a new object
                "max_obj_num": 255,  # maximal object number to track in a video
                "min_new_obj_iou": 0.8,  # the background area ratio of a new object should > 80%
            },
            sam_args={
                "sam_checkpoint": str(sam_ckpt_path),
                "model_type": "vit_b",
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
                "model": "r50_deaotl",
                "model_path": str(aot_ckpt_path),
                "long_term_mem_gap": 9999,
                "max_len_long_term": 9999,
                "gpu_id": 0,
            },
            model_cache=model_cache,
        )
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

        if not frame_data.rgb.is_cuda:
            raise RuntimeError("GPU Track Anything path requires frame.rgb to be a CUDA tensor")

        rgb_frame = frame_data.rgb

        if self.frame_idx == 0:
            pred_mask, _, pred_phrase = self.segtracker.detect_and_seg(
                rgb_frame, self.caption, **self.threshold_args
            )
            self.segtracker.add_reference(rgb_frame, pred_mask)
            self.instance_phrase.update(pred_phrase)

        elif self.frame_idx % self.sam_run_gap == 0:
            seg_mask, _, pred_phrase = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            track_mask = self.segtracker.track(rgb_frame)
            new_obj_mask, seg_to_new_mapping = self.segtracker.find_new_objs(track_mask, seg_mask)
            if torch.sum(new_obj_mask > 0).item() > rgb_frame.shape[0] * rgb_frame.shape[1] * 0.4:
                new_obj_mask = torch.zeros_like(new_obj_mask)
                seg_to_new_mapping = {}
            pred_mask = track_mask + new_obj_mask
            pred_phrase = {seg_to_new_mapping[k]: v for k, v in pred_phrase.items() if k in seg_to_new_mapping}
            self.instance_phrase.update(pred_phrase)
            self.segtracker.add_reference(rgb_frame, pred_mask)

        else:
            pred_mask = self.segtracker.track(rgb_frame, update_memory=True)

        self.frame_idx += 1

        pred_mask_unique = torch.unique(pred_mask).detach().cpu().tolist()
        pred_phrase = {
            int(k): self.instance_phrase[int(k)] for k in pred_mask_unique if int(k) in self.instance_phrase
        }

        pred_mask = pred_mask.to(dtype=torch.uint8, device=rgb_frame.device)
        return pred_mask, pred_phrase
