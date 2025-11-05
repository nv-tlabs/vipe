# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.
import json
import time, os
from pathlib import Path

import gdown
import numpy as np
import torch
from flashpack import FlashPackMixin

from vipe.streams.base import VideoFrame

from .seg_tracker import SegTracker

from .sam import sam_model_registry


class FlashPackSAMWrapper(torch.nn.Module, FlashPackMixin):
    """FlashPack wrapper for SAM model."""

    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
class TrackAnythingPipeline:
    def __init__(
        self,
        mask_phrases: list[str],
        sam_points_per_side: int = 30,
        sam_run_gap: int = 10,
        preloaded_sam=None,
        preloaded_aot=None,
        use_flashpack: bool = True,
        dtype=None,
        flashpack_cache_dir: Path = None,
    ) -> None:
        overall_start = time.perf_counter()
        # Prepare checkpoints.
        sam_ckpt_path = Path(torch.hub.get_dir()) / "sam" / "sam_vit_b_01ec64.pth" #TODO: Verify if this exists in modal cache
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
                "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view",
                output=str(aot_ckpt_path),
                fuzzy=True,
            )
            print(f"Checkpoint preparation took {time.perf_counter() - overall_start:.2f}s")

        self.threshold_args = {
            "box_threshold": 0.35,
            "text_threshold": 0.5,  # Not useful now!
            "box_size_threshold": 1.0,
            "reset_image": True,
        }
        self.frame_idx = 0
        self.caption = "".join([m + "." for m in mask_phrases])
        self.sam_run_gap = sam_run_gap
        # Use flashpack caching for faster loading
        if use_flashpack:
            flash_start = time.perf_counter()
            sam_ckpt_path, aot_ckpt_path = self._setup_flashpack_checkpoints(
                sam_ckpt_path=sam_ckpt_path,
                aot_ckpt_path=aot_ckpt_path,
                flashpack_cache_dir=flashpack_cache_dir
            )
            print(f"Flashpack checkpoint setup took {time.perf_counter() - flash_start:.2f}s")

        segtracker_start = time.perf_counter()
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
            use_flashpack=use_flashpack,
            flashpack_cache_dir=flashpack_cache_dir,
            preloaded_sam=preloaded_sam,
            preloaded_aot=preloaded_aot,
        )
        print(f"SegTracker initialization took {time.perf_counter() - segtracker_start:.2f}s")

        restart_start = time.perf_counter()
        self.segtracker.restart_tracker()
        print(f"Tracker restart took {time.perf_counter() - restart_start:.2f}s")

        self.instance_phrase = {0: "background"}
        print(f"TrackAnythingPipeline total init took {time.perf_counter() - overall_start:.2f}s")

    def _setup_flashpack_checkpoints(
        self,
        sam_ckpt_path: Path,
        aot_ckpt_path: Path,
        flashpack_cache_dir: Path = None
    ) -> tuple[Path, Path]:
        """
        Setup flashpack-optimized checkpoints for SAM and AOT models.
        Args:
            sam_ckpt_path (Path): Path to the SAM checkpoint.
            aot_ckpt_path (Path): Path to the AOT checkpoint.
            flashpack_cache_dir (Path, optional): Directory to store flashpack caches.
                Defaults to ~/.cache/vipe_trackanything_flashpack.
        Returns:
            tuple[Path, Path]: The paths to the (potentially flashpacked) sam and aot checkpoints.
        """
        if flashpack_cache_dir is None:
            flashpack_cache_dir = Path.home() / ".cache" / "vipe_trackanything_flashpack"
            flashpack_cache_dir.mkdir(parents=True, exist_ok=True)

        else:
            flashpack_cache_dir = os.path.join(flashpack_cache_dir, "vipe_trackanything_flashpack")
            os.makedirs(flashpack_cache_dir, exist_ok=True)
            flashpack_cache_dir = Path(flashpack_cache_dir)
        # SAM flashpack cache
        sam_flashpack_path = flashpack_cache_dir / "sam_vit_b.flashpack"
        sam_config_path = flashpack_cache_dir / "sam_config.json"
        if not sam_flashpack_path.exists() and sam_ckpt_path.exists():
            print("Creating flashpack for SAM model...")
            start = time.perf_counter()
            # Build SAM model and load weights
            sam_model = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt_path))
            # Save config for later
            sam_config = {
                "model_type": "vit_b",
                "encoder_embed_dim": 768,
                "encoder_depth": 12,
                "encoder_num_heads": 12,
                "encoder_global_attn_indexes": [2, 5, 8, 11],
            }
            with open(sam_config_path, 'w') as f:
                json.dump(sam_config, f)
            # Wrap and save as flashpack
            wrapped_sam = FlashPackSAMWrapper(sam_model)
            wrapped_sam.save_flashpack(str(sam_flashpack_path), target_dtype=torch.float32)
            print(f"SAM flashpack creation took {time.perf_counter() - start:.2f}s")
            del wrapped_sam, sam_model
            torch.cuda.empty_cache()

        # Use flashpack checkpoint if it exists
        if sam_flashpack_path.exists():
            sam_ckpt_path = sam_flashpack_path

        # For AOT, just use regular caching since it's smaller
        # (FlashPack overhead might not be worth it for smaller models)

        return sam_ckpt_path, aot_ckpt_path

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

        # Convert to RGB numpy images
        rgb_frame = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)

        if self.frame_idx == 0:
            pred_mask, _, pred_phrase = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            self.segtracker.add_reference(rgb_frame, pred_mask)
            self.instance_phrase.update(pred_phrase)

        elif self.frame_idx % self.sam_run_gap == 0:
            seg_mask, _, pred_phrase = self.segtracker.detect_and_seg(rgb_frame, self.caption, **self.threshold_args)
            track_mask = self.segtracker.track(rgb_frame)
            new_obj_mask, seg_to_new_mapping = self.segtracker.find_new_objs(track_mask, seg_mask)
            if np.sum(new_obj_mask > 0) > rgb_frame.shape[0] * rgb_frame.shape[1] * 0.4:
                new_obj_mask = np.zeros_like(new_obj_mask)
                seg_to_new_mapping = {}
            pred_mask = track_mask + new_obj_mask
            pred_phrase = {seg_to_new_mapping[k]: v for k, v in pred_phrase.items() if k in seg_to_new_mapping}
            self.instance_phrase.update(pred_phrase)
            self.segtracker.add_reference(rgb_frame, pred_mask)

        else:
            pred_mask = self.segtracker.track(rgb_frame, update_memory=True)

        self.frame_idx += 1

        pred_mask_unique = np.unique(pred_mask)
        pred_phrase = {k: self.instance_phrase[k] for k in pred_mask_unique}

        return torch.from_numpy(pred_mask).cuda(), pred_phrase
