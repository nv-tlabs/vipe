# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np

from .aot_tracker import get_aot
from .detector import Detector
from .segmentor import Segmentor

import time
class SegTracker:
    def __init__(self, segtracker_args, sam_args, aot_args, preloaded_sam=None, preloaded_aot=None, use_flashpack: bool = True, flashpack_cache_dir: str = None) -> None:
        """
        Initialize SAM and AOT.
        """
        sam_start = time.time()
        self.sam = Segmentor(sam_args, preloaded_sam=preloaded_sam)

        print(f"  Segmentor init took {time.time() - sam_start:.2f}s")

        aot_start = time.time()
        self.tracker = get_aot(aot_args, use_flashpack=use_flashpack, flashpack_cache_dir=flashpack_cache_dir, preloaded_model=preloaded_aot)
        print(f"  AOT tracker init took {time.time() - aot_start:.2f}s")

        detector_start = time.time()
        self.detector = Detector(self.sam.device, use_flashpack=use_flashpack, flashpack_cache_dir=flashpack_cache_dir)
        print(f"  Detector init took {time.time() - detector_start:.2f}s")
        self.sam_gap = segtracker_args["sam_gap"]
        self.min_area = segtracker_args["min_area"]
        self.max_obj_num = segtracker_args["max_obj_num"]
        self.min_new_obj_iou = segtracker_args["min_new_obj_iou"]
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None

        # debug
        self.everything_points = []
        self.everything_labels = []

    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        # obj_ids = np.unique(updated_merged_mask)
        # obj_ids = obj_ids[obj_ids!=0]
        # self.object_idx = int(max(obj_ids)) + 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self, frame, mask, frame_step=0):
        """
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        """
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
        self.tracker.add_reference_frame(frame, mask, self.curr_idx, frame_step)
        self.curr_idx += 1

    def track(self, frame, update_memory=False):
        """
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        """
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i != 0]
        return objs

    def get_obj_num(self):
        objs = self.get_tracking_objs()
        if len(objs) == 0:
            return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        """
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        """
        new_obj_mask = (track_mask == 0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids != 0]
        seg_to_new_mapping = {}
        # obj_num = self.get_obj_num() + 1
        obj_num = self.curr_idx
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask == idx)
            obj_area = np.sum(seg_mask == idx)
            if (
                new_obj_area / obj_area < self.min_new_obj_iou
                or new_obj_area < self.min_area
                or obj_num > self.max_obj_num
            ):
                new_obj_mask[new_obj_mask == idx] = 0
            else:
                new_obj_mask[new_obj_mask == idx] = obj_num
                seg_to_new_mapping[idx] = obj_num
                obj_num += 1
        return new_obj_mask, seg_to_new_mapping

    def restart_tracker(self):
        self.tracker.restart()

    def add_mask(self, interactive_mask: np.ndarray):
        """
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        """
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape, dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask

    def detect_and_seg(
        self,
        origin_frame: np.ndarray,
        grounding_caption,
        box_threshold,
        text_threshold: float = 0.0,
        box_size_threshold=1,
        reset_image=False,
    ):
        """
        Using Grounding-DINO to detect object acc Text-prompts
        Retrun:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        """
        # backup id and origin-merged-mask
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask
        seg_phrase = {}

        # get annotated_frame and boxes
        annotated_frame_shape, boxes, phrases = self.detector.run_grounding(
            origin_frame, grounding_caption, box_threshold, text_threshold
        )
        refined_merged_mask = np.zeros(annotated_frame_shape, dtype=np.uint8)
        for i in range(len(boxes)):
            bbox = boxes[i]
            phrase = phrases[i]
            if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame_shape[0] * annotated_frame_shape[
                1
            ] * box_size_threshold:
                continue
            interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            refined_merged_mask = self.add_mask(interactive_mask)
            seg_phrase[self.curr_idx] = phrase
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame_shape, seg_phrase
