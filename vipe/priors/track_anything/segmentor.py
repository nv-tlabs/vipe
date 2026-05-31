# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
import torch

from vipe.utils.model_cache import ModelCache

from .sam import SamAutomaticMaskGenerator, sam_model_registry


class Segmentor:
    def __init__(self, sam_args, model_cache: ModelCache | None = None):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            gpu_id: device
        """
        self.device = sam_args["gpu_id"]

        # The SAM network holds only weights (inference-only), so it is cached
        # and shared across streams. The mask generator / predictor below wrap
        # this network but hold per-frame image embedding state, so they are
        # always rebuilt per instance.
        def _build_sam():
            sam = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])
            sam.to(device=self.device)
            sam.eval()
            return sam

        if model_cache is not None:
            self.sam = model_cache.get(
                f"track_anything/sam/{sam_args['model_type']}/{sam_args['sam_checkpoint']}", _build_sam
            )
        else:
            self.sam = _build_sam()

        self.everything_generator = SamAutomaticMaskGenerator(model=self.sam, **sam_args["generator_args"])
        self.interactive_predictor = self.everything_generator.predictor
        self.have_embedded = False

    @torch.no_grad()
    def set_image(self, image):
        # calculate the embedding only once per frame.
        if not self.have_embedded:
            self.interactive_predictor.set_image(image)
            self.have_embedded = True

    @torch.no_grad()
    def set_image_tensor(self, image: torch.Tensor):
        if not self.have_embedded:
            self.interactive_predictor.set_image_tensor(image)
            self.have_embedded = True

    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):
        assert self.have_embedded, "image embedding for sam need be set before predict."

        if mode == "point":
            masks, scores, logits = self.interactive_predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_modes"],
                multimask_output=multimask,
            )
        elif mode == "mask":
            masks, scores, logits = self.interactive_predictor.predict(
                mask_input=prompts["mask_prompt"], multimask_output=multimask
            )
        elif mode == "point_mask":
            masks, scores, logits = self.interactive_predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_modes"],
                mask_input=prompts["mask_prompt"],
                multimask_output=multimask,
            )

        return masks, scores, logits

    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        """

        return:
            mask: one-hot
        """
        self.set_image(origin_frame)

        prompts = {
            "point_coords": coords,
            "point_modes": modes,
        }
        masks, scores, logits = self.interactive_predict(prompts, "point", multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            "point_coords": coords,
            "point_modes": modes,
            "mask_prompt": logit[None, :, :],
        }
        masks, scores, logits = self.interactive_predict(prompts, "point_mask", multimask)
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8)

    def segment_with_box(self, origin_frame, bbox, reset_image=False):
        if reset_image:
            self.interactive_predictor.set_image(origin_frame)
        else:
            self.set_image(origin_frame)
        # coord = np.array([[int((bbox[1][0] - bbox[0][0]) / 2.),  int((bbox[1][1] - bbox[0][1]) / 2)]])
        # point_label = np.array([1])

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]),
            multimask_output=True,
        )
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]]),
            mask_input=logit[None, :, :],
            multimask_output=True,
        )
        mask = masks[np.argmax(scores)]

        return [mask]

    @torch.no_grad()
    def segment_with_box_tensor(
        self,
        origin_frame: torch.Tensor,
        boxes: torch.Tensor,
        reset_image: bool = False,
    ) -> torch.Tensor:
        if not isinstance(origin_frame, torch.Tensor):
            raise TypeError("GPU SAM path requires origin_frame as a torch.Tensor")
        if not isinstance(boxes, torch.Tensor):
            raise TypeError("GPU SAM path requires boxes as a torch.Tensor")

        if reset_image:
            self.interactive_predictor.set_image_tensor(origin_frame)
            self.have_embedded = True
        else:
            self.set_image_tensor(origin_frame)

        boxes = boxes.to(device=self.interactive_predictor.device, dtype=torch.float32)
        transformed_boxes = self.interactive_predictor.transform.apply_boxes_torch(
            boxes,
            self.interactive_predictor.original_size,
        )

        _masks, scores, logits = self.interactive_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            mask_input=None,
            multimask_output=True,
            return_logits=True,
        )

        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        best_indices = scores.argmax(dim=1)
        best_logits = logits[batch_indices, best_indices].unsqueeze(1)

        masks, scores, _logits = self.interactive_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            mask_input=best_logits,
            multimask_output=True,
            return_logits=False,
        )

        best_indices = scores.argmax(dim=1)
        return masks[batch_indices, best_indices].to(torch.uint8)
