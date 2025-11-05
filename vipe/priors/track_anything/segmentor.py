# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
import torch
import os
from pathlib import Path
import time
from .sam import SamAutomaticMaskGenerator, sam_model_registry

from flashpack import FlashPackMixin
class FlashPackSAMWrapper(torch.nn.Module, FlashPackMixin):
    """FlashPack wrapper for SAM model."""

    def __init__(self, sam_model=None, sam_config=None, **kwargs):
        super().__init__()
        if sam_model is not None:
            self.sam = sam_model
        elif sam_config is not None:
            # Build SAM from config (no weights loaded yet)
            model_type = sam_config.get("model_type", "vit_b")
            self.sam = sam_model_registry[model_type](checkpoint=None)
        else:
            # Extract config from kwargs if provided by flashpack
            sam_config = kwargs.get('config', {})
            model_type = sam_config.get("model_type", "vit_b")
            self.sam = sam_model_registry[model_type](checkpoint=None)
class Segmentor:
    def __init__(self, sam_args, preloaded_sam=None):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            gpu_id: device
        preloaded_sam: Pre-loaded SAM model (for Modal optimization)
        """
        self.device = sam_args["gpu_id"]
        if preloaded_sam is not None:
            self.sam = preloaded_sam.to(device=self.device)
        else:
            # Check if using flashpack format
            sam_checkpoint = sam_args["sam_checkpoint"]
            if sam_checkpoint.endswith('.flashpack'):
                # Load from flashpack
                print("Loading SAM from flashpack...")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                # Load config
                config_path = Path(sam_checkpoint).parent / "sam_config.json"
                if config_path.exists():
                    import json
                    with open(config_path, 'r') as f:
                        sam_config = json.load(f)
                else:
                    sam_config = {"model_type": sam_args["model_type"]}

                # Convert device to torch.device if it's an integer
                device = torch.device(f"cuda:{self.device}") if isinstance(self.device, int) else self.device

                # Load from flashpack
                wrapped_sam = FlashPackSAMWrapper.from_flashpack(
                    sam_checkpoint,
                    config=sam_config,
                    device=device
                )
                self.sam = wrapped_sam.sam

                end.record()
                torch.cuda.synchronize()
                print(f"SAM flashpack loading took {start.elapsed_time(end)/1000:.2f}s")
            else:
                # Original loading
                self.sam = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])

            # Move to device
            import time
            move_start = time.time()

            self.sam.to(device=self.device)
            print(f"    SAM .to(device) took {time.time() - move_start:.2f}s")

            # Create generator
        gen_start = time.time()
        self.everything_generator = SamAutomaticMaskGenerator(model=self.sam, **sam_args["generator_args"])
        print(f"    SamAutomaticMaskGenerator init took {time.time() - gen_start:.2f}s")
        self.interactive_predictor = self.everything_generator.predictor
        self.have_embedded = False

    @torch.no_grad()
    def set_image(self, image):
        # calculate the embedding only once per frame.
        if not self.have_embedded:
            self.interactive_predictor.set_image(image)
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
