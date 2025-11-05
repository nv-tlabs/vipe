# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.
import json
import os
import time
from pathlib import Path
import numpy as np
import PIL
import torch

from torchvision.ops import box_convert

from .groundingdino.config import config
from .groundingdino.datasets import transforms as T
from .groundingdino.models import build_model as build_grounding_dino
from .groundingdino.util.inference import predict
from .groundingdino.util.utils import clean_state_dict


# Patch GroundingDINO with FlashPack support
from flashpack import FlashPackMixin
from .groundingdino.models.main.groundingdino import GroundingDINO

# Create FlashPack-enabled version  
class FlashPackGroundingDINO(GroundingDINO, FlashPackMixin):
    """FlashPack-enabled GroundingDINO"""
    # CRITICAL: Only ignore int64 buffers (relative_position_index)
    # Include relative_position_bias_table in FlashPack so it gets correct dtype
    flashpack_ignore_suffixes = ["relative_position_index"]

    @classmethod
    def from_config(cls, cfg):
        """Build from config like build_grounding_dino does"""
        model = build_grounding_dino(cfg)
        model.__class__ = cls
        return model

class Detector:
    def __init__(self, device, use_flashpack: bool = True, flashpack_cache_dir: str = None):
        args = config
        args.device = device
        self.deivce = device
        # Check environment variable to disable FlashPack (for debugging)
        disable_flashpack = os.environ.get('DISABLE_GROUNDINGDINO_FLASHPACK') == '1'
        if disable_flashpack:
            print(f"DISABLE_GROUNDINGDINO_FLASHPACK=1, skipping FlashPack for GroundingDINO")
            use_flashpack = False

        if use_flashpack:
            if flashpack_cache_dir is None:
                flashpack_cache_dir = Path.home() / ".cache" / "vipe_trackanything_flashpack"
            else:
                flashpack_cache_dir = Path(flashpack_cache_dir)
                flashpack_cache_dir.mkdir(parents=True, exist_ok=True)
            # Use different cache files for different dtypes
            dtype_suffix = "fp32"
            model_path = flashpack_cache_dir / f"groundingdino_{dtype_suffix}.flashpack"

            if not model_path.exists():
                print("Creating GroundingDINO flashpack...")
                start = time.time()
                # Build model normally
                gd_model = build_grounding_dino(args)
                checkpoint = torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                    map_location="cpu",
                )
                gd_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False, assign=True)

                # Filter out ONLY int64 buffers (relative_position_index)
                # Keep everything else including bbox_embed and relative_position_bias_table (float tensors)
                state_dict = gd_model.state_dict()
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.endswith("relative_position_index")  # Only int64 buffer
                }

                # Save filtered state_dict with flashpack
                # CRITICAL: Use specified dtype (default float16 for memory efficiency)
                from flashpack import pack_to_file
                pack_to_file(
                    filtered_state_dict,
                    str(model_path),
                    target_dtype=torch.float32,  # Use specified dtype instead of hardcoded float32
                    silent=False
                )

                print(f"GroundingDINO flashpack creation took {time.time() - start:.2f}s")
                del gd_model
                torch.cuda.empty_cache()

            # Load from flashpack  
            print(f"Loading GroundingDINO from flashpack...")
            start = time.time()
            device_str = f"cuda:{args.device}" if isinstance(args.device, int) else str(args.device)

            # Step 1: Build model structure and load bbox_embed from checkpoint
            self.gd = build_grounding_dino(args)
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                map_location="cpu",
            )
            self.gd.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False, assign=True)

            # Step 2: Overwrite main weights from flashpack (fast! skip bbox_embed already loaded)
            from flashpack import assign_from_file  
            assign_from_file(
                self.gd,
                str(model_path),
                device=device_str,
                strict_buffers=False,  # Missing int64 relative_position_index OK
                ignore_prefixes=["bbox_embed", "transformer.decoder.bbox_embed", "transformer.enc_out_bbox_embed"]  # Already loaded from checkpoint
            )
            print(f"GroundingDINO flashpack loading took {time.time() - start:.2f}s")

        self.gd.eval()
        # Store reference to model (for compatibility with code that accesses detector.model)
        self.model = self.gd
        self.backbone = self.gd.backbone if hasattr(self.gd, 'backbone') else None
    def image_transform_grounding(self, init_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(init_image, None)  # 3, h, w
        return init_image, image

    def image_transform_grounding_for_vis(self, init_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
            ]
        )
        image, _ = transform(init_image, None)  # 3, h, w
        return image

    def transfer_boxes_format(self, boxes, height, width):
        boxes = boxes * torch.Tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        transfered_boxes = []
        for i in range(len(boxes)):
            box = boxes[i]
            transfered_box = [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]]
            transfered_boxes.append(transfered_box)

        transfered_boxes = np.array(transfered_boxes)
        return transfered_boxes

    @torch.no_grad()
    def run_grounding(
        self,
        origin_frame,
        grounding_caption,
        box_threshold,
        text_threshold: float = 0.0,
    ):
        """
        return:
            annotated_frame:nd.array
            transfered_boxes: nd.array [N, 4]: [[x0, y0], [x1, y1]]
        """
        height, width, _ = origin_frame.shape
        img_pil = PIL.Image.fromarray(origin_frame)
        re_width, re_height = img_pil.size
        _, image_tensor = self.image_transform_grounding(img_pil)
        # img_pil = self.image_transform_grounding_for_vis(img_pil)

        # run grounidng
        boxes, logits, phrases = predict(
            self.gd,
            image_tensor,
            grounding_caption,
            box_threshold,
            text_threshold,
            device=self.deivce,
        )
        # annotated_frame = annotate(
        #     image_source=np.asarray(img_pil),
        #     boxes=boxes,
        #     logits=logits,
        #     phrases=phrases,
        # )[:, :, ::-1]
        # annotated_frame = cv2.resize(
        #     annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR
        # )

        # transfer boxes to sam-format
        transfered_boxes = self.transfer_boxes_format(boxes, re_height, re_width)
        return (height, width), transfered_boxes, phrases
