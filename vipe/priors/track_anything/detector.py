# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
import PIL
import torch
from torchvision.ops import box_convert

from vipe.utils.model_cache import ModelCache

from .groundingdino.config import config
from .groundingdino.datasets import transforms as T
from .groundingdino.models import build_model as build_grounding_dino
from .groundingdino.util.inference import predict
from .groundingdino.util.utils import clean_state_dict


class Detector:
    def __init__(self, device, model_cache: ModelCache | None = None):
        args = config
        args.device = device
        self.deivce = device

        # GroundingDINO is stateless across frames, so its weights are cached
        # and shared across streams.
        def _build_gd():
            gd = build_grounding_dino(args)
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                map_location="cpu",
            )
            gd.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            gd.eval()
            return gd

        if model_cache is not None:
            self.gd = model_cache.get("track_anything/groundingdino_swint", _build_gd)
        else:
            self.gd = _build_gd()

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
