# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

from vipe.utils.model_cache import ModelCache

from .groundingdino.config import config
from .groundingdino.datasets import transforms as T
from .groundingdino.models import build_model as build_grounding_dino
from .groundingdino.util.inference import predict, preprocess_caption
from .groundingdino.util.utils import clean_state_dict, get_best_phrase_from_logits


class Detector:
    def __init__(self, device, model_cache: ModelCache | None = None):
        args = config
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        args.device = str(self.device)
        self.deivce = self.device

        # GroundingDINO is stateless across frames, so its weights are cached
        # and shared across streams.
        def _build_gd():
            gd = build_grounding_dino(args)
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                map_location="cpu",
            )
            gd.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            gd.to(self.device)
            gd.eval()
            return gd

        if model_cache is not None:
            self.gd = model_cache.get("track_anything/groundingdino_swint", _build_gd)
        else:
            self.gd = _build_gd()

    def _grounding_resize_size(
        self,
        height: int,
        width: int,
        size: int = 800,
        max_size: int = 1333,
    ) -> tuple[int, int]:
        min_original_size = float(min((width, height)))
        max_original_size = float(max((width, height)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (width <= height and width == size) or (height <= width and height == size):
            return height, width

        if width < height:
            output_width = size
            output_height = int(size * height / width)
        else:
            output_height = size
            output_width = int(size * width / height)

        return output_height, output_width

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

    def image_transform_grounding_tensor(self, init_image: torch.Tensor) -> torch.Tensor:
        if not isinstance(init_image, torch.Tensor):
            raise TypeError("GPU GroundingDINO path requires image as a torch.Tensor")
        if init_image.ndim != 3 or init_image.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB tensor, got shape {tuple(init_image.shape)}")

        input_is_uint8 = init_image.dtype == torch.uint8
        image = init_image.to(device=self.device, dtype=torch.float32)
        if input_is_uint8:
            image = image / 255.0

        height, width = int(image.shape[0]), int(image.shape[1])
        target_size = self._grounding_resize_size(height, width)
        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
        image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False, antialias=True)

        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return ((image - mean) / std)[0]

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
    def run_grounding_tensor(
        self,
        origin_frame: torch.Tensor,
        grounding_caption,
        box_threshold,
        text_threshold: float = 0.0,
    ):
        """
        GPU-direct GroundingDINO path.
        Returns the original frame shape and XYXY boxes as a CUDA tensor.
        """
        height, width = int(origin_frame.shape[0]), int(origin_frame.shape[1])
        image_tensor = self.image_transform_grounding_tensor(origin_frame)
        caption = preprocess_caption(caption=grounding_caption)

        outputs = self.gd(image_tensor[None], captions=[caption])
        prediction_logits = outputs["pred_logits"].sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"][0]

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]

        tokenizer = self.gd.tokenizer
        tokenized = tokenizer(caption)
        phrases = [get_best_phrase_from_logits(logit, tokenized, tokenizer) for logit in logits]

        boxes = boxes * boxes.new_tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return (height, width), boxes, phrases

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
