# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
import torch
import torch.nn.functional as F


from vipe.utils.model_cache import ModelCache

from .aot import config as engine_config
from .aot.networks.engines import build_engine
from .aot.networks.engines.aot_engine import AOTEngine, AOTInferEngine
from .aot.networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
from .aot.networks.models import build_vos_model
from .aot.utils.checkpoint import load_network


class AOTTracker(object):
    def __init__(self, cfg, gpu_id=0, model_cache: ModelCache | None = None):
        self.gpu_id = gpu_id

        # The VOS network holds only weights and is cached/shared across streams.
        # The engine below owns the per-video memory bank (reference frames,
        # masks, object ids), so it is always rebuilt to avoid leaking tracking
        # state between videos.
        def _build_aot_model():
            model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
            model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id)
            model.eval()
            return model

        if model_cache is not None:
            self.model = model_cache.get(f"track_anything/aot/{cfg.MODEL_VOS}/{cfg.TEST_CKPT_PATH}", _build_aot_model)
        else:
            self.model = _build_aot_model()

        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=self.model,
            gpu_id=gpu_id,
            short_term_mem_skip=1,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
            max_len_long_term=cfg.MAX_LEN_LONG_TERM,
        )
        self.max_short_edge = cfg.TEST_MAX_SHORT_EDGE
        self.max_long_edge = cfg.TEST_MAX_LONG_EDGE
        self.flip = cfg.TEST_FLIP
        self.multi_scale = cfg.TEST_MULTISCALE
        self.align_corners = cfg.MODEL_ALIGN_CORNERS
        self.max_stride = 16

        self.model.eval()

    def _restricted_size(self, height: int, width: int) -> tuple[int, int]:
        if len(self.multi_scale) != 1 or self.multi_scale[0] != 1:
            raise NotImplementedError("GPU AOT path currently expects a single test scale of 1")
        if self.flip:
            raise NotImplementedError("GPU AOT path does not support test-time flip")

        new_h, new_w = float(height), float(width)

        if self.max_short_edge is not None:
            short_edge = min(height, width)
            if short_edge > self.max_short_edge:
                scale = float(self.max_short_edge) / short_edge
                new_h *= scale
                new_w *= scale

        if self.max_long_edge is not None:
            long_edge = max(new_h, new_w)
            if long_edge > self.max_long_edge:
                scale = float(self.max_long_edge) / long_edge
                new_h *= scale
                new_w *= scale

        new_h = int(new_h)
        new_w = int(new_w)

        if self.align_corners:
            if (new_h - 1) % self.max_stride != 0:
                new_h = int(np.around((new_h - 1) / self.max_stride) * self.max_stride + 1)
            if (new_w - 1) % self.max_stride != 0:
                new_w = int(np.around((new_w - 1) / self.max_stride) * self.max_stride + 1)
        else:
            if new_h % self.max_stride != 0:
                new_h = int(np.around(new_h / self.max_stride) * self.max_stride)
            if new_w % self.max_stride != 0:
                new_w = int(np.around(new_w / self.max_stride) * self.max_stride)

        return new_h, new_w

    def _prepare_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("GPU AOT path requires image as a CUDA torch.Tensor")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB image tensor, got shape {tuple(image.shape)}")

        device = torch.device(f"cuda:{self.gpu_id}")
        input_is_uint8 = image.dtype == torch.uint8
        image = image.to(device=device, dtype=torch.float32)
        if input_is_uint8:
            image = image / 255.0

        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
        new_h, new_w = self._restricted_size(image.shape[-2], image.shape[-1])
        if (new_h, new_w) != tuple(image.shape[-2:]):
            image = F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False)

        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (image - mean) / std

    @torch.no_grad()
    def add_reference_frame(self, frame, mask, obj_nums, frame_step, incremental=False):
        frame = self._prepare_image_tensor(frame)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("GPU AOT path requires mask as a CUDA torch.Tensor")
        mask = mask.to(device=frame.device, dtype=torch.float32)[None, None]
        _mask = F.interpolate(mask, size=frame.shape[-2:], mode="nearest")

        if incremental:
            self.engine.add_reference_frame_incremental(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)
        else:
            self.engine.add_reference_frame(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)

    @torch.no_grad()
    def track(self, image):
        output_height, output_width = image.shape[0], image.shape[1]
        image = self._prepare_image_tensor(image)
        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))

        # pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_logit, dim=1, keepdim=True).float()

        return pred_label

    @torch.no_grad()
    def update_memory(self, pred_label):
        self.engine.update_memory(pred_label)

    @torch.no_grad()
    def restart(self):
        self.engine.restart_engine()

    @torch.no_grad()
    def build_tracker_engine(self, name, **kwargs):
        if name == "aotengine":
            return AOTTrackerInferEngine(**kwargs)
        elif name == "deaotengine":
            return DeAOTTrackerInferEngine(**kwargs)
        else:
            raise NotImplementedError


class AOTTrackerInferEngine(AOTInferEngine):
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
    ):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while aot_num > len(self.aot_engines):
            new_engine = AOTEngine(self.AOT, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(
                    img,
                    separated_mask,
                    obj_nums=[separated_obj_num],
                    frame_step=frame_step,
                    img_embs=img_embs,
                )
            else:
                aot_engine.update_short_term_memory(separated_mask)

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


class DeAOTTrackerInferEngine(DeAOTInferEngine):
    def __init__(
        self,
        aot_model,
        gpu_id=0,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_aot_obj_num=None,
    ):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while aot_num > len(self.aot_engines):
            new_engine = DeAOTEngine(self.AOT, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(
                    img,
                    separated_mask,
                    obj_nums=[separated_obj_num],
                    frame_step=frame_step,
                    img_embs=img_embs,
                )
            else:
                aot_engine.update_short_term_memory(separated_mask)

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


def get_aot(args, model_cache: ModelCache | None = None):
    # build vos engine
    cfg = engine_config.EngineConfig(args["phase"])
    cfg.TEST_CKPT_PATH = args["model_path"]
    cfg.TEST_LONG_TERM_MEM_GAP = args["long_term_mem_gap"]
    cfg.MAX_LEN_LONG_TERM = args["max_len_long_term"]
    tracker = AOTTracker(cfg, args["gpu_id"], model_cache=model_cache)
    return tracker
