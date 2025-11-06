# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.
import time
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
from flashpack import FlashPackMixin

from torchvision import transforms

from .aot import config as engine_config
from .aot.networks.engines import build_engine
from .aot.networks.engines.aot_engine import AOTEngine, AOTInferEngine
from .aot.networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
from .aot.networks.models import build_vos_model
from .aot.transforms import video_transforms as tr
from .aot.utils.checkpoint import load_network
class FlashPackAOTWrapper(torch.nn.Module, FlashPackMixin):
    """FlashPack wrapper for AOT tracker model."""

    def __init__(self, aot_model=None, cfg=None, **kwargs):
        super().__init__()
        if aot_model is not None:
            self.aot = aot_model
        elif cfg is not None:
            self.aot = build_vos_model(cfg.MODEL_VOS, cfg)
        else:
            cfg = kwargs.get('config', None)
            if cfg is not None:
                self.aot = build_vos_model(cfg.MODEL_VOS, cfg)



class AOTTracker(object):
    def __init__(self, cfg, gpu_id=0, device="cuda", preloaded_model=None, use_flashpack: bool = True, flashpack_cache_dir: str = None):
        self.gpu_id = gpu_id
        self.device = device
        if preloaded_model is not None:
            self.model = preloaded_model.to(device)
        else:
            if use_flashpack:
                # Setup flashpack cache
                if flashpack_cache_dir is None:
                    flashpack_cache_dir = Path.home() / ".cache" / "vipe_trackanything_flashpack"
                else:
                    flashpack_cache_dir = os.path.join(flashpack_cache_dir, "vipe_trackanything_flashpack")
                    os.makedirs(flashpack_cache_dir, exist_ok=True)
                    flashpack_cache_dir = Path(flashpack_cache_dir)
                flashpack_cache_dir.mkdir(parents=True, exist_ok=True)
                aot_flashpack_path = flashpack_cache_dir / "aot_tracker.flashpack"

                # Create flashpack if doesn't exist
                if not aot_flashpack_path.exists():
                    print("Creating flashpack for AOT tracker...")
                    start = time.time()
                    # Build model and load weights normally
                    self.model = build_vos_model(cfg.MODEL_VOS, cfg)
                    if device == "cuda":
                        self.model = self.model.cuda(gpu_id)
                    else:
                        self.model = self.model.to(device)
                    self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id if device == "cuda" else device)

                    # Wrap and save as flashpack
                    wrapped_aot = FlashPackAOTWrapper(self.model)
                    wrapped_aot.save_flashpack(str(aot_flashpack_path), target_dtype=torch.float32)
                    print(f"AOT flashpack creation took {time.time() - start:.2f}s")
                    del wrapped_aot
                    torch.cuda.empty_cache()
                else:
                    # Load from flashpack
                    print("Loading AOT tracker from flashpack...")
                    start = time.time()

                    device = torch.device(f"cuda:{gpu_id}") if isinstance(gpu_id, int) else gpu_id
                    wrapped_aot = FlashPackAOTWrapper.from_flashpack(
                        str(aot_flashpack_path),
                        config=cfg,
                        device=device
                    )
                    self.model = wrapped_aot.aot
                    print(f"AOT flashpack loading took {time.time() - start:.2f}s")
            else:
                # Original loading
                self.model = build_vos_model(cfg.MODEL_VOS, cfg)
                if device == "cuda":
                    self.model = self.model.cuda(gpu_id)
                else:
                    self.model = self.model.to(device)
                self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id if device == "cuda" else device)

            
            
        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=self.model,
            gpu_id=gpu_id,
            short_term_mem_skip=1,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
            max_len_long_term=cfg.MAX_LEN_LONG_TERM,
        )

        self.transform = transforms.Compose(
            [
                tr.MultiRestrictSize(
                    cfg.TEST_MAX_SHORT_EDGE,
                    cfg.TEST_MAX_LONG_EDGE,
                    cfg.TEST_FLIP,
                    cfg.TEST_MULTISCALE,
                    cfg.MODEL_ALIGN_CORNERS,
                ),
                tr.MultiToTensor(),
            ]
        )

        self.model.eval()

    @torch.no_grad()
    def add_reference_frame(self, frame, mask, obj_nums, frame_step, incremental=False):
        # mask = cv2.resize(mask, frame.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)

        sample = {
            "current_img": frame,
            "current_label": mask,
        }

        sample = self.transform(sample)
        frame = sample[0]["current_img"].unsqueeze(0).float().cuda(self.gpu_id)
        mask = sample[0]["current_label"].unsqueeze(0).float().cuda(self.gpu_id)
        _mask = F.interpolate(mask, size=frame.shape[-2:], mode="nearest")

        if incremental:
            self.engine.add_reference_frame_incremental(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)
        else:
            self.engine.add_reference_frame(frame, _mask, obj_nums=obj_nums, frame_step=frame_step)

    @torch.no_grad()
    def track(self, image):
        output_height, output_width = image.shape[0], image.shape[1]
        sample = {"current_img": image}
        sample = self.transform(sample)
        image = sample[0]["current_img"].unsqueeze(0).float().cuda(self.gpu_id)
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


def get_aot(args, preloaded_model=None, use_flashpack: bool = True, flashpack_cache_dir: str = None):
    # build vos engine
    cfg = engine_config.EngineConfig(args["phase"])
    cfg.TEST_CKPT_PATH = args["model_path"]
    cfg.TEST_LONG_TERM_MEM_GAP = args["long_term_mem_gap"]
    cfg.MAX_LEN_LONG_TERM = args["max_len_long_term"]
    device = "cuda" if args["gpu_id"] == 0 else f"cuda:{args['gpu_id']}"
    tracker = AOTTracker(cfg, args["gpu_id"], device=device, preloaded_model=preloaded_model, use_flashpack=use_flashpack, flashpack_cache_dir=flashpack_cache_dir)
    return tracker
