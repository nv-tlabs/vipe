import argparse
import json
from pathlib import Path

import torch

from .base import OpticalFlowModel
from .sea_raft import RAFT


_HF_REPO_DEFAULT = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
_CONFIG_PATH = Path(__file__).parent / "sea_raft" / "spring-M.json"


def _load_config() -> argparse.Namespace:
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    args = argparse.Namespace()
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


class SeaRaftModel(OpticalFlowModel):
    """SEA-RAFT optical-flow estimator. Loads weights from HuggingFace.

    Args:
        repo_id: HuggingFace repo id of the pretrained SEA-RAFT model.
                 Defaults to the spring-M Tartan-C-T-TSKH checkpoint that matches
                 the bundled `spring-M.json` config.
        iters: number of refinement iterations at inference.
        device: torch device. Defaults to current CUDA device.
    """

    def __init__(
        self,
        repo_id: str = _HF_REPO_DEFAULT,
        iters: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        args = _load_config()
        if iters is not None:
            args.iters = iters
        self.args = args

        self.model = RAFT.from_pretrained(repo_id, args=args)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def estimate(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        # SEA-RAFT expects images scaled in [0, 255]: it does (img / 255 * 2 - 1) internally.
        if image1.dtype != torch.float32:
            image1 = image1.float()
            image2 = image2.float()
        if image1.max() <= 1.0 + 1e-3:
            image1 = image1 * 255.0
            image2 = image2 * 255.0

        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        out = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        return out["final"]
