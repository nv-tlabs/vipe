import math

import torch

NAME_PAGE = "submission3"
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)
DEPTH_BINS = torch.cat(
    (
        torch.logspace(math.log10(0.1), math.log10(180.0), steps=512),
        torch.tensor([260.0]),
    ),
    dim=0,
)
LOGERR_BINS = torch.linspace(-2, 2, steps=128 + 1)
LINERR_BINS = torch.linspace(-50, 50, steps=256 + 1)

VERBOSE = False
OUTDOOR_DATASETS = __all__ = [
    "Argoverse",
    "DDAD",
    "DrivingStereo",
    "Mapillary",
    "BDD",
    "A2D2",
    "Nuscenes",
    "Cityscape",
    "KITTI",
    "DENSE",
    "DIML",
    "NianticMapFree",
    "DL3DV",
    "KITTIMulti",
    "Waymo",
    "Argoverse2",
    "BEDLAM",
    "NeRDS360",
    "BlendedMVG",
    "MegaDepthS",
]
