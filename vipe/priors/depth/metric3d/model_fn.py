# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See THIRD_PARTY_LICENSES.md for details.

import os

import torch
import torch.nn as nn

from .model.dense_pipeline import BaseDepthModel


class DepthModel(BaseDepthModel):
    def __init__(self, cfg, **kwards):
        super(DepthModel, self).__init__(cfg)

    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.forward(data)
        return pred_depth, confidence, output_dict


def get_monodepth_model(cfg: dict, **kwargs) -> nn.Module:
    # config depth  model
    model = DepthModel(cfg, **kwargs)
    # model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model


def get_configured_monodepth_model(
    cfg: dict,
) -> nn.Module:
    """
    Args:
    @ configs: configures for the network.
    @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
    @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
    Returns:
    # model: depth model.
    """
    model = get_monodepth_model(cfg)
    return model


def config_from_py_file(file_path: str) -> dict:
    """
    Args:
    @ file_path: string representing path to config file.
    Returns:
    # config: dict containing all configs.
    """
    namespace = {}
    with open(file_path, "r") as f:
        exec(f.read(), namespace)
    return namespace["config"]


metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
    "ConvNeXt-Tiny": {
        "cfg_file": f"{metric3d_dir}/model/configs/convtiny.0.3_150.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth",
    },
    "ConvNeXt-Large": {
        "cfg_file": f"{metric3d_dir}/model/configs/convlarge.0.3_150.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth",
    },
    "ViT-Small": {
        "cfg_file": f"{metric3d_dir}/model/configs/vit.raft5.small.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
    },
    "ViT-Large": {
        "cfg_file": f"{metric3d_dir}/model/configs/vit.raft5.large.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
    },
    "ViT-giant2": {
        "cfg_file": f"{metric3d_dir}/model/configs/vit.raft5.giant2.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth",
    },
}


def metric3d_convnext_tiny(pretrain=False, **kwargs):
    """
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ConvNeXt-Tiny"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ConvNeXt-Tiny"]["ckpt_file"]

    cfg = config_from_py_file(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_convnext_large(pretrain=False, **kwargs):
    """
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ConvNeXt-Large"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ConvNeXt-Large"]["ckpt_file"]

    cfg = config_from_py_file(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_small(pretrain=False, **kwargs):
    """
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-Small"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-Small"]["ckpt_file"]

    cfg = config_from_py_file(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_large(pretrain=False, **kwargs):
    """
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-Large"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-Large"]["ckpt_file"]

    cfg = config_from_py_file(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_giant2(pretrain=False, **kwargs):
    """
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-giant2"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-giant2"]["ckpt_file"]

    cfg = config_from_py_file(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model
