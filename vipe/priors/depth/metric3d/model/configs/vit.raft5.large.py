# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See THIRD_PARTY_LICENSES.md for details.

from omegaconf import DictConfig

config = DictConfig(
    {
        "model": {
            "backbone": {
                "type": "vit_large_reg",
                "prefix": "backbones.",
                "out_channels": [1024, 1024, 1024, 1024],
                "drop_path_rate": 0.0,
            },
            "type": "DensePredModel",
            "decode_head": {
                "type": "RAFTDepthNormalDPT5",
                "in_channels": [1024, 1024, 1024, 1024],
                "use_cls_token": True,
                "feature_channels": [256, 512, 1024, 1024],
                "decoder_channels": [128, 256, 512, 1024, 1024],
                "up_scale": 7,
                "hidden_channels": [128, 128, 128, 128],
                "n_gru_layers": 3,
                "n_downsample": 2,
                "iters": 8,
                "slow_fast_gru": True,
                "num_register_tokens": 4,
                "prefix": "decode_heads.",
                "detach": False,
            },
        },
        "data_basic": {
            "canonical_space": {"img_size": (540, 960), "focal_length": 1000.0},
            "depth_range": (0, 1),
            "depth_normalize": (0.1, 200),
            "crop_size": (616, 1064),
            "clip_depth_range": (0.1, 200),
            "vit_size": (616, 1064),
        },
        "load_from": None,
        "cudnn_benchmark": True,
        "test_metrics": [
            "abs_rel",
            "rmse",
            "silog",
            "delta1",
            "delta2",
            "delta3",
            "rmse_log",
            "log10",
            "sq_rel",
        ],
        "max_value": 200,
        "batchsize_per_gpu": 1,
        "thread_per_gpu": 1,
    }
)
