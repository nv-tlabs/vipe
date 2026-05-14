# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""GeoCalib model definition."""

import logging
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .lm_optimizer import LMOptimizer
from .modules import MSCAN, ConvModule, LightHamHead

# mypy: ignore-errors

logger = logging.getLogger(__name__)


class LowLevelEncoder(nn.Module):
    """Very simple low-level encoder."""

    def __init__(self):
        """Simple low-level encoder."""
        super().__init__()
        self.in_channel = 3
        self.feat_dim = 64

        self.conv1 = ConvModule(self.in_channel, self.feat_dim, kernel_size=3, padding=1)
        self.conv2 = ConvModule(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = data["image"]

        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0, (
            "Image size must be multiple of 32 if not using single image input."
        )

        c1 = self.conv1(x)
        c2 = self.conv2(c1)

        return {"features": c2}


class UpDecoder(nn.Module):
    """Minimal implementation of UpDecoder."""

    def __init__(self):
        """Up decoder."""
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_up = nn.Conv2d(self.decoder.out_channels, 2, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x, log_confidence = self.decoder(data["features"])
        up = self.linear_pred_up(x)
        return {
            "up_field": F.normalize(up, dim=1),
            "up_confidence": torch.sigmoid(log_confidence),
        }


class LatitudeDecoder(nn.Module):
    """Minimal implementation of LatitudeDecoder."""

    def __init__(self):
        """Latitude decoder."""
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_latitude = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x, log_confidence = self.decoder(data["features"])
        eps = 1e-5  # avoid nan in backward of asin
        lat = torch.tanh(self.linear_pred_latitude(x))
        lat = torch.asin(torch.clamp(lat, -1 + eps, 1 - eps))
        return {
            "latitude_field": lat,
            "latitude_confidence": torch.sigmoid(log_confidence),
        }


class PerspectiveDecoder(nn.Module):
    """Minimal implementation of PerspectiveDecoder."""

    def __init__(self):
        """Perspective decoder wrapping up and latitude decoders."""
        super().__init__()
        self.up_head = UpDecoder()
        self.latitude_head = LatitudeDecoder()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.up_head(data) | self.latitude_head(data)


class GeoCalib(nn.Module):
    """GeoCalib inference model."""

    def __init__(self, **optimizer_options):
        """Initialize the GeoCalib inference model.

        Args:
            optimizer_options: Options for the lm optimizer.
        """
        super().__init__()
        self.backbone = MSCAN()
        self.ll_enc = LowLevelEncoder()
        self.perspective_decoder = PerspectiveDecoder()

        self.optimizer = LMOptimizer({**optimizer_options})

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        features = {
            "hl": self.backbone(data)["features"],
            "ll": self.ll_enc(data)["features"],
        }
        out = self.perspective_decoder({"features": features})

        out |= {k: data[k] for k in ["image", "scales", "prior_gravity", "prior_focal", "prior_k1"] if k in data}

        out |= self.optimizer(out)

        return out

    def flexible_load(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a checkpoint with flexible key names."""
        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self.named_parameters()))

        if dict_params == model_params:  # perfect fit
            logger.info("Loading all parameters of the checkpoint.")
            self.load_state_dict(state_dict, strict=True)
            return
        elif len(dict_params & model_params) == 0:  # perfect mismatch

            def strip_prefix(x):
                return ".".join(x.split(".")[:1] + x.split(".")[2:])

            state_dict = {strip_prefix(n): p for n, p in state_dict.items()}
            dict_params = set(state_dict.keys())
            if len(dict_params & model_params) == 0:
                raise ValueError(
                    "Could not manage to load the checkpoint withparameters:" + "\n\t".join(sorted(dict_params))
                )
        common_params = dict_params & model_params
        left_params = dict_params - model_params
        left_params = [p for p in left_params if "running" not in p and "num_batches_tracked" not in p]
        logger.debug("Loading parameters:\n\t" + "\n\t".join(sorted(common_params)))
        if left_params:
            # ignore running stats of batchnorm
            logger.warning("Could not load parameters:\n\t" + "\n\t".join(sorted(left_params)))
        self.load_state_dict(state_dict, strict=False)
