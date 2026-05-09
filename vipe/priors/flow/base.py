from abc import ABC, abstractmethod

import torch


class OpticalFlowModel(ABC):
    """Abstract base for pairwise optical flow estimators."""

    @abstractmethod
    def estimate(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image1, image2: (B, 3, H, W) float tensors in [0, 1] on the model device.

        Returns:
            flow: (B, 2, H, W) float tensor on the model device.
                  flow[:, 0] is horizontal displacement (in pixels),
                  flow[:, 1] is vertical displacement (in pixels),
                  s.t. image1[..., y, x] corresponds to image2[..., y + dy, x + dx].
        """
        ...
