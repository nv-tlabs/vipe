# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Image loading and general conversion utilities."""

import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torchvision

# mypy: ignore-errors


def fit_to_multiple(x: torch.Tensor, multiple: int, mode: str = "center", crop: bool = False):
    """Get padding to make the image size a multiple of the given number.

    Args:
        x (torch.Tensor): Input tensor.
        multiple (int, optional): Multiple to fit to.
        crop (bool, optional): Whether to crop or pad. Defaults to False.

    Returns:
        torch.Tensor: Padding.
    """
    h, w = x.shape[-2:]

    if crop:
        pad_w = (w // multiple) * multiple - w
        pad_h = (h // multiple) * multiple - h
    else:
        pad_w = (multiple - w % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple

    if mode == "center":
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
    elif mode == "left":
        pad_l, pad_r = 0, pad_w
        pad_t, pad_b = 0, pad_h
    else:
        raise ValueError(f"Unknown mode {mode}")

    return (pad_l, pad_r, pad_t, pad_b)


def fit_features_to_multiple(
    features: torch.Tensor, multiple: int = 32, crop: bool = False
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad or crop image to a multiple of the given number.

    Args:
        features (torch.Tensor): Input features.
        multiple (int, optional): Multiple. Defaults to 32.
        crop (bool, optional): Whether to crop or pad. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: Padded features and padding.
    """
    pad = fit_to_multiple(features, multiple, crop=crop)
    return torch.nn.functional.pad(features, pad, mode="reflect"), pad


class ImagePreprocessor:
    """Preprocess images for calibration."""

    default_conf = {
        "resize": 320,  # target edge length, None for no resizing
        "edge_divisible_by": None,
        "side": "short",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "square_crop": False,
        "add_padding_mask": False,
        "resize_backend": "kornia",  # torchvision, kornia
    }

    def __init__(self, conf) -> None:
        """Initialize the image preprocessor."""
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor, interpolation: Optional[str] = None) -> dict:
        """Resize and preprocess an image, return image and resize scale."""
        h, w = img.shape[-2:]
        size = h, w

        if self.conf.square_crop:
            min_size = min(h, w)
            offset = (h - min_size) // 2, (w - min_size) // 2
            img = img[:, offset[0] : offset[0] + min_size, offset[1] : offset[1] + min_size]
            size = img.shape[-2:]

        if self.conf.resize is not None:
            if interpolation is None:
                interpolation = self.conf.interpolation
            size = self.get_new_image_size(h, w)
            img = self.resize(img, size, interpolation)

        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0].cpu(), scale[1].cpu(), 1])

        data = {
            "scales": scale,
            "image_size": np.array(size[::-1]),
            "transform": T,
            "original_image_size": np.array([w, h]),
        }

        if self.conf.edge_divisible_by is not None:
            # crop to make the edge divisible by a number
            w_, h_ = img.shape[-1], img.shape[-2]
            img, _ = fit_features_to_multiple(img, self.conf.edge_divisible_by, crop=True)
            crop_pad = torch.Tensor([img.shape[-1] - w_, img.shape[-2] - h_]).to(img)
            data["crop_pad"] = crop_pad
            data["image_size"] = np.array([img.shape[-1], img.shape[-2]])

        data["image"] = img
        return data

    def resize(self, img: torch.Tensor, size: Tuple[int, int], interpolation: str) -> torch.Tensor:
        """Resize an image using the specified backend."""
        if self.conf.resize_backend == "kornia":
            return kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=interpolation,
            )
        elif self.conf.resize_backend == "torchvision":
            return torchvision.transforms.Resize(size, antialias=self.conf.antialias)(img)
        else:
            raise ValueError(f"{self.conf.resize_backend} not implemented.")

    def load_image(self, image_path: Path) -> dict:
        """Load an image from a path and preprocess it."""
        return self(load_image(image_path))

    def get_new_image_size(self, h: int, w: int) -> Tuple[int, int]:
        """Get the new image size after resizing."""
        side = self.conf.side
        if isinstance(self.conf.resize, collections.Iterable):
            assert len(self.conf.resize) == 2
            return tuple(self.conf.resize)
        side_size = self.conf.resize
        aspect_ratio = w / h
        if side not in ("short", "long", "vert", "horz"):
            raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
        return (
            (side_size, int(side_size * aspect_ratio))
            if side == "vert" or (side != "horz" and (side == "short") ^ (aspect_ratio < 1.0))
            else (int(side_size / aspect_ratio), side_size)
        )


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Normalize and reorder the dimensions of an image tensor."""
    if image.ndim == 3:
        image = image.permute((1, 2, 0))  # CxHxW to HxWxC
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return (image.cpu().detach().numpy() * 255).astype(np.uint8)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def write_image(img: torch.Tensor, path: Path):
    """Write an image tensor to a file."""
    img = torch_image_to_numpy(img) if isinstance(img, torch.Tensor) else img
    cv2.imwrite(str(path), img[..., ::-1])


def load_image(path: Path, grayscale: bool = False, return_tensor: bool = True) -> torch.Tensor:
    """Load an image from a path and return as a tensor."""
    image = read_image(path, grayscale=grayscale)
    if return_tensor:
        return numpy_image_to_torch(image)

    assert image.ndim in [2, 3], f"Not an image: {image.shape}"
    image = image[None] if image.ndim == 2 else image
    return torch.tensor(image.copy(), dtype=torch.uint8)


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).

    Args:
        (torch.Tensor): Vector of size (..., 3).

    Returns:
        (torch.Tensor): Skew-symmetric matrix of size (..., 3, 3).
    """
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def rad2rotmat(roll: torch.Tensor, pitch: torch.Tensor, yaw: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert (batched) roll, pitch, yaw angles (in radians) to rotation matrix.

    Args:
        roll (torch.Tensor): Roll angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        yaw (torch.Tensor, optional): Yaw angle in radians. Defaults to None.

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3).
    """
    if yaw is None:
        yaw = roll.new_zeros(roll.shape)

    Rx = pitch.new_zeros(pitch.shape + (3, 3))
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = torch.cos(pitch)
    Rx[..., 1, 2] = torch.sin(pitch)
    Rx[..., 2, 1] = -torch.sin(pitch)
    Rx[..., 2, 2] = torch.cos(pitch)

    Ry = yaw.new_zeros(yaw.shape + (3, 3))
    Ry[..., 0, 0] = torch.cos(yaw)
    Ry[..., 0, 2] = -torch.sin(yaw)
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = torch.sin(yaw)
    Ry[..., 2, 2] = torch.cos(yaw)

    Rz = roll.new_zeros(roll.shape + (3, 3))
    Rz[..., 0, 0] = torch.cos(roll)
    Rz[..., 0, 1] = torch.sin(roll)
    Rz[..., 1, 0] = -torch.sin(roll)
    Rz[..., 1, 1] = torch.cos(roll)
    Rz[..., 2, 2] = 1

    return Rz @ Rx @ Ry


def fov2focal(fov: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute focal length from (vertical/horizontal) field of view."""
    return size / 2 / torch.tan(fov / 2)


def focal2fov(focal: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute (vertical/horizontal) field of view from focal length."""
    return 2 * torch.arctan(size / (2 * focal))


def pitch2rho(pitch: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the distance from principal point to the horizon."""
    return torch.tan(pitch) * f / h


def rho2pitch(rho: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the pitch angle from the distance to the horizon."""
    return torch.atan(rho * h / f)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees."""
    return rad / torch.pi * 180


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    """Convert degrees to radians."""
    return deg / 180 * torch.pi


def get_device() -> str:
    """Get the device (cpu, cuda, mps) available."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def print_calibration(results: Dict[str, torch.Tensor]) -> None:
    """Print the calibration results."""
    camera, gravity = results["camera"], results["gravity"]
    vfov = rad2deg(camera.vfov)
    roll, pitch = rad2deg(gravity.rp).unbind(-1)

    print("\nEstimated parameters (Pred):")
    print(f"Roll:  {roll.item():.1f}° (± {rad2deg(results['roll_uncertainty']).item():.1f})°")
    print(f"Pitch: {pitch.item():.1f}° (± {rad2deg(results['pitch_uncertainty']).item():.1f})°")
    print(f"vFoV:  {vfov.item():.1f}° (± {rad2deg(results['vfov_uncertainty']).item():.1f})°")
    print(f"Focal: {camera.f[0, 1].item():.1f} px (± {results['focal_uncertainty'].item():.1f} px)")

    if hasattr(camera, "k1"):
        print(f"K1:    {camera.k1.item():.1f}")
