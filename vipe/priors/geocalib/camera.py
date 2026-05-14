# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Implementation of the pinhole, simple radial, and simple divisional camera models."""

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from torch.func import jacfwd, vmap
from torch.nn import functional as F

from .gravity import Gravity
from .misc import TensorWrapper, autocast
from .utils import deg2rad, focal2fov, fov2focal, rad2rotmat

# flake8: noqa: E741
# mypy: ignore-errors


class BaseCamera(TensorWrapper):
    """Camera tensor class."""

    eps = 1e-3

    @autocast
    def __init__(self, data: torch.Tensor):
        """Camera parameters with shape (..., {w, h, fx, fy, cx, cy, *dist}).

        Tensor convention: (..., {w, h, fx, fy, cx, cy, pitch, roll, *dist}) where
        - w, h: image size in pixels
        - fx, fy: focal lengths in pixels
        - cx, cy: principal points in normalized image coordinates
        - dist: distortion parameters

        Args:
            data (torch.Tensor): Camera parameters with shape (..., {6, 7, 8}).
        """
        # w, h, fx, fy, cx, cy, dist
        assert data.shape[-1] in {6, 7, 8}, data.shape

        pad = data.new_zeros(data.shape[:-1] + (8 - data.shape[-1],))
        data = torch.cat([data, pad], -1) if data.shape[-1] != 8 else data
        super().__init__(data)

    @classmethod
    def from_dict(cls, param_dict: Dict[str, torch.Tensor]) -> "BaseCamera":
        """Create a Camera object from a dictionary of parameters.

        Args:
            param_dict (Dict[str, torch.Tensor]): Dictionary of parameters.

        Returns:
            Camera: Camera object.
        """
        for key, value in param_dict.items():
            if not isinstance(value, torch.Tensor):
                param_dict[key] = torch.tensor(value)

        h, w = param_dict["height"], param_dict["width"]
        cx, cy = param_dict.get("cx", w / 2), param_dict.get("cy", h / 2)

        if "f" in param_dict:
            f = param_dict["f"]
        elif "vfov" in param_dict:
            vfov = param_dict["vfov"]
            f = fov2focal(vfov, h)
        else:
            raise ValueError("Focal length or vertical field of view must be provided.")

        if "dist" in param_dict:
            k1, k2 = param_dict["dist"][..., 0], param_dict["dist"][..., 1]
        elif "k1_hat" in param_dict:
            k1 = param_dict["k1_hat"] * (f / h) ** 2

            k2 = param_dict.get("k2", torch.zeros_like(k1))
        else:
            k1 = param_dict.get("k1", torch.zeros_like(f))
            k2 = param_dict.get("k2", torch.zeros_like(f))

        fx, fy = f, f
        if "scales" in param_dict:
            fx = fx * param_dict["scales"][..., 0] / param_dict["scales"][..., 1]

        params = torch.stack([w, h, fx, fy, cx, cy, k1, k2], dim=-1)
        return cls(params)

    def pinhole(self):
        """Return the pinhole camera model."""
        return self.__class__(self._data[..., :6])

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def vfov(self) -> torch.Tensor:
        """Vertical field of view in radians."""
        return focal2fov(self.f[..., 1], self.size[..., 1])

    @property
    def hfov(self) -> torch.Tensor:
        """Horizontal field of view in radians."""
        return focal2fov(self.f[..., 0], self.size[..., 0])

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    @property
    def K(self) -> torch.Tensor:
        """Returns the self intrinsic matrix with shape (..., 3, 3)."""
        shape = self.shape + (3, 3)
        K = self._data.new_zeros(shape)
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]
        K[..., 2, 2] = 1
        return K

    def update_focal(self, delta: torch.Tensor, as_log: bool = False):
        """Update the self parameters after changing the focal length."""
        f = torch.exp(torch.log(self.f) + delta) if as_log else self.f + delta

        # clamp focal length to a reasonable range for stability during training
        min_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(150), self.size[..., 1])
        max_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(5), self.size[..., 1])
        min_f = min_f.unsqueeze(-1).expand(-1, 2)
        max_f = max_f.unsqueeze(-1).expand(-1, 2)
        f = f.clamp(min=min_f, max=max_f)

        # make sure focal ration stays the same (avoid inplace operations)
        fx = f[..., 1] * self.f[..., 0] / self.f[..., 1]
        f = torch.stack([fx, f[..., 1]], -1)

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([self.size, f, self.c, dist], -1))

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the self parameters after resizing an image."""
        scales = (scales, scales) if isinstance(scales, (int, float)) else scales
        s = scales if isinstance(scales, torch.Tensor) else self.new_tensor(scales)

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([self.size * s, self.f * s, self.c * s, dist], -1))

    def crop(self, pad: Tuple[float]):
        """Update the self parameters after cropping an image."""
        pad = pad if isinstance(pad, torch.Tensor) else self.new_tensor(pad)
        size = self.size + pad.to(self.size)
        c = self.c + pad.to(self.c) / 2

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([size, self.f, c, dist], -1))

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        size = self.size.unsqueeze(-2)
        return torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the self plane and check for visibility."""
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        """Jacobian of the projection function."""
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=self.eps)
        J = torch.stack([1 / z, zero, -x / z**2, zero, 1 / z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    def undo_scale_crop(self, data: Dict[str, torch.Tensor]):
        """Undo transforms done during scaling and cropping."""
        camera = self.crop(-data["crop_pad"]) if "crop_pad" in data else self
        return camera.scale(1.0 / data["scales"])

    @abstractmethod
    def distort(self, pts: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        raise NotImplementedError("distort() must be implemented.")

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the distortion function."""
        if wrt == "scale2pts":  # (..., 2)
            J = [
                vmap(jacfwd(lambda x: self[idx].distort(x, return_scale=True)[0]))(p2d[idx])[None]
                for idx in range(p2d.shape[0])
            ]

            return torch.cat(J, dim=0).squeeze(-3, -2)

        elif wrt == "scale2dist":  # (..., 1)
            J = []
            for idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[idx, :6], x[None]], -1)
                    return self.__class__(params).distort(p2d[idx], return_scale=True)[0]

                J.append(vmap(jacfwd(func))(self[idx].dist))

            return torch.cat(J, dim=0)

        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @abstractmethod
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        raise NotImplementedError("undistort() must be implemented.")

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        if wrt == "pts":  # (..., 2, 2)
            J = [vmap(jacfwd(lambda x: self[idx].undistort(x)[0]))(p2d[idx])[None] for idx in range(p2d.shape[0])]

            return torch.cat(J, dim=0).squeeze(-3)

        elif wrt == "dist":  # (..., 1)
            J = []
            for batch_idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[batch_idx, :6], x[None]], -1)
                    return self.__class__(params).undistort(p2d[batch_idx])[0]

                J.append(vmap(jacfwd(func))(self[batch_idx].dist))

            return torch.cat(J, dim=0)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @autocast
    def up_projection_offset(self, p2d: torch.Tensor) -> torch.Tensor:
        """Compute the offset for the up-projection."""
        return self.J_distort(p2d, wrt="scale2pts")  # (B, N, 2)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the distortion offset for up-projection."""
        if wrt == "uv":  # (B, N, 2, 2)
            J = [
                vmap(jacfwd(lambda x: self[idx].up_projection_offset(x)[0, 0]))(p2d[idx])[None]
                for idx in range(p2d.shape[0])
            ]

            return torch.cat(J, dim=0)

        elif wrt == "dist":  # (B, N, 2)
            J = []
            for batch_idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[batch_idx, :6], x[None]], -1)[None]
                    return self.__class__(params).up_projection_offset(p2d[batch_idx][None])

                J.append(vmap(jacfwd(func))(self[batch_idx].dist))

            return torch.cat(J, dim=0).squeeze(1)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    def J_denormalize(self):
        """Jacobian of the denormalization function."""
        return torch.diag_embed(self.f)  # ..., 2 x 2

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / (self.f.unsqueeze(-2))

    def J_normalize(self, p2d: torch.Tensor, wrt: str = "f"):
        """Jacobian of the normalization function."""
        # ... x N x 2 x 2
        if wrt == "f":
            J_f = -(p2d - self.c.unsqueeze(-2)) / ((self.f.unsqueeze(-2)) ** 2)
            return torch.diag_embed(J_f)
        elif wrt == "pts":
            J_pts = 1 / self.f
            return torch.diag_embed(J_pts)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    def pixel_coordinates(self) -> torch.Tensor:
        """Pixel coordinates in self frame.

        Returns:
            torch.Tensor: Pixel coordinates as a tensor of shape (B, h * w, 2).
        """
        w, h = self.size[0].unbind(-1)
        h, w = h.round().to(int), w.round().to(int)

        # create grid
        x = torch.arange(0, w, dtype=self.dtype, device=self.device)
        y = torch.arange(0, h, dtype=self.dtype, device=self.device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack((x, y), dim=-1).reshape(-1, 2)  # shape (h * w, 2)

        # add batch dimension (normalize() would broadcast but we make it explicit)
        B = self.shape[0]
        xy = xy.unsqueeze(0).expand(B, -1, -1)  # if B > 0 else xy

        return xy.to(self.device).to(self.dtype)

    @autocast
    def pixel_bearing_many(self, p3d: torch.Tensor) -> torch.Tensor:
        """Get the bearing vectors of pixel coordinates by normalizing them."""
        return F.normalize(p3d, dim=-1)

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    @autocast
    def J_world2image(self, p3d: torch.Tensor):
        """Jacobian of the world2image function."""
        p2d_proj, valid = self.project(p3d)

        J_dnorm = self.J_denormalize()
        J_dist = self.J_distort(p2d_proj)
        J_proj = self.J_project(p3d)

        J = torch.einsum("...ij,...jk,...kl->...il", J_dnorm, J_dist, J_proj)
        return J, valid

    @autocast
    def image2world(self, p2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform point in the image plane to 3D world coordinates."""
        p2d = self.normalize(p2d)
        p2d, valid = self.undistort(p2d)
        ones = p2d.new_ones(p2d.shape[:-1] + (1,))
        p3d = torch.cat([p2d, ones], -1)
        return p3d, valid

    @autocast
    def J_image2world(self, p2d: torch.Tensor, wrt: str = "f") -> Tuple[torch.Tensor, torch.Tensor]:
        """Jacobian of the image2world function."""
        if wrt == "dist":
            p2d_norm = self.normalize(p2d)
            return self.J_undistort(p2d_norm, wrt)
        elif wrt == "f":
            J_norm2f = self.J_normalize(p2d, wrt)
            p2d_norm = self.normalize(p2d)
            J_dist2norm = self.J_undistort(p2d_norm, "pts")

            return torch.einsum("...ij,...jk->...ik", J_dist2norm, J_norm2f)
        else:
            raise ValueError(f"Unknown wrt: {wrt}")

    @autocast
    def undistort_image(self, img: torch.Tensor) -> torch.Tensor:
        """Undistort an image using the distortion model."""
        assert self.shape[0] == 1, "Batch size must be 1."
        W, H = self.size.unbind(-1)
        H, W = H.int().item(), W.int().item()

        x, y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="xy")
        coords = torch.stack((x, y), dim=-1).reshape(-1, 2)

        p3d, _ = self.pinhole().image2world(coords.to(self.device).to(self.dtype))
        p2d, _ = self.world2image(p3d)

        mapx, mapy = p2d[..., 0].reshape((1, H, W)), p2d[..., 1].reshape((1, H, W))
        grid = torch.stack((mapx, mapy), dim=-1)
        grid = 2.0 * grid / torch.tensor([W - 1, H - 1]).to(grid) - 1
        return F.grid_sample(img, grid, align_corners=True)

    def get_img_from_pano(
        self,
        pano_img: torch.Tensor,
        gravity: Gravity,
        yaws: torch.Tensor = 0.0,
        resize_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render an image from a panorama.

        Args:
            pano_img (torch.Tensor): Panorama image of shape (3, H, W) in [0, 1].
            gravity (Gravity): Gravity direction of the camera.
            yaws (torch.Tensor | list, optional): Yaw angle in radians. Defaults to 0.0.
            resize_factor (torch.Tensor, optional): Resize the panorama to be a multiple of the
            field of view. Defaults to 1.

        Returns:
            torch.Tensor: Image rendered from the panorama.
        """
        B = self.shape[0]
        if B > 0:
            assert self.size[..., 0].unique().shape[0] == 1, "All images must have the same width."
            assert self.size[..., 1].unique().shape[0] == 1, "All images must have the same height."

        w, h = self.size[0].unbind(-1)
        h, w = h.round().to(int), w.round().to(int)

        if isinstance(yaws, (int, float)):
            yaws = [yaws]
        if isinstance(resize_factor, (int, float)):
            resize_factor = [resize_factor]

        yaws = yaws.to(self.dtype).to(self.device) if isinstance(yaws, torch.Tensor) else self.new_tensor(yaws)

        if isinstance(resize_factor, torch.Tensor):
            resize_factor = resize_factor.to(self.dtype).to(self.device)
        elif resize_factor is not None:
            resize_factor = self.new_tensor(resize_factor)

        assert isinstance(pano_img, torch.Tensor), "Panorama image must be a torch.Tensor."
        pano_img = pano_img if pano_img.dim() == 4 else pano_img.unsqueeze(0)  # B x 3 x H x W

        pano_imgs = []
        for i, yaw in enumerate(yaws):
            if resize_factor is not None:
                # resize the panorama such that the fov of the panorama has the same height as the
                # image
                vfov = self.vfov[i] if B != 0 else self.vfov
                scale = torch.pi / float(vfov) * float(h) / pano_img.shape[-2] * resize_factor[i]
                pano_shape = (
                    int(pano_img.shape[-2] * scale),
                    int(pano_img.shape[-1] * scale),
                )

                mode = "bicubic" if scale >= 1 else "area"
                resized_pano = F.interpolate(pano_img, size=pano_shape, mode=mode)
            else:
                # make sure to copy: resized_pano = pano_img
                resized_pano = pano_img
                pano_shape = pano_img.shape[-2:][::-1]

            pano_imgs.append((resized_pano, pano_shape))

        xy = self.pixel_coordinates()
        uv1, _ = self.image2world(xy)
        bearings = self.pixel_bearing_many(uv1)

        # rotate bearings
        R_yaw = rad2rotmat(self.new_zeros(yaw.shape), self.new_zeros(yaw.shape), yaws)
        rotated_bearings = bearings @ gravity.R @ R_yaw

        # spherical coordinates
        lon = torch.atan2(rotated_bearings[..., 0], rotated_bearings[..., 2])
        lat = torch.atan2(rotated_bearings[..., 1], torch.norm(rotated_bearings[..., [0, 2]], dim=-1))

        images = []
        for idx, (resized_pano, pano_shape) in enumerate(pano_imgs):
            min_lon, max_lon = -torch.pi, torch.pi
            min_lat, max_lat = -torch.pi / 2.0, torch.pi / 2.0
            min_x, max_x = 0, pano_shape[0] - 1.0
            min_y, max_y = 0, pano_shape[1] - 1.0

            # map Spherical Coordinates to Panoramic Coordinates
            nx = (lon[idx] - min_lon) / (max_lon - min_lon) * (max_x - min_x) + min_x
            ny = (lat[idx] - min_lat) / (max_lat - min_lat) * (max_y - min_y) + min_y

            # reshape and cast to numpy for remap
            mapx, mapy = nx.reshape((1, h, w)), ny.reshape((1, h, w))

            grid = torch.stack((mapx, mapy), dim=-1)  # Add batch dimension
            # Normalize to [-1, 1]
            grid = 2.0 * grid / torch.tensor([pano_shape[-2] - 1, pano_shape[-1] - 1]).to(grid) - 1
            # Apply grid sample
            image = F.grid_sample(resized_pano, grid, align_corners=True)
            images.append(image)

        return torch.concatenate(images, 0) if B > 0 else images[0]

    def __repr__(self):
        """Print the Camera object."""
        return f"{self.__class__.__name__} {self.shape} {self.dtype} {self.device}"


class Pinhole(BaseCamera):
    """Implementation of the pinhole camera model.

    Use this model for undistorted images.
    """

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates."""
        if return_scale:
            return p2d.new_ones(p2d.shape[:-1] + (1,))

        return p2d, p2d.new_ones((p2d.shape[0], 1)).bool()

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the distortion function."""
        if wrt == "pts":
            return torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))

        raise ValueError(f"Unknown wrt: {wrt}")

    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates."""
        return pts, pts.new_ones((pts.shape[0], 1)).bool()

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        if wrt == "pts":
            return torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))

        raise ValueError(f"Unknown wrt: {wrt}")

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset."""
        if wrt == "uv":
            return torch.zeros(p2d.shape[:-1] + (2, 2), device=p2d.device, dtype=p2d.dtype)

        raise ValueError(f"Unknown wrt: {wrt}")


class SimpleRadial(BaseCamera):
    """Implementation of the simple radial camera model.

    Use this model for weakly distorted images.

    The distortion model is 1 + k1 * r^2 where r^2 = x^2 + y^2.
    The undistortion model is 1 - k1 * r^2 estimated as in
    "An Exact Formula for Calculating Inverse Radial Lens Distortions" by Pierre Drap.
    """

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., 1)."""
        return self._data[..., 6:]

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-0.7, 0.7)):
        """Update the self parameters after changing the k1 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    @autocast
    def check_valid(self, p2d: torch.Tensor) -> torch.Tensor:
        """Check if the distorted points are valid."""
        return p2d.new_ones(p2d.shape[:-1]).bool()

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 + self.k1[..., None, None] * r2

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        if wrt == "scale2dist":  # (..., 1)
            return torch.sum(p2d**2, -1, keepdim=True)
        elif wrt == "scale2pts":  # (..., 2)
            return 2 * self.k1[..., None, None] * p2d
        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        b1 = -self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 + b1 * r2
        return p2d * radial, self.check_valid(p2d)

    @autocast
    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        b1 = -self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        if wrt == "dist":
            return -r2 * p2d
        elif wrt == "pts":
            radial = 1 + b1 * r2
            radial_diag = torch.diag_embed(radial.expand(radial.shape[:-1] + (2,)))
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            return (2 * b1[..., None] * ppT) + radial_diag
        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset."""
        if wrt == "uv":  # (..., 2, 2)
            return torch.diag_embed((2 * self.k1[..., None, None]).expand(p2d.shape[:-1] + (2,)))
        elif wrt == "dist":
            return 2 * p2d  # (..., 2)
        else:
            return super().J_up_projection_offset(p2d, wrt)


class SimpleDivisional(BaseCamera):
    """Implementation of the simple divisional camera model.

    Use this model for strongly distorted images.
    k1 should be negative mostly.

    The distortion model is (1 - sqrt(1 - 4 * k1 * r^2)) / (2 * k1 * r^2) where r^2 = x^2 + y^2.
    The undistortion model is 1 / (1 + k1 * r^2).
    """

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., 1)."""
        return self._data[..., 6:]

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-3.0, 3.0)):
        """Update the self parameters after changing the k1 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    @autocast
    def check_valid(self, p2d: torch.Tensor) -> torch.Tensor:
        """Check if the distorted points are valid."""
        return p2d.new_ones(p2d.shape[:-1]).bool()

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 - torch.sqrt((1 - 4 * self.k1[..., None, None] * r2).clamp(min=0))
        denom = 2 * self.k1[..., None, None] * r2

        ones = radial.new_ones(radial.shape)
        radial = torch.where(denom == 0, ones, radial / denom.masked_fill(denom == 0, 1e6))

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        t0 = torch.sqrt((1 - 4 * self.k1[..., None, None] * r2).clamp(min=1e-6))
        if wrt == "scale2pts":  # (B, N, 2)
            d1 = t0 * 2 * r2
            d2 = self.k1[..., None, None] * r2**2
            denom = d1 * d2
            return p2d * (4 * d2 - (1 - t0) * d1) / denom.masked_fill(denom == 0, 1e6)

        elif wrt == "scale2dist":
            d1 = 2 * self.k1[..., None, None] * t0
            d2 = 2 * r2 * self.k1[..., None, None] ** 2
            denom = d1 * d2
            return (2 * d2 - (1 - t0) * d1) / denom.masked_fill(denom == 0, 1e6)

        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        denom = 1 + self.k1[..., None, None] * r2
        radial = 1 / denom.masked_fill(denom == 0, 1e6)
        return p2d * radial, self.check_valid(p2d)

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        # return super().J_undistort(p2d, wrt)
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1 = self.k1[..., None, None]
        if wrt == "dist":
            denom = (1 + k1 * r2) ** 2
            return -r2 / denom.masked_fill(denom == 0, 1e6) * p2d
        elif wrt == "pts":
            t0 = 1 + k1 * r2
            t0 = t0.masked_fill(t0 == 0, 1e6)
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            J = torch.diag_embed((1 / t0).expand(p2d.shape[:-1] + (2,)))
            return J - 2 * k1[..., None] * ppT / t0[..., None] ** 2  # (..., N, 2, 2)

        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset.

        func(uv, dist) = 4 / (2 * norm2(uv)^2 * (1-4*k1*norm2(uv)^2)^0.5) * uv
        - (1-(1-4*k1*norm2(uv)^2)^0.5) / (k1 * norm2(uv)^4) * uv
        """
        k1 = self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        t0 = (1 - 4 * k1 * r2).clamp(min=1e-6)
        t1 = torch.sqrt(t0)
        if wrt == "dist":
            denom = 4 * t0 ** (3 / 2)
            denom = denom.masked_fill(denom == 0, 1e6)
            J = 16 / denom

            denom = r2 * t1 * k1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 2 / denom

            denom = (r2 * k1) ** 2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (1 - t1) / denom

            return J * p2d
        elif wrt == "uv":
            # ! unstable (gradient checker might fail), rewrite to use single division (by denom)
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)

            denom = 2 * r2 * t1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = torch.diag_embed((4 / denom).expand(p2d.shape[:-1] + (2,)))

            denom = 4 * t1 * r2**2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 16 / denom[..., None] * ppT

            denom = 4 * r2 * t0 ** (3 / 2)
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (32 * k1[..., None]) / denom[..., None] * ppT

            denom = r2**2 * t1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 4 / denom[..., None] * ppT

            denom = k1 * r2**3
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (4 * (1 - t1) / denom)[..., None] * ppT

            denom = k1 * r2**2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - torch.diag_embed(((1 - t1) / denom).expand(p2d.shape[:-1] + (2,)))

            return J
        else:
            return super().J_up_projection_offset(p2d, wrt)


class SimpleMei(BaseCamera):
    """Implementation of the Mei's unified camera model.

    Distortion model (related to proj) is 1 / (1 + k1 * sqrt(1 + r^2))
    Undistortion model (related to iproj) is:
        let f = (k1 + sqrt(1 + (1 - k1**2) * r^2)) / (1 + r^2)
        model is then f / (f - k1)
    k1 is not restricted to a specific range.

    NOTE: This model is able to represent very distorted cameras, but note
    that the slope of the distortion function at 0 is not 1, so the distorted
    image plane will be zoomed even for the region around the optical center.
    In practice, this:
        - *might* cause problem for LM optimization.
        - needs special care for projecting to nearest pinhole (cannot just remove the dist param).
    """

    @staticmethod
    def build_from_other(camera: BaseCamera, max_r: float = 1.0, resolution: int = 50) -> "SimpleMei":
        import numpy as np
        from scipy.optimize import curve_fit

        r_samples = np.linspace(0, max_r, resolution)
        r_pcd = np.pad(r_samples[:, None], (1, 0), mode="constant", constant_values=0)[1:][None]
        ref_radial = camera.distort(torch.tensor(r_pcd, dtype=camera.dtype).to(camera.device), return_scale=True)[0]
        ref_radial = ref_radial.cpu().numpy()[0, :, 0]

        def mei_model(r, k1, f):
            return 1 / (1 + k1 * np.sqrt(1 + r**2)) * f

        popt, pcov = curve_fit(mei_model, r_samples, ref_radial)
        opt_k = popt[0]
        opt_f = popt[1]
        data = torch.cat(
            [
                camera.size,
                camera.f * opt_f,
                camera.c,
                torch.tensor(opt_k).to(camera.device).to(camera.dtype)[None, None],
            ],
            -1,
        )

        return SimpleMei(data)

    def pinhole(self):
        original_data = self._data[..., :6].clone()
        # breakpoint()
        scale = 1 / (1 + self.k1)
        # breakpoint()
        original_data[:, 2:4] *= scale
        return SimpleMei(original_data)

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., 1)."""
        return self._data[..., 6:]

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-10.0, 10.0)):
        """Update the self parameters after changing the k1 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)

        # After updating dist, we have to rescale the focal so that it's similar
        # new_f = self.f / (1 + self.dist) * (1 + dist)
        # data = torch.cat([self.size, new_f, self.c, dist], -1)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        k1 = self.k1[..., None, None] if len(p2d.shape) != 1 else self.k1

        r2 = torch.sum(p2d**2, -1, keepdim=True)
        denom = 1 + k1 * torch.sqrt(1 + r2)
        # radial = denom.reciprocal()   # Should be well-behaved since >1
        radial = 1.0 / denom

        if return_scale:
            return radial, None

        return p2d * radial, p2d.new_ones(p2d.shape[:-1]).bool()

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        sqrt_r21 = torch.sqrt(r2 + 1)
        d2 = (1 + self.k1[..., None, None] * sqrt_r21) ** 2

        if wrt == "scale2pts":  # (B, N, 2)
            return -p2d * self.k1[..., None, None] / (d2 * sqrt_r21)

        elif wrt == "scale2dist":
            return -sqrt_r21 / d2

        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1 = self.k1[..., None, None]
        fac = (k1 + torch.sqrt(1 + (1 - k1**2) * r2)) / (1 + r2)
        fac = fac / (fac - k1)
        return p2d * fac, p2d.new_ones((p2d.shape[0], 1)).bool()

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1 = self.k1[..., None, None]
        f = (k1 + torch.sqrt(1 + (1 - k1**2) * r2)) / (1 + r2)
        A = torch.sqrt(1 + (1 - k1**2) * r2)
        df_dk = (1 - k1 * r2 / A) / (1 + r2)

        if wrt == "dist":
            ds_sk = (-df_dk * k1 + f) / (f - k1) ** 2
            res = ds_sk * p2d
            return res
        elif wrt == "pts":
            u, v = p2d.unbind(-1)
            u, v = u[..., None], v[..., None]

            dA_du = (1 - k1**2) * u / A
            df_du = (dA_du * (1 + r2) - (k1 + A) * 2 * u) / (1 + r2) ** 2
            ds_du = -k1 * df_du / (f - k1) ** 2
            dA_dv = (1 - k1**2) * v / A
            df_dv = (dA_dv * (1 + r2) - (k1 + A) * 2 * v) / (1 + r2) ** 2
            ds_dv = -k1 * df_dv / (f - k1) ** 2
            s = f / (f - k1)

            J = torch.cat(
                [
                    ds_du * u + s,
                    ds_du * v,
                    ds_dv * u,
                    ds_dv * v + s,
                ],
                dim=-1,
            ).reshape(p2d.shape[:-1] + (2, 2))
            return J

        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1 = self.k1[..., None, None]
        R = torch.sqrt(1 + r2)
        kRi = 1 + k1 * R
        kRi2 = kRi**2
        kRi3 = kRi**3

        if wrt == "uv":  # (..., 2, 2)
            u, v = p2d.unbind(-1)
            u, v = u[..., None], v[..., None]
            upper_c = 2 * k1 * kRi * R + kRi2
            lower_s = kRi2 * R
            lower_b = kRi3 * kRi * (R**3)
            uu = 1 / lower_s - (u**2) * upper_c / lower_b
            uv = -u * v * upper_c / lower_b
            vv = 1 / lower_s - (v**2) * upper_c / lower_b
            J = torch.cat([uu, uv, uv, vv], dim=-1).reshape(p2d.shape[:-1] + (2, 2))
            res = -k1[..., None] * J
            return res
        elif wrt == "dist":
            rhs = 2 * k1 / kRi3 - 1 / (kRi2 * R)
            res = rhs * p2d
            return res
        else:
            return super().J_up_projection_offset(p2d, wrt)


camera_models = {
    "pinhole": Pinhole,
    "simple_radial": SimpleRadial,
    "simple_divisional": SimpleDivisional,
    "simple_mei": SimpleMei,
}
