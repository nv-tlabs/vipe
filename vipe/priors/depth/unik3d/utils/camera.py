"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .coordinate import coords_grid
from .misc import recursive_apply, squeeze_list


def invert_pinhole(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    K_inv = torch.zeros_like(K)
    K_inv[..., 0, 0] = 1.0 / fx
    K_inv[..., 1, 1] = 1.0 / fy
    K_inv[..., 0, 2] = -cx / fx
    K_inv[..., 1, 2] = -cy / fy
    K_inv[..., 2, 2] = 1.0
    return K_inv


def unproject_pinhole(depth, K):
    b, _, h, w = depth.shape
    K_inv = invert_pinhole(K)
    grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)  # [B, 3, H, W]
    grid_flat = grid.reshape(b, -1, h * w)  # [B, 3, H*W]
    cam_coords = K_inv @ grid_flat
    pcd = cam_coords.reshape(b, -1, h, w) * depth
    return pcd


def project_pinhole(pcd, K):
    b, _, h, w = pcd.shape
    pcd_flat = pcd.reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = K @ pcd_flat
    pcd_proj = cam_coords[:, :2] / cam_coords[:, 2:].clamp(min=0.01)
    pcd_proj = pcd_proj.reshape(b, 2, h, w)
    return pcd_proj


class Camera:
    def __init__(self, params=None, K=None):
        if params.ndim == 1:
            params = params.unsqueeze(0)

        if K is None:
            K = torch.eye(3, device=params.device, dtype=params.dtype).unsqueeze(0).repeat(params.shape[0], 1, 1)
            K[..., 0, 0] = params[..., 0]
            K[..., 1, 1] = params[..., 1]
            K[..., 0, 2] = params[..., 2]
            K[..., 1, 2] = params[..., 3]

        self.params = params
        self.K = K
        self.overlap_mask = None
        self.projection_mask = None

    def project(self, xyz):
        raise NotImplementedError

    def unproject(self, uv):
        raise NotImplementedError

    def get_projection_mask(self):
        return self.projection_mask

    def get_overlap_mask(self):
        return self.overlap_mask

    def reconstruct(self, depth):
        id_coords = coords_grid(1, depth.shape[-2], depth.shape[-1], device=depth.device)
        rays = self.unproject(id_coords)
        return rays / rays[:, -1:].clamp(min=1e-4) * depth.clamp(min=1e-4)  # assumption z>0!!!

    def resize(self, factor):
        self.K[..., :2, :] *= factor
        self.params[..., :4] *= factor
        return self

    def to(self, device, non_blocking=False):
        self.params = self.params.to(device, non_blocking=non_blocking)
        self.K = self.K.to(device, non_blocking=non_blocking)
        return self

    def get_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(1, h, w, device=self.K.device, noisy=noisy)
        rays = self.unproject(uv)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def get_pinhole_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(b, h, w, device=self.K.device, homogeneous=True, noisy=noisy)
        rays = (invert_pinhole(self.K) @ uv.reshape(b, 3, -1)).reshape(b, 3, h, w)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def flip(self, H, W, direction="horizontal"):
        new_cx = W - self.params[:, 2] if direction == "horizontal" else self.params[:, 2]
        new_cy = H - self.params[:, 3] if direction == "vertical" else self.params[:, 3]
        self.params = torch.stack([self.params[:, 0], self.params[:, 1], new_cx, new_cy], dim=1)
        self.K[..., 0, 2] = new_cx
        self.K[..., 1, 2] = new_cy
        return self

    def clone(self):
        return deepcopy(self)

    def crop(self, left, top, right=None, bottom=None):
        self.K[..., 0, 2] -= left
        self.K[..., 1, 2] -= top
        self.params[..., 2] -= left
        self.params[..., 3] -= top
        return self

    # helper function to get how fov changes based on new original size and new size
    def get_new_fov(self, new_shape, original_shape):
        new_hfov = 2 * torch.atan(self.params[..., 2] / self.params[..., 0] * new_shape[1] / original_shape[1])
        new_vfov = 2 * torch.atan(self.params[..., 3] / self.params[..., 1] * new_shape[0] / original_shape[0])
        return new_hfov, new_vfov

    def mask_overlap_projection(self, projected):
        B, _, H, W = projected.shape
        id_coords = coords_grid(B, H, W, device=projected.device)

        # check for mask where flow would overlap with other part of the image
        # eleemtns coming from the border are then masked out
        flow = projected - id_coords
        gamma = 0.1
        sample_grid = gamma * flow + id_coords  # sample along the flow
        sample_grid[:, 0] = sample_grid[:, 0] / (W - 1) * 2 - 1
        sample_grid[:, 1] = sample_grid[:, 1] / (H - 1) * 2 - 1
        sampled_flow = F.grid_sample(
            flow,
            sample_grid.permute(0, 2, 3, 1),
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        mask = ((1 - gamma) * torch.norm(flow, dim=1, keepdim=True) < torch.norm(sampled_flow, dim=1, keepdim=True)) | (
            torch.norm(flow, dim=1, keepdim=True) < 1
        )
        return mask

    def _pad_params(self):
        # Ensure params are padded to length 16
        if self.params.shape[1] < 16:
            padding = torch.zeros(
                16 - self.params.shape[1],
                device=self.params.device,
                dtype=self.params.dtype,
            )
            padding = padding.view(*[(self.params.ndim - 1) * [1] + [-1]])
            padding = padding.repeat(self.params.shape[:-1] + (1,))
            return torch.cat([self.params, padding], dim=-1)
        return self.params

    @staticmethod
    def flatten_cameras(cameras):  # -> list[Camera]:
        # Recursively flatten BatchCamera into primitive cameras
        flattened_cameras = []
        for camera in cameras:
            if isinstance(camera, BatchCamera):
                flattened_cameras.extend(BatchCamera.flatten_cameras(camera.cameras))
            elif isinstance(camera, list):
                flattened_cameras.extend(camera)
            else:
                flattened_cameras.append(camera)
        return flattened_cameras

    @staticmethod
    def _stack_or_cat_cameras(cameras, func, **kwargs):
        # Generalized method to handle stacking or concatenation
        flat_cameras = BatchCamera.flatten_cameras(cameras)
        K_matrices = [camera.K for camera in flat_cameras]
        padded_params = [camera._pad_params() for camera in flat_cameras]

        stacked_K = func(K_matrices, **kwargs)
        stacked_params = func(padded_params, **kwargs)

        # Keep track of the original classes
        original_class = [x.__class__.__name__ for x in flat_cameras]
        return BatchCamera(stacked_params, stacked_K, original_class, flat_cameras)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.cat:
            return Camera._stack_or_cat_cameras(args[0], func, **kwargs)

        if func is torch.stack:
            return Camera._stack_or_cat_cameras(args[0], func, **kwargs)

        if func is torch.flatten:
            return Camera._stack_or_cat_cameras(args[0], torch.cat, **kwargs)
        return super().__torch_function__(func, types, args, kwargs)

    @property
    def device(self):
        return self.K.device

    # here we assume that cx,cy are more or less H/2 and W/2
    @property
    def hfov(self):
        return 2 * torch.atan(self.params[..., 2] / self.params[..., 0])

    @property
    def vfov(self):
        return 2 * torch.atan(self.params[..., 3] / self.params[..., 1])

    @property
    def max_fov(self):
        return 150.0 / 180.0 * np.pi, 150.0 / 180.0 * np.pi


class Pinhole(Camera):
    def __init__(self, params=None, K=None):
        assert params is not None or K is not None
        # params = [fx, fy, cx, cy]
        if params is None:
            params = torch.stack([K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]], dim=-1)
        super().__init__(params=params, K=K)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, pcd):
        b, _, h, w = pcd.shape
        pcd_flat = pcd.reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = self.K @ pcd_flat
        pcd_proj = cam_coords[:, :2] / cam_coords[:, -1:].clamp(min=0.01)
        pcd_proj = pcd_proj.reshape(b, 2, h, w)
        invalid = (pcd_proj[:, 0] < 0) & (pcd_proj[:, 0] >= w) & (pcd_proj[:, 1] < 0) & (pcd_proj[:, 1] >= h)
        self.projection_mask = (~invalid).unsqueeze(1)
        return pcd_proj

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        b, _, h, w = uv.shape
        uv_flat = uv.reshape(b, 2, -1)  # [B, 2, H*W]
        uv_homogeneous = torch.cat([uv_flat, torch.ones(b, 1, h * w, device=uv.device)], dim=1)  # [B, 3, H*W]
        K_inv = torch.inverse(self.K.float())
        xyz = K_inv @ uv_homogeneous
        xyz = xyz / xyz[:, -1:].clip(min=1e-4)
        xyz = xyz.reshape(b, 3, h, w)
        self.unprojection_mask = xyz[:, -1:] > 1e-4
        return xyz

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def reconstruct(self, depth):
        b, _, h, w = depth.shape
        uv = coords_grid(b, h, w, device=depth.device)
        xyz = self.unproject(uv) * depth.clip(min=0.0)
        return xyz


class EUCM(Camera):
    def __init__(self, params):
        # params = [fx, fy, cx, cy, alpha, beta]
        super().__init__(params=params, K=None)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, xyz):
        H, W = xyz.shape[-2:]
        fx, fy, cx, cy, alpha, beta = self.params[:6].unbind(dim=1)
        x, y, z = xyz.unbind(dim=1)
        d = torch.sqrt(beta * (x**2 + y**2) + z**2)

        x = x / (alpha * d + (1 - alpha) * z).clip(min=1e-3)
        y = y / (alpha * d + (1 - alpha) * z).clip(min=1e-3)

        Xnorm = fx * x + cx
        Ynorm = fy * y + cy

        coords = torch.stack([Xnorm, Ynorm], dim=1)

        invalid = (coords[:, 0] < 0) | (coords[:, 0] > W) | (coords[:, 1] < 0) | (coords[:, 1] > H) | (z < 0)
        self.projection_mask = (~invalid).unsqueeze(1)

        return coords

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        u, v = uv.unbind(dim=1)
        fx, fy, cx, cy, alpha, beta = self.params.unbind(dim=1)
        mx = (u - cx) / fx
        my = (v - cy) / fy
        r_square = mx**2 + my**2
        valid_mask = r_square < torch.where(alpha < 0.5, 1e6, 1 / (beta * (2 * alpha - 1)))
        sqrt_val = 1 - (2 * alpha - 1) * beta * r_square
        mz = (1 - beta * (alpha**2) * r_square) / (alpha * torch.sqrt(sqrt_val.clip(min=1e-5)) + (1 - alpha))
        coeff = 1 / torch.sqrt(mx**2 + my**2 + mz**2 + 1e-5)

        x = coeff * mx
        y = coeff * my
        z = coeff * mz
        self.unprojection_mask = valid_mask & (z > 1e-3)

        xnorm = torch.stack((x, y, z.clamp(1e-3)), dim=1)
        return xnorm


class Spherical(Camera):
    def __init__(self, params):
        # Hfov and Vofv are in radians and halved!
        # params: [fx, fy, cx, cy, W, H, HFoV/2, VFoV/2]
        # fx,fy,cx,cy = dummy values
        super().__init__(params=params, K=None)

    def resize(self, factor):
        self.K[..., :2, :] *= factor
        self.params[..., :6] *= factor
        return self

    def crop(self, left, top, right, bottom):
        self.K[..., 0, 2] -= left
        self.K[..., 1, 2] -= top
        self.params[..., 2] -= left
        self.params[..., 3] -= top
        W, H = self.params[..., 4], self.params[..., 5]
        angle_ratio_W = (W - left - right) / W
        angle_ratio_H = (H - top - bottom) / H

        self.params[..., 4] -= left + right
        self.params[..., 5] -= top + bottom

        # rescale hfov and vfov
        self.params[..., 6] *= angle_ratio_W
        self.params[..., 7] *= angle_ratio_H
        return self

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, xyz):
        width, height = self.params[..., 4], self.params[..., 5]
        hfov, vfov = 2 * self.params[..., 6], 2 * self.params[..., 7]
        longitude = torch.atan2(xyz[:, 0], xyz[:, 2])
        latitude = torch.asin(xyz[:, 1] / torch.norm(xyz, dim=1).clamp(min=1e-5))

        u = longitude / hfov * (width - 1) + (width - 1) / 2
        v = latitude / vfov * (height - 1) + (height - 1) / 2

        return torch.stack([u, v], dim=1)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        u, v = uv.unbind(dim=1)

        width, height = self.params[..., 4], self.params[..., 5]
        hfov, vfov = 2 * self.params[..., 6], 2 * self.params[..., 7]
        longitude = (u - (width - 1) / 2) / (width - 1) * hfov
        latitude = (v - (height - 1) / 2) / (height - 1) * vfov
        x = torch.cos(latitude) * torch.sin(longitude)
        z = torch.cos(latitude) * torch.cos(longitude)
        y = torch.sin(latitude)
        unit_sphere = torch.stack([x, y, z], dim=1)
        unit_sphere = unit_sphere / torch.norm(unit_sphere, dim=1, keepdim=True).clip(min=1e-5)

        return unit_sphere

    def reconstruct(self, depth):
        id_coords = coords_grid(1, depth.shape[-2], depth.shape[-1], device=depth.device)
        return self.unproject(id_coords) * depth

    def get_new_fov(self, new_shape, original_shape):
        new_hfov = 2 * self.params[..., 6] * new_shape[1] / original_shape[1]
        new_vfov = 2 * self.params[..., 7] * new_shape[0] / original_shape[0]
        return new_hfov, new_vfov

    @property
    def hfov(self):
        return 2 * self.params[..., 6]

    @property
    def vfov(self):
        return 2 * self.params[..., 7]

    @property
    def max_fov(self):
        return 2 * np.pi, 0.9 * np.pi  # avoid strong distortion on tops


class OPENCV(Camera):
    def __init__(self, params):
        super().__init__(params=params, K=None)
        # params: [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
        self.use_radial = self.params[..., 4:10].abs().sum() > 1e-6
        assert self.params[..., 7:10].abs().sum() == 0.0, "Do not support poly division model"
        self.use_tangential = self.params[..., 10:12].abs().sum() > 1e-6
        self.use_thin_prism = self.params[..., 12:].abs().sum() > 1e-6

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, xyz):
        eps = 1e-9
        B, _, H, W = xyz.shape
        N = H * W
        xyz = xyz.permute(0, 2, 3, 1).reshape(B, N, 3)

        # Radial correction.
        z = xyz[:, :, 2].reshape(B, N, 1)
        z = torch.where(torch.abs(z) < eps, eps * torch.sign(z), z)
        ab = xyz[:, :, :2] / z
        r = torch.norm(ab, dim=-1, p=2, keepdim=True)
        th = r

        th_pow = torch.cat(
            [torch.pow(th, 2 + i * 2) for i in range(3)], dim=-1
        )  # Create powers of th (th^3, th^5, ...)
        distortion_coeffs_num = self.params[:, 4:7].reshape(B, 1, 3)
        distortion_coeffs_den = self.params[:, 7:10].reshape(B, 1, 3)
        th_num = 1 + torch.sum(th_pow * distortion_coeffs_num, dim=-1, keepdim=True)
        th_den = 1 + torch.sum(th_pow * distortion_coeffs_den, dim=-1, keepdim=True)

        xr_yr = ab * th_num / th_den
        uv_dist = xr_yr

        # Tangential correction.
        p0 = self.params[..., -6].reshape(B, 1)
        p1 = self.params[..., -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
        uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
        uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)  # Avoids in-place complaint.

        # Thin Prism correction.
        s0 = self.params[..., -4].reshape(B, 1)
        s1 = self.params[..., -3].reshape(B, 1)
        s2 = self.params[..., -2].reshape(B, 1)
        s3 = self.params[..., -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

        # Finally, apply standard terms: focal length and camera centers.
        if self.params.shape[-1] == 15:
            fx_fy = self.params[..., 0].reshape(B, 1, 1)
            cx_cy = self.params[..., 1:3].reshape(B, 1, 2)
        else:
            fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
            cx_cy = self.params[..., 2:4].reshape(B, 1, 2)
        result = uv_dist * fx_fy + cx_cy

        result = result.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        invalid = (result[:, 0] < 0) | (result[:, 0] > W) | (result[:, 1] < 0) | (result[:, 1] > H)
        self.projection_mask = (~invalid).unsqueeze(1)
        self.overlap_mask = self.mask_overlap_projection(result)

        return result

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv, max_iters: int = 10):
        eps = 1e-3
        B, _, H, W = uv.shape
        N = H * W
        uv = uv.permute(0, 2, 3, 1).reshape(B, N, 2)

        if self.params.shape[-1] == 15:
            fx_fy = self.params[..., 0].reshape(B, 1, 1)
            cx_cy = self.params[..., 1:3].reshape(B, 1, 2)
        else:
            fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
            cx_cy = self.params[..., 2:4].reshape(B, 1, 2)

        uv_dist = (uv - cx_cy) / fx_fy

        # Compute xr_yr using Newton's method.
        xr_yr = uv_dist.clone()  # Initial guess.
        max_iters_tanprism = max_iters if self.use_thin_prism or self.use_tangential else 0

        for _ in range(max_iters_tanprism):
            uv_dist_est = xr_yr.clone()
            xr = xr_yr[..., 0].reshape(B, N)
            yr = xr_yr[..., 1].reshape(B, N)
            xr_yr_sq = torch.square(xr_yr)
            xr_sq = xr_yr_sq[..., 0].reshape(B, N)
            yr_sq = xr_yr_sq[..., 1].reshape(B, N)
            rd_sq = xr_sq + yr_sq

            if self.use_tangential:
                # Tangential terms.
                p0 = self.params[..., -6].reshape(B, 1)
                p1 = self.params[..., -5].reshape(B, 1)
                uv_dist_est[..., 0] = uv_dist_est[..., 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
                uv_dist_est[..., 1] = uv_dist_est[..., 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)

            if self.use_thin_prism:
                # Thin Prism terms.
                s0 = self.params[..., -4].reshape(B, 1)
                s1 = self.params[..., -3].reshape(B, 1)
                s2 = self.params[..., -2].reshape(B, 1)
                s3 = self.params[..., -1].reshape(B, 1)
                rd_4 = torch.square(rd_sq)
                uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
                uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

            # Compute the derivative of uv_dist w.r.t. xr_yr.
            duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)

            if self.use_tangential:
                duv_dist_dxr_yr[..., 0, 0] = 1.0 + 6.0 * xr * p0 + 2.0 * yr * p1
                offdiag = 2.0 * (xr * p1 + yr * p0)
                duv_dist_dxr_yr[..., 0, 1] = offdiag
                duv_dist_dxr_yr[..., 1, 0] = offdiag
                duv_dist_dxr_yr[..., 1, 1] = 1.0 + 6.0 * yr * p1 + 2.0 * xr * p0

            if self.use_thin_prism:
                xr_yr_sq_norm = xr_sq + yr_sq
                temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
                duv_dist_dxr_yr[..., 0, 0] = duv_dist_dxr_yr[..., 0, 0] + (xr * temp1)
                duv_dist_dxr_yr[..., 0, 1] = duv_dist_dxr_yr[..., 0, 1] + (yr * temp1)
                temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
                duv_dist_dxr_yr[..., 1, 0] = duv_dist_dxr_yr[..., 1, 0] + (xr * temp2)
                duv_dist_dxr_yr[..., 1, 1] = duv_dist_dxr_yr[..., 1, 1] + (yr * temp2)

            mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
            a = mat[:, 0, 0].reshape(-1, 1, 1)
            b = mat[:, 0, 1].reshape(-1, 1, 1)
            c = mat[:, 1, 0].reshape(-1, 1, 1)
            d = mat[:, 1, 1].reshape(-1, 1, 1)
            det = 1.0 / ((a * d) - (b * c))
            top = torch.cat([d, -b], dim=-1)
            bot = torch.cat([-c, a], dim=-1)
            inv = det * torch.cat([top, bot], dim=-2)
            inv = inv.reshape(B, N, 2, 2)
            diff = uv_dist - uv_dist_est
            a = inv[..., 0, 0]
            b = inv[..., 0, 1]
            c = inv[..., 1, 0]
            d = inv[..., 1, 1]
            e = diff[..., 0]
            f = diff[..., 1]
            step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
            # Newton step.
            xr_yr = xr_yr + step

        # Compute theta using Newton's method.
        xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
        th = xr_yr_norm.clone()
        max_iters_radial = max_iters if self.use_radial else 0
        c = torch.tensor([2.0 * i + 3 for i in range(3)], device=self.device).reshape(1, 1, 3).repeat(B, 1, 1)
        radial_params_num = self.params[..., 4:7].reshape(B, 1, 3)

        # Trust region parameters
        delta = torch.full((B, N, 1), 0.1, device=self.device)  # Initial trust radius
        delta_max = torch.tensor(1.0, device=self.device)  # Maximum trust radius
        eta = 0.1  # Acceptable reduction threshold

        for i in range(max_iters_radial):
            th_sq = th * th  # th^2
            # Compute powers of th^2 up to th^(12)
            theta_powers = torch.cat([th_sq ** (i + 1) for i in range(3)], dim=-1)  # Shape: (B, N, 6)

            # Compute th_radial: radial distortion model applied to th
            th_radial = 1.0 + torch.sum(theta_powers * radial_params_num, dim=-1, keepdim=True)
            th_radial = th_radial * th  # Multiply by th at the end

            # Compute derivative dthd_th
            dthd_th = 1.0 + torch.sum(c * radial_params_num * theta_powers, dim=-1, keepdim=True)
            dthd_th = dthd_th  # Already includes derivative terms

            # Compute residual
            residual = th_radial - xr_yr_norm  # Shape: (B, N, 1)
            residual_norm = torch.norm(residual, dim=2, keepdim=True)  # For each pixel

            # Check for convergence
            if torch.max(torch.abs(residual)) < eps:
                break

            # Avoid division by zero by adding a small epsilon
            safe_dthd_th = dthd_th.clone()
            zero_derivative_mask = dthd_th.abs() < eps
            safe_dthd_th[zero_derivative_mask] = eps

            # Compute Newton's step
            step = -residual / safe_dthd_th

            # Compute predicted reduction
            predicted_reduction = -(residual * step).sum(dim=2, keepdim=True)

            # Adjust step based on trust region
            step_norm = torch.norm(step, dim=2, keepdim=True)
            over_trust_mask = step_norm > delta

            # Scale step if it exceeds trust radius
            step_scaled = step.clone()
            step_scaled[over_trust_mask] = step[over_trust_mask] * (delta[over_trust_mask] / step_norm[over_trust_mask])

            # Update theta
            th_new = th + step_scaled

            # Compute new residual
            th_sq_new = th_new * th_new
            theta_powers_new = torch.cat([th_sq_new ** (j + 1) for j in range(3)], dim=-1)
            th_radial_new = 1.0 + torch.sum(theta_powers_new * radial_params_num, dim=-1, keepdim=True)
            th_radial_new = th_radial_new * th_new
            residual_new = th_radial_new - xr_yr_norm
            residual_new_norm = torch.norm(residual_new, dim=2, keepdim=True)

            # Compute actual reduction
            actual_reduction = residual_norm - residual_new_norm

            # Compute ratio of actual to predicted reduction
            # predicted_reduction[predicted_reduction.abs() < eps] = eps #* torch.sign(predicted_reduction[predicted_reduction.abs() < eps])
            rho = actual_reduction / predicted_reduction
            rho[(actual_reduction == 0) & (predicted_reduction == 0)] = 1.0

            # Update trust radius delta
            delta_update_mask = rho > 0.5
            delta[delta_update_mask] = torch.min(2.0 * delta[delta_update_mask], delta_max)

            delta_decrease_mask = rho < 0.2
            delta[delta_decrease_mask] = 0.25 * delta[delta_decrease_mask]

            # Accept or reject the step
            accept_step_mask = rho > eta
            th = torch.where(accept_step_mask, th_new, th)

        # Compute the ray direction using theta and xr_yr.
        close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
        ray_dir = torch.where(close_to_zero, xr_yr, th / xr_yr_norm * xr_yr)

        ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
        ray = ray.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return ray


class Fisheye624(Camera):
    def __init__(self, params):
        super().__init__(params=params, K=None)
        # params: [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
        self.use_radial = self.params[..., 4:10].abs().sum() > 1e-6
        self.use_tangential = self.params[..., 10:12].abs().sum() > 1e-6
        self.use_thin_prism = self.params[..., 12:].abs().sum() > 1e-6

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, xyz):
        eps = 1e-9
        B, _, H, W = xyz.shape
        N = H * W
        xyz = xyz.permute(0, 2, 3, 1).reshape(B, N, 3)

        # Radial correction.
        z = xyz[:, :, 2].reshape(B, N, 1)
        z = torch.where(torch.abs(z) < eps, eps * torch.sign(z), z)
        ab = xyz[:, :, :2] / z
        r = torch.norm(ab, dim=-1, p=2, keepdim=True)
        th = torch.atan(r)
        th_divr = torch.where(r < eps, torch.ones_like(ab), ab / r)

        th_pow = torch.cat(
            [torch.pow(th, 3 + i * 2) for i in range(6)], dim=-1
        )  # Create powers of th (th^3, th^5, ...)
        distortion_coeffs = self.params[:, 4:10].reshape(B, 1, 6)
        th_k = th + torch.sum(th_pow * distortion_coeffs, dim=-1, keepdim=True)

        xr_yr = th_k * th_divr
        uv_dist = xr_yr

        # Tangential correction.
        p0 = self.params[..., -6].reshape(B, 1)
        p1 = self.params[..., -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
        uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
        uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)  # Avoids in-place complaint.

        # Thin Prism correction.
        s0 = self.params[..., -4].reshape(B, 1)
        s1 = self.params[..., -3].reshape(B, 1)
        s2 = self.params[..., -2].reshape(B, 1)
        s3 = self.params[..., -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

        # Finally, apply standard terms: focal length and camera centers.
        if self.params.shape[-1] == 15:
            fx_fy = self.params[..., 0].reshape(B, 1, 1)
            cx_cy = self.params[..., 1:3].reshape(B, 1, 2)
        else:
            fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
            cx_cy = self.params[..., 2:4].reshape(B, 1, 2)
        result = uv_dist * fx_fy + cx_cy

        result = result.reshape(B, H, W, 2).permute(0, 3, 1, 2)
        invalid = (result[:, 0] < 0) | (result[:, 0] > W) | (result[:, 1] < 0) | (result[:, 1] > H)
        self.projection_mask = (~invalid).unsqueeze(1)
        self.overlap_mask = self.mask_overlap_projection(result)

        return result

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv, max_iters: int = 10):
        eps = 1e-3
        B, _, H, W = uv.shape
        N = H * W
        uv = uv.permute(0, 2, 3, 1).reshape(B, N, 2)

        if self.params.shape[-1] == 15:
            fx_fy = self.params[..., 0].reshape(B, 1, 1)
            cx_cy = self.params[..., 1:3].reshape(B, 1, 2)
        else:
            fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
            cx_cy = self.params[..., 2:4].reshape(B, 1, 2)

        uv_dist = (uv - cx_cy) / fx_fy

        # Compute xr_yr using Newton's method.
        xr_yr = uv_dist.clone()  # Initial guess.
        max_iters_tanprism = max_iters if self.use_thin_prism or self.use_tangential else 0

        for _ in range(max_iters_tanprism):
            uv_dist_est = xr_yr.clone()
            xr = xr_yr[..., 0].reshape(B, N)
            yr = xr_yr[..., 1].reshape(B, N)
            xr_yr_sq = torch.square(xr_yr)
            xr_sq = xr_yr_sq[..., 0].reshape(B, N)
            yr_sq = xr_yr_sq[..., 1].reshape(B, N)
            rd_sq = xr_sq + yr_sq

            if self.use_tangential:
                # Tangential terms.
                p0 = self.params[..., -6].reshape(B, 1)
                p1 = self.params[..., -5].reshape(B, 1)
                uv_dist_est[..., 0] = uv_dist_est[..., 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
                uv_dist_est[..., 1] = uv_dist_est[..., 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)

            if self.use_thin_prism:
                # Thin Prism terms.
                s0 = self.params[..., -4].reshape(B, 1)
                s1 = self.params[..., -3].reshape(B, 1)
                s2 = self.params[..., -2].reshape(B, 1)
                s3 = self.params[..., -1].reshape(B, 1)
                rd_4 = torch.square(rd_sq)
                uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
                uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

            # Compute the derivative of uv_dist w.r.t. xr_yr.
            duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)

            if self.use_tangential:
                duv_dist_dxr_yr[..., 0, 0] = 1.0 + 6.0 * xr * p0 + 2.0 * yr * p1
                offdiag = 2.0 * (xr * p1 + yr * p0)
                duv_dist_dxr_yr[..., 0, 1] = offdiag
                duv_dist_dxr_yr[..., 1, 0] = offdiag
                duv_dist_dxr_yr[..., 1, 1] = 1.0 + 6.0 * yr * p1 + 2.0 * xr * p0

            if self.use_thin_prism:
                xr_yr_sq_norm = xr_sq + yr_sq
                temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
                duv_dist_dxr_yr[..., 0, 0] = duv_dist_dxr_yr[..., 0, 0] + (xr * temp1)
                duv_dist_dxr_yr[..., 0, 1] = duv_dist_dxr_yr[..., 0, 1] + (yr * temp1)
                temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
                duv_dist_dxr_yr[..., 1, 0] = duv_dist_dxr_yr[..., 1, 0] + (xr * temp2)
                duv_dist_dxr_yr[..., 1, 1] = duv_dist_dxr_yr[..., 1, 1] + (yr * temp2)
            # Compute 2x2 inverse manually here since torch.inverse() is very slow.
            # Because this is slow: inv = duv_dist_dxr_yr.inverse()
            # About a 10x reduction in speed with above line.
            mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
            a = mat[:, 0, 0].reshape(-1, 1, 1)
            b = mat[:, 0, 1].reshape(-1, 1, 1)
            c = mat[:, 1, 0].reshape(-1, 1, 1)
            d = mat[:, 1, 1].reshape(-1, 1, 1)
            det = 1.0 / ((a * d) - (b * c))
            top = torch.cat([d, -b], dim=-1)
            bot = torch.cat([-c, a], dim=-1)
            inv = det * torch.cat([top, bot], dim=-2)
            inv = inv.reshape(B, N, 2, 2)
            diff = uv_dist - uv_dist_est
            a = inv[..., 0, 0]
            b = inv[..., 0, 1]
            c = inv[..., 1, 0]
            d = inv[..., 1, 1]
            e = diff[..., 0]
            f = diff[..., 1]
            step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
            # Newton step.
            xr_yr = xr_yr + step

        # Compute theta using Newton's method.
        xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
        th = xr_yr_norm.clone()
        max_iters_radial = max_iters if self.use_radial else 0
        c = torch.tensor([2.0 * i + 3 for i in range(6)], device=self.device).reshape(1, 1, 6).repeat(B, 1, 1)
        radial_params = self.params[..., 4:10].reshape(B, 1, 6)

        # Trust region parameters
        delta = torch.full((B, N, 1), 0.1, device=self.device)  # Initial trust radius
        delta_max = torch.tensor(1.0, device=self.device)  # Maximum trust radius
        eta = 0.1  # Acceptable reduction threshold

        for i in range(max_iters_radial):
            th_sq = th * th
            # Compute powers of th^2 up to th^(12)
            theta_powers = torch.cat([th_sq ** (i + 1) for i in range(6)], dim=-1)  # Shape: (B, N, 6)

            # Compute th_radial: radial distortion model applied to th
            th_radial = 1.0 + torch.sum(theta_powers * radial_params, dim=-1, keepdim=True)
            th_radial = th_radial * th

            # Compute derivative dthd_th
            dthd_th = 1.0 + torch.sum(c * radial_params * theta_powers, dim=-1, keepdim=True)
            dthd_th = dthd_th

            # Compute residual
            residual = th_radial - xr_yr_norm  # Shape: (B, N, 1)
            residual_norm = torch.norm(residual, dim=2, keepdim=True)

            # Check for convergence
            if torch.max(torch.abs(residual)) < eps:
                break

            # Avoid division by zero by adding a small epsilon
            safe_dthd_th = dthd_th.clone()
            zero_derivative_mask = dthd_th.abs() < eps
            safe_dthd_th[zero_derivative_mask] = eps

            # Compute Newton's step
            step = -residual / safe_dthd_th

            # Compute predicted reduction
            predicted_reduction = -(residual * step).sum(dim=2, keepdim=True)

            # Adjust step based on trust region
            step_norm = torch.norm(step, dim=2, keepdim=True)
            over_trust_mask = step_norm > delta

            # Scale step if it exceeds trust radius
            step_scaled = step.clone()
            step_scaled[over_trust_mask] = step[over_trust_mask] * (delta[over_trust_mask] / step_norm[over_trust_mask])

            # Update theta
            th_new = th + step_scaled

            # Compute new residual
            th_sq_new = th_new * th_new
            theta_powers_new = torch.cat([th_sq_new ** (j + 1) for j in range(6)], dim=-1)
            th_radial_new = 1.0 + torch.sum(theta_powers_new * radial_params, dim=-1, keepdim=True)
            th_radial_new = th_radial_new * th_new
            residual_new = th_radial_new - xr_yr_norm
            residual_new_norm = torch.norm(residual_new, dim=2, keepdim=True)

            # Compute actual reduction
            actual_reduction = residual_norm - residual_new_norm

            # Compute ratio of actual to predicted reduction
            rho = actual_reduction / predicted_reduction
            rho[(actual_reduction == 0) & (predicted_reduction == 0)] = 1.0

            # Update trust radius delta
            delta_update_mask = rho > 0.5
            delta[delta_update_mask] = torch.min(2.0 * delta[delta_update_mask], delta_max)

            delta_decrease_mask = rho < 0.2
            delta[delta_decrease_mask] = 0.25 * delta[delta_decrease_mask]

            # Accept or reject the step
            accept_step_mask = rho > eta
            th = torch.where(accept_step_mask, th_new, th)

        # Compute the ray direction using theta and xr_yr.
        close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
        ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)

        ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
        ray = ray.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return ray


class MEI(Camera):
    def __init__(self, params):
        super().__init__(params=params, K=None)
        # fx fy cx cy k1 k2 p1 p2 xi
        self.use_radial = self.params[..., 4:6].abs().sum() > 1e-6
        self.use_tangential = self.params[..., 6:8].abs().sum() > 1e-6

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv, max_iters: int = 20):
        eps = 1e-6
        B, _, H, W = uv.shape
        N = H * W
        uv = uv.permute(0, 2, 3, 1).reshape(B, N, 2)

        k1, k2, p0, p1, xi = self.params[..., 4:9].unbind(dim=1)
        fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
        cx_cy = self.params[..., 2:4].reshape(B, 1, 2)

        uv_dist = (uv - cx_cy) / fx_fy

        # Compute xr_yr using Newton's method.
        xr_yr = uv_dist.clone()  # Initial guess.
        max_iters_tangential = max_iters if self.use_tangential else 0
        for _ in range(max_iters_tangential):
            uv_dist_est = xr_yr.clone()

            # Tangential terms.
            xr = xr_yr[..., 0]
            yr = xr_yr[..., 1]
            xr_yr_sq = xr_yr**2
            xr_sq = xr_yr_sq[..., 0]
            yr_sq = xr_yr_sq[..., 1]
            rd_sq = xr_sq + yr_sq
            uv_dist_est[..., 0] = uv_dist_est[..., 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
            uv_dist_est[..., 1] = uv_dist_est[..., 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)

            # Compute the derivative of uv_dist w.r.t. xr_yr.
            duv_dist_dxr_yr = torch.ones((B, N, 2, 2), device=uv.device)
            duv_dist_dxr_yr[..., 0, 0] = 1.0 + 6.0 * xr * p0 + 2.0 * yr * p1
            offdiag = 2.0 * (xr * p1 + yr * p0)
            duv_dist_dxr_yr[..., 0, 1] = offdiag
            duv_dist_dxr_yr[..., 1, 0] = offdiag
            duv_dist_dxr_yr[..., 1, 1] = 1.0 + 6.0 * yr * p1 + 2.0 * xr * p0

            mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
            a = mat[:, 0, 0].reshape(-1, 1, 1)
            b = mat[:, 0, 1].reshape(-1, 1, 1)
            c = mat[:, 1, 0].reshape(-1, 1, 1)
            d = mat[:, 1, 1].reshape(-1, 1, 1)
            det = 1.0 / ((a * d) - (b * c))
            top = torch.cat([d, -b], dim=-1)
            bot = torch.cat([-c, a], dim=-1)
            inv = det * torch.cat([top, bot], dim=-2)
            inv = inv.reshape(B, N, 2, 2)

            diff = uv_dist - uv_dist_est
            a = inv[..., 0, 0]
            b = inv[..., 0, 1]
            c = inv[..., 1, 0]
            d = inv[..., 1, 1]
            e = diff[..., 0]
            f = diff[..., 1]
            step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)

            # Newton step.
            xr_yr = xr_yr + step

        # Compute theta using Newton's method.
        xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
        th = xr_yr_norm.clone()
        max_iters_radial = max_iters if self.use_radial else 0
        for _ in range(max_iters_radial):
            th_radial = 1.0 + k1 * torch.pow(th, 2) + k2 * torch.pow(th, 4)
            dthd_th = 1.0 + 3.0 * k1 * torch.pow(th, 2) + 5.0 * k2 * torch.pow(th, 4)
            th_radial = th_radial * th
            step = (xr_yr_norm - th_radial) / dthd_th
            # handle dthd_th close to 0.
            step = torch.where(torch.abs(dthd_th) > eps, step, torch.sign(step) * eps * 10.0)
            th = th + step

        # Compute the ray direction using theta and xr_yr.
        close_to_zero = (torch.abs(th) < eps) & (torch.abs(xr_yr_norm) < eps)
        ray_dir = torch.where(close_to_zero, xr_yr, th * xr_yr / xr_yr_norm)

        # Compute the 3D projective ray
        rho2_u = ray_dir.norm(p=2, dim=2, keepdim=True) ** 2  # B N 1 # x_c * x_c + y_c * y_c
        xi = xi.reshape(B, 1, 1)
        sqrt_term = torch.sqrt(1.0 + (1.0 - xi * xi) * rho2_u)
        P_z = 1.0 - xi * (rho2_u + 1.0) / (xi + sqrt_term)

        # Special case when xi is 1.0 (unit sphere projection ??)
        P_z = torch.where(xi == 1.0, (1.0 - rho2_u) / 2.0, P_z)

        ray = torch.cat([ray_dir, P_z], dim=-1)
        ray = ray.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # remove nans
        where_nan = ray.isnan().any(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        ray = torch.where(where_nan, torch.zeros_like(ray), ray)

        return ray

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, xyz):
        is_flat = xyz.ndim == 3
        B, N = xyz.shape[:2]

        if not is_flat:
            B, _, H, W = xyz.shape
            N = H * W
            xyz = xyz.permute(0, 2, 3, 1).reshape(B, N, 3)

        k1, k2, p0, p1, xi = self.params[..., 4:].unbind(dim=1)
        fx_fy = self.params[..., 0:2].reshape(B, 1, 2)
        cx_cy = self.params[..., 2:4].reshape(B, 1, 2)

        norm = xyz.norm(p=2, dim=-1, keepdim=True)
        ab = xyz[..., :-1] / (xyz[..., -1:] + xi.reshape(B, 1, 1) * norm)

        # radial correction
        r = ab.norm(dim=-1, p=2, keepdim=True)
        k1 = self.params[..., 4].reshape(B, 1, 1)
        k2 = self.params[..., 5].reshape(B, 1, 1)
        # ab / r * th * (1 + k1 * (th ** 2) + k2 * (th**4))
        # but here r = th, no spherical distortion
        xr_yr = ab * (1 + k1 * (r**2) + k2 * (r**4))

        # Tangential correction.
        uv_dist = xr_yr
        p0 = self.params[:, -3].reshape(B, 1)
        p1 = self.params[:, -2].reshape(B, 1)
        xr = xr_yr[..., 0].reshape(B, N)
        yr = xr_yr[..., 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
        uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
        uv_dist = torch.stack([uv_dist_tu, uv_dist_tv], dim=-1)  # Avoids in-place complaint.

        result = uv_dist * fx_fy + cx_cy

        if not is_flat:
            result = result.reshape(B, H, W, 2).permute(0, 3, 1, 2)
            invalid = (result[:, 0] < 0) | (result[:, 0] > W) | (result[:, 1] < 0) | (result[:, 1] > H)
            self.projection_mask = (~invalid).unsqueeze(1)
            # creates hole in the middle... ??
            # self.overlap_mask = self.mask_overlap_projection(result)

        return result


class BatchCamera(Camera):
    def __init__(self, params, K, original_class, cameras):
        super().__init__(params, K)
        self.original_class = original_class
        self.cameras = cameras

    # Delegate these methods to original camera
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, points_3d):
        return torch.cat([camera.project(points_3d[i : i + 1]) for i, camera in enumerate(self.cameras)])

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, points_2d):
        def recursive_unproject(cameras):
            if isinstance(cameras, list):
                return [recursive_unproject(camera) for camera in cameras]
            else:
                return cameras.unproject(points_2d)

        def flatten_and_cat(nested_list):
            if isinstance(nested_list[0], list):
                return torch.cat([flatten_and_cat(sublist) for sublist in nested_list], dim=0)
            else:
                return torch.cat(nested_list, dim=0)

        unprojected = recursive_unproject(self.cameras)
        return flatten_and_cat(unprojected)

    def crop(self, left, top, right=None, bottom=None):
        val = torch.cat([camera.crop(left, top, right, bottom) for i, camera in enumerate(self.cameras)])
        return val

    def resize(self, ratio):
        val = torch.cat([camera.resize(ratio) for i, camera in enumerate(self.cameras)])
        return val

    def reconstruct(self, depth):
        val = torch.cat([camera.reconstruct(depth[i : i + 1]) for i, camera in enumerate(self.cameras)])
        return val

    def get_projection_mask(self):
        return torch.cat([camera.projection_mask for i, camera in enumerate(self.cameras)])

    def to(self, device, non_blocking=False):
        self = super().to(device, non_blocking=non_blocking)
        self.cameras = recursive_apply(self.cameras, lambda camera: camera.to(device, non_blocking=non_blocking))
        return self

    def reshape(self, *shape):
        # Reshape the intrinsic matrix (K) and params
        # we know that the shape of K is (..., 3, 3) and params is (..., 16)
        reshaped_K = self.K.reshape(*shape, 3, 3)
        reshaped_params = self.params.reshape(*shape, self.params.shape[-1])

        self.cameras = np.array(self.cameras, dtype=object).reshape(shape).tolist()
        self.original_class = np.array(self.original_class, dtype=object).reshape(shape).tolist()

        # Create a new BatchCamera with reshaped K and params
        return BatchCamera(reshaped_params, reshaped_K, self.original_class, self.cameras)

    def get_new_fov(self, new_shape, original_shape):
        return [camera.get_new_fov(new_shape, original_shape) for i, camera in enumerate(self.cameras)]

    def squeeze(self, dim):
        return BatchCamera(
            self.params.squeeze(dim),
            self.K.squeeze(dim),
            squeeze_list(self.original_class, dim=dim),
            squeeze_list(self.cameras, dim=dim),
        )

    def __getitem__(self, idx):
        # If it's an integer index, return a single camera
        if isinstance(idx, int):
            return self.cameras[idx]

        # If it's a slice, return a new BatchCamera with sliced cameras
        elif isinstance(idx, slice):
            return BatchCamera(
                self.params[idx],
                self.K[idx],
                self.original_class[idx],
                self.cameras[idx],
            )

        raise TypeError(f"Invalid index type: {type(idx)}")

    def __setitem__(self, idx, value):
        # If it's an integer index, return a single camera
        if isinstance(idx, int):
            self.cameras[idx] = value
            self.params[idx, :] = 0.0
            self.params[idx, : value.params.shape[1]] = value.params[0]
            self.K[idx] = value.K[0]

            self.original_class[idx] = getattr(value, "original_class", value.__class__.__name__)

        # If it's a slice, return a new BatchCamera with sliced cameras
        elif isinstance(idx, slice):
            # Update each internal attribute using the slice
            self.params[idx] = value.params
            self.K[idx] = value.K
            self.original_class[idx] = value.original_class
            self.cameras[idx] = value.cameras

    def __len__(self):
        return len(self.cameras)

    @classmethod
    def from_camera(cls, camera):
        return cls(camera.params, camera.K, [camera.__class__.__name__], [camera])

    @property
    def is_perspective(self):
        return recursive_apply(self.cameras, lambda camera: isinstance(camera, Pinhole))

    @property
    def is_spherical(self):
        return recursive_apply(self.cameras, lambda camera: isinstance(camera, Spherical))

    @property
    def is_eucm(self):
        return recursive_apply(self.cameras, lambda camera: isinstance(camera, EUCM))

    @property
    def is_fisheye(self):
        return recursive_apply(self.cameras, lambda camera: isinstance(camera, Fisheye624))

    @property
    def is_pinhole(self):
        return recursive_apply(self.cameras, lambda camera: isinstance(camera, Pinhole))

    @property
    def hfov(self):
        return recursive_apply(self.cameras, lambda camera: camera.hfov)

    @property
    def vfov(self):
        return recursive_apply(self.cameras, lambda camera: camera.vfov)

    @property
    def max_fov(self):
        return recursive_apply(self.cameras, lambda camera: camera.max_fov)


import json
import random


# sampler helpers
from math import log

import torch.nn as nn


def eucm(boundaries, mult, batch, device, dtype):
    alpha_min, alpha_max = boundaries[0][0] * mult, boundaries[0][1] * mult
    beta_mean, beta_std = boundaries[1][0] * mult, boundaries[1][1] * mult
    alpha = torch.rand(batch, device=device, dtype=dtype) * (alpha_max - alpha_min) + alpha_min
    beta = F.softplus(
        torch.randn(batch, device=device, dtype=dtype) * beta_std + beta_mean,
        beta=log(2),
    )
    return alpha, beta


def free_fisheye(boundaries, mult, batch, device, dtype):
    k1_min, k1_max = boundaries[0][0] * mult, boundaries[0][1] * mult
    k2_min, k2_max = boundaries[1][0] * mult, boundaries[1][1] * mult
    k3_min, k3_max = boundaries[2][0] * mult, boundaries[2][1] * mult
    k4_min, k4_max = boundaries[3][0] * mult, boundaries[3][1] * mult
    k5_min, k5_max = boundaries[4][0] * mult, boundaries[4][1] * mult
    k6_min, k6_max = boundaries[5][0] * mult, boundaries[5][1] * mult
    p1_min, p1_max = boundaries[6][0] * mult, boundaries[6][1] * mult
    p2_min, p2_max = boundaries[7][0] * mult, boundaries[7][1] * mult
    s1_min, s1_max = boundaries[8][0] * mult, boundaries[8][1] * mult
    s2_min, s2_max = boundaries[9][0] * mult, boundaries[9][1] * mult
    s3_min, s3_max = boundaries[10][0] * mult, boundaries[10][1] * mult
    s4_min, s4_max = boundaries[11][0] * mult, boundaries[11][1] * mult
    k1 = torch.rand(batch, device=device, dtype=dtype) * (k1_max - k1_min) + k1_min
    k2 = torch.rand(batch, device=device, dtype=dtype) * (k2_max - k2_min) + k2_min
    k3 = torch.rand(batch, device=device, dtype=dtype) * (k3_max - k3_min) + k3_min
    k4 = torch.rand(batch, device=device, dtype=dtype) * (k4_max - k4_min) + k4_min
    k5 = torch.rand(batch, device=device, dtype=dtype) * (k5_max - k5_min) + k5_min
    k6 = torch.rand(batch, device=device, dtype=dtype) * (k6_max - k6_min) + k6_min
    p1 = torch.rand(batch, device=device, dtype=dtype) * (p1_max - p1_min) + p1_min
    p2 = torch.rand(batch, device=device, dtype=dtype) * (p2_max - p2_min) + p2_min
    s1 = torch.rand(batch, device=device, dtype=dtype) * (s1_max - s1_min) + s1_min
    s2 = torch.rand(batch, device=device, dtype=dtype) * (s2_max - s2_min) + s2_min
    s3 = torch.rand(batch, device=device, dtype=dtype) * (s3_max - s3_min) + s3_min
    s4 = torch.rand(batch, device=device, dtype=dtype) * (s4_max - s4_min) + s4_min
    return k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4


def mei(boundaries, mult, batch, device, dtype):
    k1_min, k1_max = boundaries[0][0] * mult, boundaries[0][1] * mult
    k2_min, k2_max = boundaries[1][0] * mult, boundaries[1][1] * mult
    p1_min, p1_max = boundaries[2][0] * mult, boundaries[2][1] * mult
    p2_min, p2_max = boundaries[3][0] * mult, boundaries[3][1] * mult
    xi_min, xi_max = boundaries[4][0] * mult, boundaries[4][1] * mult
    k1 = torch.rand(batch, device=device, dtype=dtype) * (k1_max - k1_min) + k1_min
    k2 = torch.rand(batch, device=device, dtype=dtype) * (k2_max - k2_min) + k2_min
    p1 = torch.rand(batch, device=device, dtype=dtype) * (p1_max - p1_min) + p1_min
    p2 = torch.rand(batch, device=device, dtype=dtype) * (p2_max - p2_min) + p2_min
    xi = torch.rand(batch, device=device, dtype=dtype) * (xi_max - xi_min) + xi_min
    return k1, k2, p1, p2, xi


def consistent_fisheye(boundaries, mult, batch, device, dtype):
    sign = random.choice([-1, 1])
    return free_fisheye(boundaries, sign * mult, batch, device, dtype)


def invert_fisheye(boundaries, mult, batch, device, dtype):
    k1_min, k1_max = boundaries[0][0] * mult, boundaries[0][1] * mult
    k2_min, k2_max = boundaries[1][0] * mult, boundaries[1][1] * mult
    k3_min, k3_max = boundaries[2][0] * mult, boundaries[2][1] * mult
    k4_min, k4_max = boundaries[3][0] * mult, boundaries[3][1] * mult
    k5_min, k5_max = boundaries[4][0] * mult, boundaries[4][1] * mult
    k6_min, k6_max = boundaries[5][0] * mult, boundaries[5][1] * mult
    p1_min, p1_max = boundaries[6][0] * mult, boundaries[6][1] * mult
    p2_min, p2_max = boundaries[7][0] * mult, boundaries[7][1] * mult
    s1_min, s1_max = boundaries[8][0] * mult, boundaries[8][1] * mult
    s2_min, s2_max = boundaries[9][0] * mult, boundaries[9][1] * mult
    s3_min, s3_max = boundaries[10][0] * mult, boundaries[10][1] * mult
    s4_min, s4_max = boundaries[11][0] * mult, boundaries[11][1] * mult

    sign = random.choice([-1, 1])
    k1 = torch.rand(batch, device=device, dtype=dtype) * (k1_max - k1_min) + k1_min
    k1 = sign * k1
    k2 = torch.rand(batch, device=device, dtype=dtype) * (k2_max - k2_min) + k2_min
    k2 = -1 * sign * k2
    k3 = torch.rand(batch, device=device, dtype=dtype) * (k3_max - k3_min) + k3_min
    k3 = sign * k3
    k4 = torch.rand(batch, device=device, dtype=dtype) * (k4_max - k4_min) + k4_min
    k4 = -1 * sign * k4
    k5 = torch.rand(batch, device=device, dtype=dtype) * (k5_max - k5_min) + k5_min
    k5 = sign * k5
    k6 = torch.rand(batch, device=device, dtype=dtype) * (k6_max - k6_min) + k6_min
    k6 = -1 * sign * k6

    p1 = torch.rand(batch, device=device, dtype=dtype) * (p1_max - p1_min) + p1_min
    p2 = torch.rand(batch, device=device, dtype=dtype) * (p2_max - p2_min) + p2_min
    s1 = torch.rand(batch, device=device, dtype=dtype) * (s1_max - s1_min) + s1_min
    s2 = torch.rand(batch, device=device, dtype=dtype) * (s2_max - s2_min) + s2_min
    s3 = torch.rand(batch, device=device, dtype=dtype) * (s3_max - s3_min) + s3_min
    s4 = torch.rand(batch, device=device, dtype=dtype) * (s4_max - s4_min) + s4_min
    return k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4


class CameraSampler(nn.Module):
    def __init__(self):
        super().__init__()
        with open("camera_sampler.json", "r") as f:
            config = json.load(f)
        self.camera_type = config["type"]
        self.sampling_fn = config["fn"]
        self.boundaries = nn.ParameterList(
            [nn.Parameter(torch.tensor(x), requires_grad=False) for x in config["boundaries"]]
        )
        self.probs = nn.Parameter(torch.tensor(config["probs"]), requires_grad=False)

    def forward(self, fx, fy, cx, cy, mult, ratio, H):
        selected_idx = torch.multinomial(self.probs, num_samples=1)
        device, dtype = fx.device, fx.dtype

        selected_camera = self.camera_type[selected_idx]
        selected_sampling_fn = self.sampling_fn[selected_idx]
        selected_boundaries = self.boundaries[selected_idx]
        if "Fisheye" in selected_camera or "OPENCV" in selected_camera:
            mult = mult * ratio

        params = eval(selected_sampling_fn)(selected_boundaries, mult, len(fx), device, dtype)
        params = torch.stack([fx, fy, cx, cy, *params], dim=1)
        camera = eval(selected_camera)(params=params)
        return camera
