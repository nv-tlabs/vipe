# This file includes code originally from the Geocalib repository:
# https://github.com/cvg/GeoCalib
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

"""Implementation of perspective fields.

Adapted from https://github.com/jinlinyi/PerspectiveFields/blob/main/perspective2d/utils/panocam.py
"""

from typing import Tuple

import torch
from torch.nn import functional as F

from .camera import BaseCamera
from .gravity import Gravity
from .misc import J_up_projection, J_vecnorm, SphericalManifold

# flake8: noqa: E266


def get_horizon_line(camera: BaseCamera, gravity: Gravity, relative: bool = True) -> torch.Tensor:
    """Get the horizon line from the camera parameters.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.
        relative (bool, optional): Whether to normalize horizon line by img_h. Defaults to True.

    Returns:
        torch.Tensor: In image frame, fraction of image left/right border intersection with
        respect to image height.
    """
    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    # project horizon midpoint to image plane
    horizon_midpoint = camera.new_tensor([0, 0, 1])
    horizon_midpoint = camera.K @ gravity.R @ horizon_midpoint
    midpoint = horizon_midpoint[:2] / horizon_midpoint[2]

    # compute left and right offset to borders
    left_offset = midpoint[0] * torch.tan(gravity.roll)
    right_offset = (camera.size[0] - midpoint[0]) * torch.tan(gravity.roll)
    left, right = midpoint[1] + left_offset, midpoint[1] - right_offset

    horizon = camera.new_tensor([left, right])
    return horizon / camera.size[1] if relative else horizon


def get_up_field(camera: BaseCamera, gravity: Gravity, normalize: bool = True) -> torch.Tensor:
    """Get the up vector field from the camera parameters.

    Args:
        camera (Camera): Camera parameters.
        normalize (bool, optional): Whether to normalize the up vector. Defaults to True.

    Returns:
        torch.Tensor: up vector field as tensor of shape (..., h, w, 2).
    """
    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    uv = camera.normalize(camera.pixel_coordinates())

    # projected up is (a, b) - c * (u, v)
    abc = gravity.vec3d
    projected_up2d = abc[..., None, :2] - abc[..., 2, None, None] * uv  # (..., N, 2)

    if hasattr(camera, "dist"):
        d_uv = camera.distort(uv, return_scale=True)[0]  # (..., N, 1)
        d_uv = torch.diag_embed(d_uv.expand(d_uv.shape[:-1] + (2,)))  # (..., N, 2, 2)
        offset = camera.up_projection_offset(uv)  # (..., N, 2)
        offset = torch.einsum("...i,...j->...ij", offset, uv)  # (..., N, 2, 2)

        # (..., N, 2)
        projected_up2d = torch.einsum("...Nij,...Nj->...Ni", d_uv + offset, projected_up2d)

    if normalize:
        projected_up2d = F.normalize(projected_up2d, dim=-1)  # (..., N, 2)

    return projected_up2d.reshape(camera.shape[0], h, w, 2)


def J_up_field(
    camera: BaseCamera,
    gravity: Gravity,
    spherical: bool = False,
    log_focal: bool = False,
) -> torch.Tensor:
    """Get the jacobian of the up field.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.
        spherical (bool, optional): Whether to use spherical coordinates. Defaults to False.
        log_focal (bool, optional): Whether to use log-focal length. Defaults to False.

    Returns:
        torch.Tensor: Jacobian of the up field as a tensor of shape (..., h, w, 2, 2, 3).
    """
    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    # Forward
    xy = camera.pixel_coordinates()
    uv = camera.normalize(xy)

    projected_up2d = gravity.vec3d[..., None, :2] - gravity.vec3d[..., 2, None, None] * uv

    # Backward
    J = []

    # (..., N, 2, 2)
    J_norm2proj = J_vecnorm(get_up_field(camera, gravity, normalize=False).reshape(camera.shape[0], -1, 2))

    # distortion values
    if hasattr(camera, "dist"):
        d_uv = camera.distort(uv, return_scale=True)[0]  # (..., N, 1)
        d_uv = torch.diag_embed(d_uv.expand(d_uv.shape[:-1] + (2,)))  # (..., N, 2, 2)
        offset = camera.up_projection_offset(uv)  # (..., N, 2)
        offset_uv = torch.einsum("...i,...j->...ij", offset, uv)  # (..., N, 2, 2)

    ######################
    ## Gravity Jacobian ##
    ######################

    J_proj2abc = J_up_projection(uv, gravity.vec3d, wrt="abc")  # (..., N, 2, 3)

    if hasattr(camera, "dist"):
        # (..., N, 2, 3)
        J_proj2abc = torch.einsum("...Nij,...Njk->...Nik", d_uv + offset_uv, J_proj2abc)

    J_abc2delta = SphericalManifold.J_plus(gravity.vec3d) if spherical else gravity.J_rp()
    J_proj2delta = torch.einsum("...Nij,...jk->...Nik", J_proj2abc, J_abc2delta)
    J_up2delta = torch.einsum("...Nij,...Njk->...Nik", J_norm2proj, J_proj2delta)
    J.append(J_up2delta)

    ######################
    ### Focal Jacobian ###
    ######################

    J_proj2uv = J_up_projection(uv, gravity.vec3d, wrt="uv")  # (..., N, 2, 2)

    if hasattr(camera, "dist"):
        J_proj2up = torch.einsum("...Nij,...Njk->...Nik", d_uv + offset_uv, J_proj2uv)
        J_proj2duv = torch.einsum("...i,...j->...ji", offset, projected_up2d)

        inner = (uv * projected_up2d).sum(-1)[..., None, None]
        J_proj2offset1 = inner * camera.J_up_projection_offset(uv, wrt="uv")
        J_proj2offset2 = torch.einsum("...i,...j->...ij", offset, projected_up2d)  # (..., N, 2, 2)
        J_proj2uv = (J_proj2duv + J_proj2offset1 + J_proj2offset2) + J_proj2up

    J_uv2f = camera.J_normalize(xy)  # (..., N, 2, 2)

    if log_focal:
        J_uv2f = J_uv2f * camera.f[..., None, None, :]  # (..., N, 2, 2)

    J_uv2f = J_uv2f.sum(-1)  # (..., N, 2)

    J_proj2f = torch.einsum("...ij,...j->...i", J_proj2uv, J_uv2f)  # (..., N, 2)
    J_up2f = torch.einsum("...Nij,...Nj->...Ni", J_norm2proj, J_proj2f)[..., None]  # (..., N, 2, 1)
    J.append(J_up2f)

    ######################
    ##### K1 Jacobian ####
    ######################

    if hasattr(camera, "dist"):
        J_duv = camera.J_distort(uv, wrt="scale2dist")
        J_duv = torch.diag_embed(J_duv.expand(J_duv.shape[:-1] + (2,)))  # (..., N, 2, 2)
        J_offset = torch.einsum("...i,...j->...ij", camera.J_up_projection_offset(uv, wrt="dist"), uv)
        J_proj2k1 = torch.einsum("...Nij,...Nj->...Ni", J_duv + J_offset, projected_up2d)
        J_k1 = torch.einsum("...Nij,...Nj->...Ni", J_norm2proj, J_proj2k1)[..., None]
        J.append(J_k1)

    n_params = sum(j.shape[-1] for j in J)
    return torch.cat(J, axis=-1).reshape(camera.shape[0], h, w, 2, n_params)


def get_latitude_field(camera: BaseCamera, gravity: Gravity) -> torch.Tensor:
    """Get the latitudes of the camera pixels in radians.

    Latitudes are defined as the angle between the ray and the up vector.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.

    Returns:
        torch.Tensor: Latitudes in radians as a tensor of shape (..., h, w, 1).
    """
    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    uv1, _ = camera.image2world(camera.pixel_coordinates())
    rays = camera.pixel_bearing_many(uv1)

    lat = torch.einsum("...Nj,...j->...N", rays, gravity.vec3d)

    eps = 1e-6
    lat_asin = torch.asin(lat.clamp(min=-1 + eps, max=1 - eps))

    return lat_asin.reshape(camera.shape[0], h, w, 1)


def J_latitude_field(
    camera: BaseCamera,
    gravity: Gravity,
    spherical: bool = False,
    log_focal: bool = False,
) -> torch.Tensor:
    """Get the jacobian of the latitude field.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.
        spherical (bool, optional): Whether to use spherical coordinates. Defaults to False.
        log_focal (bool, optional): Whether to use log-focal length. Defaults to False.

    Returns:
        torch.Tensor: Jacobian of the latitude field as a tensor of shape (..., h, w, 1, 3).
    """
    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    # Forward
    xy = camera.pixel_coordinates()
    uv1, _ = camera.image2world(xy)
    uv1_norm = camera.pixel_bearing_many(uv1)  # (..., N, 3)

    # Backward
    J = []
    J_norm2w_to_img = J_vecnorm(uv1)[..., :2]  # (..., N, 2)

    ######################
    ## Gravity Jacobian ##
    ######################

    J_delta = SphericalManifold.J_plus(gravity.vec3d) if spherical else gravity.J_rp()
    J_delta = torch.einsum("...Ni,...ij->...Nj", uv1_norm, J_delta)  # (..., N, 2)
    J.append(J_delta)

    ######################
    ### Focal Jacobian ###
    ######################

    J_w_to_img2f = camera.J_image2world(xy, "f")  # (..., N, 2, 2)
    if log_focal:
        J_w_to_img2f = J_w_to_img2f * camera.f[..., None, None, :]
    J_w_to_img2f = J_w_to_img2f.sum(-1)  # (..., N, 2)

    J_norm2f = torch.einsum("...Nij,...Nj->...Ni", J_norm2w_to_img, J_w_to_img2f)  # (..., N, 3)
    J_f = torch.einsum("...Ni,...i->...N", J_norm2f, gravity.vec3d).unsqueeze(-1)  # (..., N, 1)
    J.append(J_f)

    ######################
    ##### K1 Jacobian ####
    ######################

    if hasattr(camera, "dist"):
        J_w_to_img2k1 = camera.J_image2world(xy, "dist")  # (..., N, 2)
        # (..., N, 2)
        J_norm2k1 = torch.einsum("...Nij,...Nj->...Ni", J_norm2w_to_img, J_w_to_img2k1)
        # (..., N, 1)
        J_k1 = torch.einsum("...Ni,...i->...N", J_norm2k1, gravity.vec3d).unsqueeze(-1)
        J.append(J_k1)

    n_params = sum(j.shape[-1] for j in J)
    return torch.cat(J, axis=-1).reshape(camera.shape[0], h, w, 1, n_params)


def get_perspective_field(
    camera: BaseCamera,
    gravity: Gravity,
    use_up: bool = True,
    use_latitude: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the perspective field from the camera parameters.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.
        use_up (bool, optional): Whether to include the up vector field. Defaults to True.
        use_latitude (bool, optional): Whether to include the latitude field. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Up and latitude fields as tensors of shape
        (..., 2, h, w) and (..., 1, h, w).
    """
    assert use_up or use_latitude, "At least one of use_up or use_latitude must be True."

    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    if use_up:
        permute = (0, 3, 1, 2)
        # (..., 2, h, w)
        up = get_up_field(camera, gravity).permute(permute)
    else:
        shape = (camera.shape[0], 2, h, w)
        up = camera.new_zeros(shape)

    if use_latitude:
        permute = (0, 3, 1, 2)
        # (..., 1, h, w)
        lat = get_latitude_field(camera, gravity).permute(permute)
    else:
        shape = (camera.shape[0], 1, h, w)
        lat = camera.new_zeros(shape)

    return up, lat


def J_perspective_field(
    camera: BaseCamera,
    gravity: Gravity,
    use_up: bool = True,
    use_latitude: bool = True,
    spherical: bool = False,
    log_focal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the jacobian of the perspective field.

    Args:
        camera (Camera): Camera parameters.
        gravity (Gravity): Gravity vector.
        use_up (bool, optional): Whether to include the up vector field. Defaults to True.
        use_latitude (bool, optional): Whether to include the latitude field. Defaults to True.
        spherical (bool, optional): Whether to use spherical coordinates. Defaults to False.
        log_focal (bool, optional): Whether to use log-focal length. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Up and latitude jacobians as tensors of shape
        (..., h, w, 2, 4) and (..., h, w, 1, 4).
    """
    assert use_up or use_latitude, "At least one of use_up or use_latitude must be True."

    camera = camera.unsqueeze(0) if len(camera.shape) == 0 else camera
    gravity = gravity.unsqueeze(0) if len(gravity.shape) == 0 else gravity

    w, h = camera.size[0].unbind(-1)
    h, w = h.round().to(int), w.round().to(int)

    if use_up:
        J_up = J_up_field(camera, gravity, spherical, log_focal)  # (..., h, w, 2, 4)
    else:
        shape = (camera.shape[0], h, w, 2, 4)
        J_up = camera.new_zeros(shape)

    if use_latitude:
        J_lat = J_latitude_field(camera, gravity, spherical, log_focal)  # (..., h, w, 1, 4)
    else:
        shape = (camera.shape[0], h, w, 1, 4)
        J_lat = camera.new_zeros(shape)

    return J_up, J_lat
