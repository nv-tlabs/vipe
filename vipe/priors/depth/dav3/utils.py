# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import einsum
from PIL import Image
from tqdm import tqdm


class Color:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    RESET = "\033[0m"


LOG_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}

COLOR_MAP = {"ERROR": Color.RED, "WARN": Color.YELLOW, "INFO": Color.WHITE, "DEBUG": Color.GREEN}


def get_env_log_level():
    level = os.environ.get("DA3_LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level, LOG_LEVELS["INFO"])


class Logger:
    def __init__(self):
        self.level = get_env_log_level()

    def log(self, level_str, *args, **kwargs):
        level_key = level_str.split(":")[0].strip()
        level_val = LOG_LEVELS.get(level_key)
        if level_val is None:
            raise ValueError(f"Unknown log level: {level_str}")
        if self.level >= level_val:
            color = COLOR_MAP[level_key]
            msg = " ".join(str(arg) for arg in args)

            # Align log level output in square brackets
            # ERROR and DEBUG are 5 characters, INFO and WARN have an extra space for alignment
            tag = level_key
            if tag in ("INFO", "WARN"):
                tag += " "
            print(
                f"{color}[{tag}] {msg}{Color.RESET}",
                file=sys.stderr if level_key == "ERROR" else sys.stdout,
                **kwargs,
            )

    def error(self, *args, **kwargs):
        self.log("ERROR:", *args, **kwargs)

    def warn(self, *args, **kwargs):
        self.log("WARN:", *args, **kwargs)

    def info(self, *args, **kwargs):
        self.log("INFO:", *args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log("DEBUG:", *args, **kwargs)


logger = Logger()

__all__ = ["logger"]

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")


def as_homogeneous(ext):
    """
    Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix.
    Supports torch.Tensor or np.ndarray.
    """
    if isinstance(ext, torch.Tensor):
        # If already in homogeneous form
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            # Create a new homogeneous matrix
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def transpose_last_two_axes(arr):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,  # float coordinates (xy indexing), "*shape dim"
    torch.Tensor,  # integer indices (ij indexing), "*shape dim"
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(points: torch.Tensor) -> torch.Tensor:  # "*batch dim"  # "*batch dim+1"
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:  #  "*batch dim"  # "*batch dim+1"
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    transformation: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(
        transformation,
        homogeneous_coordinates.to(transformation.dtype),
        "... i j, ... j -> ... i",
    )


def transform_cam2world(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates: torch.Tensor,  # "*#batch dim"
    z: torch.Tensor,  # "*#batch"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> torch.Tensor:  # "*batch dim+1"
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.float().inverse().to(intrinsics),
        coordinates.to(intrinsics.dtype),
        "... i j, ... j -> ... i",
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim+2 dim+2"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> tuple[
    torch.Tensor,  # origins, "*batch dim+1"
    torch.Tensor,  # directions, "*batch dim+1"
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:  # "batch 3 3" -> "batch 2"
    intrinsics_inv = intrinsics.float().inverse().to(intrinsics)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=intrinsics.dtype, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def map_pdf_to_opacity(
    pdf: torch.Tensor,  # " *batch"
    global_step: int = 0,
    opacity_mapping: Optional[dict] = None,
) -> torch.Tensor:  # " *batch"
    # https://www.desmos.com/calculator/opvwti3ba9

    # Figure out the exponent.
    if opacity_mapping is not None:
        cfg = SimpleNamespace(**opacity_mapping)
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    else:
        x = 0.0
    exponent = 2**x

    # Map the probability density to an opacity.
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


def least_squares_scale_scalar(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute least squares scale factor s such that a ≈ s * b.

    Args:
        a: First tensor
        b: Second tensor
        eps: Small epsilon for numerical stability

    Returns:
        Scalar tensor containing the scale factor

    Raises:
        ValueError: If tensors have mismatched shapes or devices
        TypeError: If tensors are not floating point
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"Device mismatch: {a.device} vs {b.device}")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must be floating point type")

    # Compute dot products for least squares solution
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.dot(b.reshape(-1), b.reshape(-1)).clamp_min(eps)
    return num / den


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Compute non-sky mask from sky prediction.

    Args:
        sky_prediction: Sky prediction tensor
        threshold: Threshold for sky classification

    Returns:
        Boolean mask where True indicates non-sky regions
    """
    return sky_prediction < threshold


def compute_alignment_mask(
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    depth: torch.Tensor,
    metric_depth: torch.Tensor,
    median_conf: torch.Tensor,
    min_depth_threshold: float = 1e-3,
    min_metric_depth_threshold: float = 1e-2,
) -> torch.Tensor:
    """
    Compute mask for depth alignment based on confidence and depth thresholds.

    Args:
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        depth: Predicted depth tensor
        metric_depth: Metric depth tensor
        median_conf: Median confidence threshold
        min_depth_threshold: Minimum depth threshold
        min_metric_depth_threshold: Minimum metric depth threshold

    Returns:
        Boolean mask for valid alignment regions
    """
    return (
        (depth_conf >= median_conf)
        & non_sky_mask
        & (metric_depth > min_metric_depth_threshold)
        & (depth > min_depth_threshold)
    )


def sample_tensor_for_quantile(tensor: torch.Tensor, max_samples: int = 100000) -> torch.Tensor:
    """
    Sample tensor elements for quantile computation to reduce memory usage.

    Args:
        tensor: Input tensor to sample
        max_samples: Maximum number of samples to take

    Returns:
        Sampled tensor
    """
    if tensor.numel() <= max_samples:
        return tensor

    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor.flatten()[idx]


def apply_metric_scaling(depth: torch.Tensor, intrinsics: torch.Tensor, scale_factor: float = 300.0) -> torch.Tensor:
    """
    Apply metric scaling to depth based on camera intrinsics.

    Args:
        depth: Input depth tensor
        intrinsics: Camera intrinsics tensor
        scale_factor: Scaling factor for metric conversion

    Returns:
        Scaled depth tensor
    """
    focal_length = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2
    return depth * (focal_length[:, :, None, None] / scale_factor)


def set_sky_regions_to_max_depth(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    max_depth: float = 200.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Set sky regions to maximum depth and high confidence.

    Args:
        depth: Depth tensor
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.clone()
    depth_conf = depth_conf.clone()

    # Set sky regions to max depth and high confidence
    depth[~non_sky_mask] = max_depth
    depth_conf[~non_sky_mask] = 1.0

    return depth, depth_conf


def batch_apply_alignment_to_enc(
    rots: torch.Tensor, trans: torch.Tensor, scales: torch.Tensor, enc_list: List[torch.Tensor]
):
    pass


def batch_apply_alignment_to_ext(rots: torch.Tensor, trans: torch.Tensor, scales: torch.Tensor, ext: torch.Tensor):
    device, _ = ext.device, ext.dtype
    if ext.shape[-2:] == (3, 4):
        pad = torch.zeros((*ext.shape[:-2], 4, 4), dtype=ext.dtype, device=device)
        pad[..., :3, :4] = ext
        pad[..., 3, 3] = 1.0
        ext = pad
    pose_est = affine_inverse(ext)
    pose_new_align_rot = rots[:, None] @ pose_est[..., :3, :3]
    pose_new_align_trans = scales[:, None, None] * (rots[:, None] @ pose_est[..., :3, 3:])[..., 0] + trans[:, None]
    pose_new_align = torch.zeros_like(ext)
    pose_new_align[..., :3, :3] = pose_new_align_rot
    pose_new_align[..., :3, 3] = pose_new_align_trans
    pose_new_align[..., 3, 3] = 1.0
    return affine_inverse(pose_new_align)[:, :3]


def batch_align_poses_umeyama(ext_ref: torch.Tensor, ext_est: torch.Tensor):
    device, dtype = ext_ref.device, ext_ref.dtype
    assert ext_ref.dtype in [torch.float32, torch.float64]
    assert ext_est.dtype in [torch.float32, torch.float64]
    assert ext_ref.requires_grad is False
    assert ext_est.requires_grad is False
    rots, trans, scales = [], [], []
    for b in range(ext_ref.shape[0]):
        r, t, s = align_poses_umeyama(ext_ref[b].cpu().numpy(), ext_est[b].cpu().numpy())
        rots.append(torch.from_numpy(r).to(device=device, dtype=dtype))
        trans.append(torch.from_numpy(t).to(device=device, dtype=dtype))
        scales.append(torch.tensor(s, device=device, dtype=dtype))
    return torch.stack(rots), torch.stack(trans), torch.stack(scales)


def _to44(ext):
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
        return out
    return ext


def _poses_from_ext(ext_ref, ext_est):
    ext_ref = _to44(ext_ref)
    ext_est = _to44(ext_est)
    pose_ref = affine_inverse_np(ext_ref)
    pose_est = affine_inverse_np(ext_est)
    return pose_ref, pose_est


def _umeyama_sim3_from_paths(pose_ref, pose_est):
    ref_points = pose_ref[:, :3, 3]
    est_points = pose_est[:, :3, 3]
    r, t, s = _umeyama_points(est_points, ref_points)
    pose_est_aligned = _apply_sim3_to_poses(pose_est, r, t, s)
    return r, t, s, pose_est_aligned


def _umeyama_points(source: np.ndarray, target: np.ndarray):
    """Return R, t, s such that target ~= s * R @ source + t."""
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: {source.shape} vs {target.shape}")
    if source.shape[0] < 3:
        return np.eye(3), target.mean(axis=0) - source.mean(axis=0), 1.0

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / source.shape[0]

    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0

    rotation = vt.T @ correction @ u.T
    variance = np.sum(source_centered * source_centered) / source.shape[0]
    scale = float(np.sum(singular_values * np.diag(correction)) / max(variance, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    return rotation, translation, scale


def _apply_sim3_to_poses(poses, r, t, s):
    out = poses.copy()
    Ri = poses[:, :3, :3]
    ti = poses[:, :3, 3]
    out[:, :3, :3] = r @ Ri
    out[:, :3, 3] = (r @ (s * ti.T)).T + t
    return out


def _median_nn_thresh(pose_ref, pose_est_aligned):
    P_ref = pose_ref[:, :3, 3]
    P_est = pose_est_aligned[:, :3, 3]
    dists = []
    for p in P_est:
        dd = np.linalg.norm(P_ref - p[None, :], axis=1)
        dists.append(dd.min())
    return float(np.median(dists)) if dists else 0.0


def _ransac_align_sim3(pose_ref, pose_est, sub_n=None, inlier_thresh=None, max_iters=10, random_state=None):
    rng = np.random.default_rng(random_state)
    N = pose_ref.shape[0]
    idx_all = np.arange(N)
    if sub_n is None:
        sub_n = max(3, (N + 1) // 2)
    else:
        sub_n = max(3, min(sub_n, N))

    # Pre-alignment + default threshold
    r0, t0, s0, pose_est0 = _umeyama_sim3_from_paths(pose_ref, pose_est)
    if inlier_thresh is None:
        inlier_thresh = _median_nn_thresh(pose_ref, pose_est0)

    P_ref_all = pose_ref[:, :3, 3]

    best_model = (r0, t0, s0)
    best_inliers = None
    best_score = (-1, np.inf)  # (num_inliers, mean_err)

    for _ in range(max_iters):
        sample = rng.choice(idx_all, size=sub_n, replace=False)
        try:
            r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[sample], pose_est[sample])
        except Exception:
            continue
        pose_h = _apply_sim3_to_poses(pose_est, r, t, s)
        P_h = pose_h[:, :3, 3]
        errs = np.linalg.norm(P_h - P_ref_all, axis=1)  # Match by same index
        inliers = errs <= inlier_thresh
        k = int(inliers.sum())
        mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
        if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
            best_score = (k, mean_err)
            best_model = (r, t, s)
            best_inliers = inliers

    # Fit again with best inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[best_inliers], pose_est[best_inliers])
    else:
        r, t, s = best_model
    return r, t, s


def align_poses_umeyama(
    ext_ref: np.ndarray,
    ext_est: np.ndarray,
    return_aligned=False,
    ransac=False,
    sub_n=None,
    inlier_thresh=None,
    ransac_max_iters=10,
    random_state=None,
):
    """
    Align estimated trajectory to reference using Umeyama Sim(3).
    Default no RANSAC; if ransac=True, use RANSAC (max iterations default 10).
    - sub_n defaults to half the number of frames (rounded up, at least 3)
    - inlier_thresh defaults to median of "distance from each estimated pose to
      nearest reference pose after pre-alignment"
    Returns rotation (3x3), translation (3,), scale; optionally returns aligned extrinsics (4x4).
    """
    pose_ref, pose_est = _poses_from_ext(ext_ref, ext_est)

    if not ransac:
        r, t, s, pose_est_aligned = _umeyama_sim3_from_paths(pose_ref, pose_est)
    else:
        r, t, s = _ransac_align_sim3(
            pose_ref,
            pose_est,
            sub_n=sub_n,
            inlier_thresh=inlier_thresh,
            max_iters=ransac_max_iters,
            random_state=random_state,
        )
        pose_est_aligned = _apply_sim3_to_poses(pose_est, r, t, s)

    if return_aligned:
        ext_est_aligned = affine_inverse_np(pose_est_aligned)
        return r, t, s, ext_est_aligned
    return r, t, s


def apply_umeyama_alignment_to_ext(
    rot: np.ndarray,  # (3,3)
    trans: np.ndarray,  # (3,) or (1,3)
    scale: float,
    ext_est: np.ndarray,  # (...,4,4) or (...,3,4)
) -> np.ndarray:
    """
    Apply Sim(3) (R, t, s) to a batch of world-to-camera extrinsics ext_est.
    Returns the aligned extrinsics, with the same shape as input.
    """

    # Allow 3x4 extrinsics: pad to 4x4
    if ext_est.shape[-2:] == (3, 4):
        pad = np.zeros((*ext_est.shape[:-2], 4, 4), dtype=ext_est.dtype)
        pad[..., :3, :4] = ext_est
        pad[..., 3, 3] = 1.0
        ext_est = pad

    # Convert world-to-camera to camera-to-world
    pose_est = affine_inverse_np(ext_est)  # (...,4,4)
    R_e = pose_est[..., :3, :3]  # (...,3,3)
    t_e = pose_est[..., :3, 3]  # (...,3)

    # Apply Sim(3) transformation
    R_a = np.einsum("ij,...jk->...ik", rot, R_e)  # (...,3,3)
    t_a = scale * np.einsum("ij,...j->...i", rot, t_e) + trans  # (...,3)

    # Assemble the transformed pose
    pose_a = np.zeros_like(pose_est)
    pose_a[..., :3, :3] = R_a
    pose_a[..., :3, 3] = t_a
    pose_a[..., 3, 3] = 1.0

    # Convert back to world-to-camera
    return affine_inverse_np(pose_a)


def transform_points_sim3(points, rot, trans, scale, inverse=False):
    """
    Sim(3) transform point cloud
    points: (N, 3)
    rot: (3, 3)
    trans: (3,) or (1, 3)
    scale: float
    inverse: Whether to do inverse transform (ref->est)
    Returns: (N, 3)
    """
    if not inverse:
        # Forward: est -> ref
        return scale * (points @ rot.T) + trans
    else:
        # Inverse: ref -> est
        return ((points - trans) @ rot) / scale


def _rand_rot():
    u1, u2, u3 = np.random.rand(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.math.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.math.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.math.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.math.pi * u3),
        ]
    )
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rand_pose():
    R, t = _rand_rot(), np.random.randn(3)
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


if __name__ == "__main__":
    np.random.seed(42)
    # 1. Randomly generate reference trajectory and Sim(3)
    N = 8
    pose_ref = np.stack([_rand_pose() for _ in range(N)])  # (N,4,4)  cam→world
    rot_gt = _rand_rot()
    scale_gt = 2.3
    trans_gt = np.random.randn(3)
    # 2. Generate estimated trajectory (apply Sim(3))
    pose_est = np.zeros_like(pose_ref)
    for i in range(N):
        R = pose_ref[i][:3, :3]
        t = pose_ref[i][:3, 3]
        pose_est[i][:3, :3] = rot_gt @ R
        pose_est[i][:3, 3] = scale_gt * (rot_gt @ t) + trans_gt
        pose_est[i][3, 3] = 1.0
    # 3. Get extrinsics (world->cam)
    ext_ref = affine_inverse_np(pose_ref)
    ext_est = affine_inverse_np(pose_est)
    # 4. Use umeyama alignment, estimate Sim(3)
    r_est, t_est, s_est = align_poses_umeyama(ext_ref, ext_est)
    print("GT scale:", scale_gt, "Estimated:", s_est)
    print("GT trans:", trans_gt, "Estimated:", t_est)
    print("GT rot:\n", rot_gt, "\nEstimated:\n", r_est)
    # 5. Random point cloud, in ref frame
    num_points = 100
    points_ref = np.random.randn(num_points, 3)
    # 6. Use GT Sim(3) inverse transform to est frame
    points_est = transform_points_sim3(points_ref, rot_gt, trans_gt, scale_gt, inverse=True)
    # 7. Use estimated Sim(3) forward transform back to ref frame
    points_ref_recovered = transform_points_sim3(points_est, r_est, t_est, s_est, inverse=False)
    # 8. Check error
    err = np.abs(points_ref_recovered - points_ref)
    print("Point cloud sim3 transform error (mean abs):", err.mean())
    print("Point cloud sim3 transform error (max abs):", err.max())
    assert err.mean() < 1e-6, "Mean sim3 transform error too large!"
    assert err.max() < 1e-5, "Max sim3 transform error too large!"
    print("Sim(3) point cloud transform & alignment test passed!")


def parallel_execution(
    *args,
    action: Callable,
    num_processes: int = 32,
    print_progress: bool = False,
    sequential: bool = False,
    desc: str | None = None,
    **kwargs,
):
    args = list(args)

    def get_length() -> int:
        for arg in args:
            if isinstance(arg, list):
                return len(arg)
        for value in kwargs.values():
            if isinstance(value, list):
                return len(value)
        raise NotImplementedError("No distributed list argument found.")

    def get_action_args(length: int, index: int):
        action_args = [(arg[index] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        action_kwargs = {
            key: (value[index] if isinstance(value, list) and len(value) == length else value)
            for key, value in kwargs.items()
        }
        return action_args, action_kwargs

    length = get_length()
    if sequential:
        results = []
        for index in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, index)
            results.append(action(*action_args, **action_kwargs))
        return results

    pool = ThreadPool(processes=num_processes)
    async_results = []
    for index in range(length):
        action_args, action_kwargs = get_action_args(length, index)
        async_results.append(pool.apply_async(action, action_args, action_kwargs))

    results = [async_result.get() for async_result in tqdm(async_results, desc=desc, disable=not print_progress)]
    pool.close()
    pool.join()
    return results


class InputProcessor:
    """Prepares a batch of images for model inference.
    This processor converts a list of image file paths into a single, model-ready
    tensor. The processing pipeline is executed in parallel across multiple workers
    for efficiency.

    Pipeline:
      1) Load image and convert to RGB
      2) Boundary resize (upper/lower bound, preserving aspect ratio)
      3) Enforce divisibility by PATCH_SIZE:
         - "*resize" methods: each dimension is rounded to nearest multiple
           (may up/downscale a few px)
         - "*crop"   methods: each dimension is floored to nearest multiple via center crop
      4) Convert to tensor and apply ImageNet normalization
      5) Stack into (1, N, 3, H, W)

    Parallelization:
      - Each image is processed independently in a worker.
      - Order of outputs matches the input order.
    """

    NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    PATCH_SIZE = 14

    def __init__(self):
        pass

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        num_workers: int = 8,
        print_progress: bool = False,
        sequential: bool | None = None,
        desc: str | None = "Preprocess",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            (tensor, extrinsics_list, intrinsics_list)
            tensor shape: (1, N, 3, H, W)
        """
        sequential = self._resolve_sequential(sequential, num_workers)
        exts_list, ixts_list = self._validate_and_pack_meta(image, extrinsics, intrinsics)

        results = self._run_parallel(
            image=image,
            exts_list=exts_list,
            ixts_list=ixts_list,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
        )

        proc_imgs, out_sizes, out_ixts, out_exts = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(proc_imgs, out_sizes, out_ixts)

        batch_tensor = self._stack_batch(proc_imgs)
        out_exts = (
            torch.from_numpy(np.asarray(out_exts)).float() if out_exts is not None and out_exts[0] is not None else None
        )
        out_ixts = (
            torch.from_numpy(np.asarray(out_ixts)).float() if out_ixts is not None and out_ixts[0] is not None else None
        )
        return (batch_tensor, out_exts, out_ixts)

    # -----------------------------
    # __call__ helpers
    # -----------------------------
    def _resolve_sequential(self, sequential: bool | None, num_workers: int) -> bool:
        return (num_workers <= 1) if sequential is None else sequential

    def _validate_and_pack_meta(
        self,
        images: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[list[np.ndarray | None] | None, list[np.ndarray | None] | None]:
        if extrinsics is not None and len(extrinsics) != len(images):
            raise ValueError("Length of extrinsics must match images when provided.")
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError("Length of intrinsics must match images when provided.")
        exts_list = [e for e in extrinsics] if extrinsics is not None else None
        ixts_list = [k for k in intrinsics] if intrinsics is not None else None
        return exts_list, ixts_list

    def _run_parallel(
        self,
        *,
        image: list[np.ndarray | Image.Image | str],
        exts_list: list[np.ndarray | None] | None,
        ixts_list: list[np.ndarray | None] | None,
        process_res: int,
        process_res_method: str,
        num_workers: int,
        print_progress: bool,
        sequential: bool,
        desc: str | None,
    ):
        results = parallel_execution(
            image,
            exts_list,
            ixts_list,
            action=self._process_one,  # (img, extrinsic, intrinsic, ...)
            num_processes=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if not results:
            raise RuntimeError("No preprocessing results returned. Check inputs and parallel_execution.")
        return results

    def _unpack_results(self, results):
        """
        results: List[Tuple[torch.Tensor, Tuple[H, W], Optional[np.ndarray], Optional[np.ndarray]]]
        -> processed_images, out_sizes, out_intrinsics, out_extrinsics
        """
        try:
            processed_images, out_sizes, out_intrinsics, out_extrinsics = zip(*results)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected results structure from parallel_execution: {type(results)} / sample: {results[0]}"
            ) from e

        return list(processed_images), list(out_sizes), list(out_intrinsics), list(out_extrinsics)

    def _unify_batch_shapes(
        self,
        processed_images: list[torch.Tensor],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all tensors to the smallest H, W; adjust intrinsics' cx, cy accordingly."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)
        logger.warn(
            f"Images in batch have different sizes {out_sizes}; center-cropping all to smallest ({min_h},{min_w})"
        )

        center_crop = T.CenterCrop((min_h, min_w))
        new_imgs, new_sizes, new_ixts = [], [], []
        for img_t, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
            crop_top = max(0, (H - min_h) // 2)
            crop_left = max(0, (W - min_w) // 2)
            new_imgs.append(center_crop(img_t))
            new_sizes.append((min_h, min_w))
            if K is None:
                new_ixts.append(None)
            else:
                K_adj = K.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)
        return new_imgs, new_sizes, new_ixts

    def _stack_batch(self, processed_images: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(processed_images)

    # -----------------------------
    # Per-item worker
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[torch.Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        # Load & remember original size
        pil_img = self._load_image(img)
        orig_w, orig_h = pil_img.size

        # Boundary resize
        pil_img = self._resize_image(pil_img, process_res, process_res_method)
        w, h = pil_img.size
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            pil_img = self._make_divisible_by_resize(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        elif process_res_method.endswith("crop"):
            pil_img = self._make_divisible_by_crop(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._crop_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Convert to tensor & normalize
        img_tensor = self._normalize_image(pil_img)
        _, H, W = img_tensor.shape
        assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

        # Return: (img_tensor, (H, W), intrinsic, extrinsic)
        return img_tensor, (H, W), intrinsic, extrinsic

    # -----------------------------
    # Intrinsics transforms
    # -----------------------------
    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def _crop_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        crop_h = (orig_h - h) // 2
        crop_w = (orig_w - w) // 2
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        return K

    # -----------------------------
    # I/O & normalization
    # -----------------------------
    def _load_image(self, img: np.ndarray | Image.Image | str) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            # Assume HxWxC uint8/RGB
            return Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _normalize_image(self, img: Image.Image) -> torch.Tensor:
        img_tensor = T.ToTensor()(img)
        return self.NORMALIZE(img_tensor)

    # -----------------------------
    # Boundary resizing
    # -----------------------------
    def _resize_image(self, img: Image.Image, target_size: int, method: str) -> Image.Image:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_shortest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    # -----------------------------
    # Make divisible by PATCH_SIZE
    # -----------------------------
    def _make_divisible_by_crop(self, img: Image.Image, patch: int) -> Image.Image:
        """
        Floor each dimension to the nearest multiple of PATCH_SIZE via center crop.
        Example: 504x377 -> 504x364
        """
        w, h = img.size
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def _make_divisible_by_resize(self, img: Image.Image, patch: int) -> Image.Image:
        """
        Round each dimension to nearest multiple of PATCH_SIZE via small resize.
        """
        w, h = img.size

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        upscale = (new_w > w) or (new_h > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)


# Backward compatibility alias
InputAdapter = InputProcessor
