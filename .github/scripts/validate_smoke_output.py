from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from vipe.utils import io


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise AssertionError(f"Missing expected artifact: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Artifact is empty: {path}")


def _require_frame_indices(name: str, indices: np.ndarray, expected_frames: int) -> None:
    expected = np.arange(expected_frames)
    if indices.shape != expected.shape or not np.array_equal(indices, expected):
        raise AssertionError(f"{name} frame indices are not exactly 0..{expected_frames - 1}: {indices}")


def _require_tensor_count(name: str, items: list[tuple[int, torch.Tensor]], expected_frames: int) -> None:
    indices = np.array([idx for idx, _ in items], dtype=np.int64)
    _require_frame_indices(name, indices, expected_frames)


def validate_smoke_output(base_path: Path, artifact_name: str, expected_frames: int) -> None:
    artifact = io.ArtifactPath(base_path, artifact_name)

    for path in [
        artifact.rgb_path,
        artifact.pose_path,
        artifact.depth_path,
        artifact.intrinsics_path,
        artifact.camera_type_path,
        artifact.mask_path,
        artifact.mask_phrase_path,
        artifact.meta_info_path,
    ]:
        _require_file(path)

    rgb_frames = list(io.read_rgb_artifacts(artifact.rgb_path))
    if len(rgb_frames) != expected_frames:
        raise AssertionError(f"RGB artifact has {len(rgb_frames)} frames, expected {expected_frames}.")
    if not all(frame.ndim == 3 and frame.shape[-1] == 3 for _, frame in rgb_frames):
        raise AssertionError("RGB artifact contains frames with invalid shape.")

    pose_npz = np.load(artifact.pose_path)
    _require_frame_indices("pose", pose_npz["inds"], expected_frames)
    pose = pose_npz["data"]
    if pose.shape != (expected_frames, 4, 4) or not np.isfinite(pose).all():
        raise AssertionError(f"Pose artifact has invalid shape or values: {pose.shape}.")
    if not np.allclose(pose[:, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
        raise AssertionError("Pose artifact does not contain valid homogeneous transforms.")

    intr_npz = np.load(artifact.intrinsics_path)
    _require_frame_indices("intrinsics", intr_npz["inds"], expected_frames)
    intrinsics = intr_npz["data"]
    if intrinsics.shape[0] != expected_frames or intrinsics.shape[1] < 4 or not np.isfinite(intrinsics).all():
        raise AssertionError(f"Intrinsics artifact has invalid shape or values: {intrinsics.shape}.")
    if not (intrinsics[:, 0] > 0).all() or not (intrinsics[:, 1] > 0).all():
        raise AssertionError("Intrinsics artifact contains non-positive focal lengths.")

    depth_frames = list(io.read_depth_artifacts(artifact.depth_path))
    _require_tensor_count("depth", depth_frames, expected_frames)
    depth_shape = depth_frames[0][1].shape
    finite_depth_pixels = 0
    positive_depth_pixels = 0
    total_depth_pixels = 0
    for _, depth in depth_frames:
        if depth.shape != depth_shape or depth.ndim != 2:
            raise AssertionError("Depth artifact contains inconsistent frame shapes.")
        finite = torch.isfinite(depth)
        finite_depth_pixels += int(finite.sum())
        positive_depth_pixels += int((depth[finite] > 0).sum())
        total_depth_pixels += depth.numel()
    if finite_depth_pixels < int(0.99 * total_depth_pixels):
        raise AssertionError("Depth artifact contains too many non-finite pixels.")
    if positive_depth_pixels == 0:
        raise AssertionError("Depth artifact contains no positive depth values.")

    mask_frames = list(io.read_instance_artifacts(artifact.mask_path))
    _require_tensor_count("mask", mask_frames, expected_frames)
    for _, mask in mask_frames:
        if mask.shape != depth_shape or mask.ndim != 2:
            raise AssertionError("Mask artifact shape does not match depth shape.")

    phrases = io.read_instance_phrases(artifact.mask_phrase_path)
    if not phrases:
        raise AssertionError("Instance phrase artifact is empty.")

    with artifact.meta_info_path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict) or "ba_residual" not in meta:
        raise AssertionError("Metadata artifact is missing ba_residual.")

    discovered = list(io.ArtifactPath.glob_artifacts(base_path))
    if not any(path.artifact_name == artifact_name for path in discovered):
        raise AssertionError(f"{artifact_name} was not discoverable through ArtifactPath.glob_artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ViPE dog-example smoke test artifacts.")
    parser.add_argument("base_path", type=Path)
    parser.add_argument("artifact_name")
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()

    validate_smoke_output(args.base_path, args.artifact_name, args.expected_frames)
    print(f"Validated {args.artifact_name} artifacts in {args.base_path}")


if __name__ == "__main__":
    main()
