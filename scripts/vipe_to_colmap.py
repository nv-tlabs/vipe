#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world matrix to COLMAP format.
    COLMAP uses world-to-camera transformation.
    """
    w2c = np.linalg.inv(c2w_matrix)
    quaternion = quaternion_from_matrix(w2c)
    translation = w2c[:3, 3]
    return quaternion, translation


def write_cameras_txt(output_dir: Path, artifact: ArtifactPath, frame_width: int, frame_height: int):
    """Write COLMAP cameras.txt file."""
    cameras_file = output_dir / "cameras.txt"

    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)

    # Use first frame's intrinsics (assuming constant intrinsics)
    assert camera_types[0] == CameraType.PINHOLE, "Only PINHOLE camera type is supported"
    fx, fy, cx, cy = intrinsics[0].cpu().numpy()

    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        fx, fy, cx, cy = intrinsics[0]

        # COLMAP camera format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        f.write(f"1 PINHOLE {frame_width} {frame_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    logger.info(f"Written cameras.txt with intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


def write_images_txt(output_dir: Path, artifact: ArtifactPath):
    """Write COLMAP images.txt file."""
    images_file = output_dir / "images.txt"

    # Load pose data
    pose_data = np.load(artifact.pose_path)
    poses = pose_data["data"]  # Shape: (N, 4, 4)
    indices = pose_data["inds"]  # Frame indices

    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for i, (pose_matrix, frame_idx) in enumerate(zip(poses, indices)):
            # Convert pose to COLMAP format
            quaternion, translation = matrix_to_colmap_pose(pose_matrix)
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation

            # Image filename
            image_name = f"images/frame_{frame_idx:06d}.jpg"

            # Write image line
            f.write(f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {image_name}\n")
            # Empty points2D line (no 2D-3D correspondences)
            f.write("\n")

    logger.info(f"Written images.txt with {len(poses)} images")


def write_points3d_txt_from_slam_map(output_dir: Path, artifact: ArtifactPath):
    """Write points3D.txt from SLAM map (placeholder implementation)."""
    assert artifact.slam_map_path.exists(), "SLAM map not found, please refer to README.md for more details."

    slam_map = SLAMMap.load(artifact.slam_map_path, device=torch.device("cpu"))

    points3d_file = output_dir / "points3D.txt"
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {slam_map.dense_disp_xyz.shape[0]}\n")

        point_id = 1
        for keyframe_idx, frame_idx in enumerate(slam_map.dense_disp_frame_inds):
            xyz, rgb = slam_map.get_dense_disp_pcd(keyframe_idx)
            xyz = xyz.cpu().numpy()
            rgb = rgb.cpu().numpy()

            for xyz, rgb in zip(xyz, rgb):
                x, y, z = xyz
                r, g, b = (rgb * 255).astype(np.uint8)
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0 {frame_idx} {point_id} 0 0 0 0\n")
                point_id += 1


def write_points3d_txt_from_depth(
    output_dir: Path, artifact: ArtifactPath, depth_step: int, spatial_subsample: int = 4
):
    """Write empty COLMAP points3D.txt file."""
    _, pose_data = read_pose_artifacts(artifact.pose_path)
    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    camera_type = camera_types[0]
    points3d_file = output_dir / "points3D.txt"

    image_dir = output_dir / "images"
    images = sorted(list(image_dir.glob("*.jpg")))
    # Collect all 3D points first
    all_points = []
    point_id = 1

    rays: np.ndarray | None = None

    for idx, (_, depth) in enumerate(read_depth_artifacts(artifact.depth_path)):
        if idx % 30 == 0:
            logger.info(f"Processed {idx} depth maps")

        if idx % depth_step != 0:
            continue

        image = cv2.imread(str(images[idx]), cv2.IMREAD_COLOR)
        assert image is not None, f"Failed to read image: {images[idx]}"
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = rgb.shape[:2]
        rgb = rgb[::spatial_subsample, ::spatial_subsample]

        if rays is None:
            camera_model = camera_type.build_camera_model(intrinsics[idx])
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_height).float()[::spatial_subsample],
                torch.arange(frame_width).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_height - 1)
                disp_u = disp_u / (frame_width - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        if depth is not None:
            pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
            depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
            rgb, pcd = rgb[depth_mask], pcd[depth_mask]
            c2w_matrix = pose_data[idx].matrix().numpy()
            pcd = pcd @ c2w_matrix[:3, :3].T + c2w_matrix[:3, 3][None]

            for pts_rgb, pts_xyz in zip(rgb, pcd):
                all_points.append(
                    (
                        point_id,
                        pts_xyz[0],
                        pts_xyz[1],
                        pts_xyz[2],
                        int(pts_rgb[0]),
                        int(pts_rgb[1]),
                        int(pts_rgb[2]),
                        0.0,
                        idx + 1,
                    )
                )
                point_id += 1

    # Write points to file
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_points)}\n")

        for point_id, point_data in enumerate(all_points):
            point_id, x, y, z, r, g, b, error, image_id = point_data
            # The last 4 values are for visualization purposes.
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {image_id} {point_id} 0 0 0 0\n")

    logger.info(f"Written points3D.txt with {len(all_points)} points")


def extract_frames(artifact: ArtifactPath, output_dir: Path) -> Tuple[int, int]:
    """Extract frames from video to individual image files."""
    video_path = artifact.rgb_path
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting frames from {video_path}")

    for frame_idx, rgb in read_rgb_artifacts(video_path):
        frame_path = images_dir / f"frame_{frame_idx:06d}.jpg"
        frame_height, frame_width = rgb.shape[:2]
        imageio.imwrite(str(frame_path), (rgb.cpu().numpy() * 255).astype(np.uint8))
        if frame_idx % 30 == 0:
            logger.info(f"Extracted {frame_idx} frames")

    logger.info(f"Extracted {frame_idx} frames to {images_dir}")

    return frame_width, frame_height


def convert_vipe_to_colmap(artifact: ArtifactPath, output_path: Path, depth_step: int, use_slam_map: bool):
    """Convert ViPE reconstruction results to COLMAP format."""

    logger.info(
        f"Converting ViPE results from {artifact.base_path} ({artifact.artifact_name}) to COLMAP format at {output_path}"
    )

    # Verify required files exist
    required_files = [artifact.rgb_path, artifact.pose_path, artifact.intrinsics_path, artifact.depth_path]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract frames and get video dimensions
    frame_width, frame_height = extract_frames(artifact, output_path)

    # Write COLMAP files
    write_cameras_txt(output_path, artifact, frame_width, frame_height)
    write_images_txt(output_path, artifact)
    if use_slam_map:
        write_points3d_txt_from_slam_map(output_path, artifact)
    else:
        write_points3d_txt_from_depth(output_path, artifact, depth_step)

    logger.info("COLMAP conversion completed successfully!")
    logger.info(f"Output directory: {output_path}")
    logger.info("Files created:")
    logger.info("  - cameras.txt: Camera intrinsics")
    logger.info("  - images.txt: Camera poses")
    logger.info("  - points3D.txt: 3D points")
    logger.info("  - images/: Individual frame images")


def main():
    """Main function for ViPE to COLMAP conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert ViPE reconstruction results to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vipe_path", type=Path, help="Path to ViPE results directory")
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        help="Sequence name (if not provided will convert all sequences in the directory)",
        default=None,
    )
    parser.add_argument("--use_slam_map", action="store_true", help="Use SLAM map to unproject depth maps")
    parser.add_argument("--depth_step", type=int, default=16, help="Step size for depth extraction (default: 16)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for COLMAP format (default: <vipe_path>_colmap)",
    )

    args = parser.parse_args()

    if not args.vipe_path.exists():
        print(f"Error: ViPE path '{args.vipe_path}' does not exist.")
        return 1

    # Find artifacts
    artifacts = list(ArtifactPath.glob_artifacts(args.vipe_path, use_video=True))
    if args.sequence is not None:
        artifacts = [artifact for artifact in artifacts if artifact.artifact_name == args.sequence]

    # Set default output path
    if args.output is None:
        args.output = args.vipe_path.parent / f"{args.vipe_path.name}_colmap"

    for artifact in artifacts:
        convert_vipe_to_colmap(artifact, args.output / artifact.artifact_name, args.depth_step, args.use_slam_map)
    return 0


if __name__ == "__main__":
    exit(main())
