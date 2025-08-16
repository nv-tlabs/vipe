#!/usr/bin/env python3
"""
ViPE to GEN3C Format Converter

This script converts ViPE output format to GEN3C input format.
ViPE outputs data in separate subdirectories with specific formats,
while GEN3C expects a unified directory structure.
"""

import numpy as np
import cv2
import zipfile
import OpenEXR
import Imath
from pathlib import Path
import shutil
import argparse
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_vipe_depth_data(depth_zip_path: Path) -> np.ndarray:
    """
    Read depth data from ViPE's ZIP file containing EXR files.
    
    Args:
        depth_zip_path: Path to the depth.zip file
        
    Returns:
        numpy array of shape [T, H, W] with depth values
    """
    depth_arrays = []
    
    with zipfile.ZipFile(depth_zip_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                    header = exr.header()
                    dw = header["dataWindow"]
                    width = dw.max.x - dw.min.x + 1
                    height = dw.max.y - dw.min.y + 1
                    channels = exr.channels(["Z"])
                    depth_data = np.frombuffer(channels[0], dtype=np.float16)
                    depth_data = depth_data.reshape((height, width)).astype(np.float32)
                    depth_arrays.append(depth_data)
                    logger.debug(f"Loaded depth frame {frame_idx}: {width}x{height}")
                except Exception as e:
                    logger.warning(f"Failed to load depth frame {file_name}: {e}")
                    # Create a dummy depth map filled with NaN
                    if depth_arrays:
                        h, w = depth_arrays[-1].shape
                        depth_arrays.append(np.full((h, w), np.nan, dtype=np.float32))
                    else:
                        raise ValueError("Cannot determine depth map dimensions")
    
    return np.stack(depth_arrays, axis=0)


def read_vipe_mask_data(mask_zip_path: Path) -> np.ndarray:
    """
    Read mask data from ViPE's ZIP file containing PNG files.
    
    Args:
        mask_zip_path: Path to the mask.zip file
        
    Returns:
        numpy array of shape [T, H, W] with mask values
    """
    mask_arrays = []
    
    with zipfile.ZipFile(mask_zip_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                    mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        raise ValueError(f"Failed to decode PNG mask {file_name}")
                    mask_arrays.append(mask.astype(np.float32))
                    logger.debug(f"Loaded mask frame {frame_idx}: {mask.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load mask frame {file_name}: {e}")
                    # Create a dummy mask filled with ones
                    if mask_arrays:
                        h, w = mask_arrays[-1].shape[:2]
                        mask_arrays.append(np.ones((h, w), dtype=np.float32))
                    else:
                        raise ValueError("Cannot determine mask dimensions")
    
    return np.stack(mask_arrays, axis=0)


def convert_intrinsics_to_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Convert intrinsic parameters to 3x3 matrix format.
    
    Args:
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
        
    Returns:
        3x3 intrinsics matrix
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def interpolate_sparse_data(data: np.ndarray, indices: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolate sparse data to fill all frames.
    
    Args:
        data: Data array with shape [N, ...]
        indices: Frame indices corresponding to data
        target_length: Total number of frames needed
        
    Returns:
        Interpolated data array with shape [target_length, ...]
    """
    if len(data) == 0:
        raise ValueError("No data to interpolate")
    
    result = []
    for i in range(target_length):
        if i in indices:
            # Use exact data
            idx = np.where(indices == i)[0][0]
            result.append(data[idx])
        else:
            # Find nearest available data
            if i < indices[0]:
                # Use first available
                result.append(data[0])
            elif i > indices[-1]:
                # Use last available
                result.append(data[-1])
            else:
                # Interpolate between nearest neighbors
                prev_idx = np.where(indices < i)[0][-1]
                next_idx = np.where(indices > i)[0][0]
                
                # Simple linear interpolation for matrices
                alpha = (i - indices[prev_idx]) / (indices[next_idx] - indices[prev_idx])
                interpolated = (1 - alpha) * data[prev_idx] + alpha * data[next_idx]
                result.append(interpolated)
    
    return np.stack(result, axis=0)


def convert_vipe_to_gen3c(vipe_dir: str, output_dir: str, video_name: Optional[str] = None) -> Path:
    """
    Convert ViPE output format to GEN3C input format.
    
    Args:
        vipe_dir: Path to ViPE output directory
        output_dir: Path to output directory for GEN3C format
        video_name: Optional specific video name to convert (if None, uses first found)
        
    Returns:
        Path to the converted output directory
    """
    vipe_path = Path(vipe_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting ViPE output from {vipe_dir} to {output_dir}")
    
    # Find video files
    rgb_files = list((vipe_path / "rgb").glob("*.mp4"))
    if not rgb_files:
        raise FileNotFoundError(f"No MP4 files found in {vipe_path / 'rgb'}")
    
    # If video_name specified, find matching file
    if video_name:
        matching_files = [f for f in rgb_files if video_name in f.stem]
        if not matching_files:
            raise FileNotFoundError(f"No video file found matching '{video_name}'")
        rgb_file = matching_files[0]
        base_name = rgb_file.stem
    else:
        rgb_file = rgb_files[0]
        base_name = rgb_file.stem
    
    logger.info(f"Converting video: {base_name}")
    
    # 1. Copy RGB video (rename to standard name)
    logger.info("Copying RGB video...")
    shutil.copy(rgb_file, output_path / "rgb.mp4")
    
    # 2. Convert depth data
    depth_files = list((vipe_path / "depth").glob(f"{base_name}.zip"))
    if depth_files:
        logger.info("Converting depth data...")
        depth_stack = read_vipe_depth_data(depth_files[0])
        np.savez_compressed(output_path / "depth.npz", depth=depth_stack)
        num_frames = len(depth_stack)
        logger.info(f"Converted {num_frames} depth frames")
    else:
        logger.warning("No depth data found")
        num_frames = None
    
    # 3. Convert camera parameters
    pose_files = list((vipe_path / "pose").glob(f"{base_name}.npz"))
    intrinsics_files = list((vipe_path / "intrinsics").glob(f"{base_name}.npz"))
    
    if pose_files and intrinsics_files:
        logger.info("Converting camera parameters...")
        
        # Load pose data
        pose_data = np.load(pose_files[0])
        w2c_matrices = pose_data['data']  # [N, 4, 4]
        pose_inds = pose_data['inds']
        
        # Load intrinsics data
        intr_data = np.load(intrinsics_files[0])
        intr_params = intr_data['data']  # [N, 4] - fx, fy, cx, cy
        intr_inds = intr_data['inds']
        
        # Determine number of frames
        if num_frames is None:
            # Use the maximum index from camera data + 1
            num_frames = max(pose_inds.max(), intr_inds.max()) + 1
            logger.info(f"Inferred {num_frames} frames from camera data")
        
        # Interpolate poses to fill all frames
        full_w2c = interpolate_sparse_data(w2c_matrices, pose_inds, num_frames)
        
        # Convert and interpolate intrinsics
        intr_matrices = np.array([convert_intrinsics_to_matrix(*params) for params in intr_params])
        full_intrinsics = interpolate_sparse_data(intr_matrices, intr_inds, num_frames)
        
        # Save combined camera parameters
        np.savez(output_path / "camera.npz",
                 w2c=full_w2c.astype(np.float32),
                 intrinsics=full_intrinsics.astype(np.float32))
        
        logger.info(f"Converted camera parameters for {num_frames} frames")
    else:
        raise FileNotFoundError("Missing pose or intrinsics files")
    
    # 4. Convert mask data (optional)
    mask_files = list((vipe_path / "mask").glob(f"{base_name}.zip"))
    if mask_files:
        logger.info("Converting mask data...")
        mask_stack = read_vipe_mask_data(mask_files[0])
        np.savez_compressed(output_path / "mask.npz", mask=mask_stack)
        logger.info(f"Converted {len(mask_stack)} mask frames")
    else:
        logger.info("No mask data found, creating default masks...")
        # Create default masks (all ones)
        if num_frames and 'depth_stack' in locals():
            _, h, w = depth_stack.shape
            default_mask = np.ones((num_frames, h, w), dtype=np.float32)
            np.savez_compressed(output_path / "mask.npz", mask=default_mask)
    
    logger.info(f"Conversion completed successfully: {vipe_dir} -> {output_dir}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ViPE output to GEN3C input format")
    parser.add_argument("vipe_dir", help="Path to ViPE output directory")
    parser.add_argument("output_dir", help="Path to output directory for GEN3C format")
    parser.add_argument("--video-name", help="Specific video name to convert (optional)")
    parser.add_argument("--run-gen3c", action="store_true", help="Run GEN3C after conversion")
    parser.add_argument("--trajectory", default="up", choices=[
        "left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise"
    ], help="Camera trajectory for GEN3C (default: up)")
    parser.add_argument("--movement-distance", type=float, default=0.5, 
                       help="Camera movement distance (default: 0.5)")
    parser.add_argument("--camera-rotation", default="center_facing", 
                       choices=["center_facing", "no_rotation", "trajectory_aligned"],
                       help="Camera rotation mode (default: center_facing)")
    parser.add_argument("--checkpoint-dir", default="GEN3C/checkpoints",
                       help="GEN3C checkpoint directory")
    parser.add_argument("--video-save-name", default="converted_output",
                       help="Output video name for GEN3C")
    
    args = parser.parse_args()
    
    try:
        # Convert format
        converted_path = convert_vipe_to_gen3c(args.vipe_dir, args.output_dir, args.video_name)
        
        if args.run_gen3c:
            logger.info("Running GEN3C with converted data...")
            import subprocess
            
            cmd = [
                "python", "GEN3C/cosmos_predict1/diffusion/inference/gen3c_dynamic.py",
                "--checkpoint_dir", args.checkpoint_dir,
                "--input_image_path", str(converted_path),
                "--video_save_name", args.video_save_name,
                "--trajectory", args.trajectory,
                "--movement_distance", str(args.movement_distance),
                "--camera_rotation", args.camera_rotation,
                "--guidance", "1"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("GEN3C completed successfully!")
                logger.info(f"Output: {result.stdout}")
            else:
                logger.error(f"GEN3C failed with return code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()