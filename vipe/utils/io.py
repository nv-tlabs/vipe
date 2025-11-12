# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import zlib
import json
import logging
import struct
import tempfile
import zipfile

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import Imath
import numpy as np
import OpenEXR
import torch

from vipe.ext.lietorch import SE3
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import VideoWriter


logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.bin.zlib"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}_camera.txt"

    @property
    def flow_path(self) -> Path:
        return self.base_path / "flow" / f"{self.artifact_name}.zip"

    @property
    def mask_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.zip"

    @property
    def mask_phrase_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.txt"

    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_info.pkl"

    @property
    def binary_path(self) -> Path:
        return self.base_path / "binary" / f"{self.artifact_name}.bin"

    @classmethod
    def glob_artifacts(cls, base_path: Path, use_video: bool = False) -> Iterator["ArtifactPath"]:
        if use_video:
            for artifact_path in (base_path / "rgb").glob("*.mp4"):
                artifact_name = artifact_path.stem
                yield cls(base_path, artifact_name)
        else:
            for artifact_path in (base_path / "vipe").glob("*_info.pkl"):
                artifact_name = artifact_path.stem.replace("_info", "")
                yield cls(base_path, artifact_name)

    @property
    def meta_vis_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_vis.mp4"

    @property
    def slam_map_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_slam_map.pt"

    @property
    def essential_paths(self) -> list[Path]:
        return [
            self.rgb_path,
            self.pose_path,
            self.depth_path,
            self.intrinsics_path,
            self.flow_path,
            self.mask_path,
            self.mask_phrase_path,
            self.meta_info_path,
            self.meta_vis_path,
        ]

    @property
    def eval_metrics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_metrics.pkl"

    @property
    def eval_traj_vis_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_trajectory_vis.png"

    @property
    def eval_gt_pose_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_pose_gt.npz"

    @property
    def eval_gt_intrinsics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_intrinsics_gt.npz"

    @property
    def eval_gt_camera_type_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_camera_gt.txt"

    @property
    def eval_gt_depth_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_depth_gt.bin.zlib"

    @property
    def aux_vis_plot_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_plot.png"

    @property
    def aux_vis_traj_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_traj.mp4"


def save_pose_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save OpenCV cam2world matrices as binary file
    # Also compute and save world2cam (inverse) matrices
    if gt:
        pose_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.POSE)
        base_path = out_path.base_path / "pose"
    else:
        pose_list = cached_final_stream.get_stream_attribute(FrameAttribute.POSE)
        base_path = out_path.base_path / "pose"

    pose_list = [
        (frame_idx, pose_data.matrix().cpu().numpy())
        for frame_idx, pose_data in enumerate(pose_list)
        if pose_data is not None
    ]
    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0).astype(np.float32)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        
        # Compute inverse poses (world2cam)
        poses_inv = np.linalg.inv(pose_data.astype(np.float64)).astype(np.float32)
        
        base_path.mkdir(exist_ok=True, parents=True)
        
        # Save cam2world poses
        pose_data.tofile(base_path / "poses.bin")
        
        # Save world2cam poses
        poses_inv.tofile(base_path / "poses_inv.bin")
        
        # Save metadata
        metadata = {
            "shape": pose_data.shape,
            "dtype": "float32",
            "frame_indices": pose_inds.tolist()
        }
        with open(base_path / "poses_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def read_pose_artifacts(npz_file_path: Path) -> tuple[np.ndarray, SE3]:
    data = np.load(npz_file_path)
    return data["inds"], se3_matrix_to_se3(data["data"])


def read_pose_artifacts_benchmark(npz_file_path: Path) -> dict:
    data = np.load(npz_file_path)
    return dict(
        ids=data["inds"],
        trajectory=se3_matrix_to_se3(data["data"]),
        runtime=data.get("runtime", None),
        keyframe_ids=data.get("keyframe_ids", None),
        frame_num=len(data["inds"]),
    )


def save_intrinsics_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save intrinsics as [fx, fy, cx, cy] in binary file
    if gt:
        intrinsics_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.CAMERA_TYPE)
        base_path = out_path.base_path / "intrinsics"
        camera_type_path = out_path.eval_gt_camera_type_path
    else:
        intrinsics_list = cached_final_stream.get_stream_attribute(FrameAttribute.INTRINSICS)
        camera_type_list = cached_final_stream.get_stream_attribute(FrameAttribute.CAMERA_TYPE)
        base_path = out_path.base_path / "intrinsics"
        camera_type_path = out_path.camera_type_path

    intrinsics_list = [
        (frame_idx, intr_data.cpu().numpy())
        for frame_idx, intr_data in enumerate(intrinsics_list)
        if intr_data is not None
    ]
    if len(intrinsics_list) > 0:
        intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0).astype(np.float32)
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
        base_path.mkdir(exist_ok=True, parents=True)
        
        # Save intrinsics as binary
        intrinsics_data.tofile(base_path / "intrinsics.bin")
        
        # Save metadata
        metadata = {
            "shape": intrinsics_data.shape,
            "dtype": "float32",
            "format": "[fx, fy, cx, cy] per frame",
            "frame_indices": intrinsics_inds.tolist()
        }
        with open(base_path / "intrinsics_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    camera_type_list = [
        (frame_idx, camera_type_data)
        for frame_idx, camera_type_data in enumerate(camera_type_list)
        if camera_type_data is not None
    ]
    if len(camera_type_list) > 0:
        camera_type_path.parent.mkdir(exist_ok=True, parents=True)
        with camera_type_path.open("w") as f:
            for frame_idx, camera_type_data in camera_type_list:
                f.write(f"{frame_idx}: {camera_type_data.name}\n")


def read_intrinsics_artifacts(
    intr_file_path: Path, camera_file_path: Path | None = None
) -> tuple[np.ndarray, torch.Tensor, list[CameraType]]:
    data = np.load(intr_file_path)
    inds, intrinsics = data["inds"], torch.from_numpy(data["data"])
    if camera_file_path is None or not camera_file_path.exists():
        assert intrinsics.shape[1] == 4
        camera_types = [CameraType.PINHOLE] * intrinsics.shape[0]

    else:
        with camera_file_path.open("r") as f:
            camera_types = [CameraType[line.split(":")[1].strip()] for line in f.readlines()]

    return inds, intrinsics, camera_types


def save_rgb_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    # Save original RGB as H264-encoded video.
    with VideoWriter(out_path.rgb_path, cached_final_stream.fps()) as rgb_writer:
        for frame_data in cached_final_stream:
            rgb_writer.write((frame_data.rgb.cpu().numpy() * 255).astype(np.uint8))


def save_rgb_frames_as_webp(
    out_path: ArtifactPath, 
    cached_final_stream: VideoStream,
    quality: int = 95
) -> None:
    import cv2
    
    rgb_dir = out_path.rgb_path.parent / out_path.artifact_name
    rgb_dir.mkdir(exist_ok=True, parents=True)
    
    for frame_idx, frame_data in enumerate(cached_final_stream):
        rgb = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_path = rgb_dir / f"{frame_idx:05d}.webp"
        
        success, buf = cv2.imencode('.webp', bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not success:
            _, buf = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            frame_path = rgb_dir / f"{frame_idx:05d}.png"
        
        with open(frame_path, 'wb') as f:
            f.write(buf.tobytes())


def read_rgb_artifacts(rgb_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read RGB from H264-encoded video.
    """
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        yield frame_idx, rgb


def save_depth_artifacts(
    out_path: ArtifactPath, 
    cached_final_stream: VideoStream, 
    gt: bool = False,
    zlib_level: int = 6,
    separate_frames: bool = True,
) -> None:
    """Save depth frames as individual per-frame binaries.
    
    Args:
        out_path: Output artifact path
        cached_final_stream: Video stream with depth data
        gt: Whether this is ground truth depth
        zlib_level: Compression level (0-9). Set to 0 to disable compression.
        separate_frames: If True, save each frame as separate .bin file (default, recommended).
    
    Note:
        Uses compression level 3 automatically for optimal speed/size balance.
        Compression adds ~10-20ms per frame but reduces size by ~50%.
    """
    if gt:
        metric_depth_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.METRIC_DEPTH)
        base_path = out_path.eval_gt_depth_path.parent
    else:
        metric_depth_list = cached_final_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
        base_path = out_path.depth_path.parent

    metric_depth_list = [
        (frame_idx, depth_data.cpu().numpy())
        for frame_idx, depth_data in enumerate(metric_depth_list)
        if depth_data is not None
    ]
    
    if len(metric_depth_list) == 0:
        return
    
    base_path.mkdir(exist_ok=True, parents=True)
    
    if not separate_frames:
        logger.warning("separate_frames=False is deprecated. Using separate_frames=True.")
    
    # Save each depth frame as individual binary file
    depth_dir = base_path / out_path.artifact_name
    depth_dir.mkdir(exist_ok=True, parents=True)
    
    # Get frame shape from first frame
    first_depth = metric_depth_list[0][1]
    H, W = first_depth.shape
    
    # Create manifest with metadata
    manifest = {
        "format": "fp16_zlib_per_frame",
        "dtype": "float16",
        "compression": f"zlib_level_{zlib_level}",
        "total_frames": len(metric_depth_list),
        "frame_shape": [H, W],
        "frames": {}
    }
    
    # Use lower compression for per-frame (faster, still good ratio)
    # Per-frame compression is less efficient, so we use level 3 instead of 6
    # This trades ~5% larger files for ~2x faster compression
    frame_zlib_level = min(3, zlib_level) if zlib_level > 0 else 0
    
    # Save each frame
    for frame_idx, depth_data in metric_depth_list:
        frame_path = depth_dir / f"{frame_idx:05d}.bin"
        
        # Convert to fp16
        depth_bytes = depth_data.astype(np.float16).tobytes()
        
        # Compress if requested, otherwise save raw
        if frame_zlib_level > 0:
            compressed_bytes = zlib.compress(depth_bytes, level=frame_zlib_level)
        else:
            compressed_bytes = depth_bytes
        
        # Save frame data
        with open(frame_path, 'wb') as f:
            f.write(compressed_bytes)
        
        # Record frame metadata
        manifest["frames"][frame_idx] = {
            "uncompressed_size": len(depth_bytes),
            "compressed_size": len(compressed_bytes),
            "shape": [H, W]
        }
    
    # Save manifest
    manifest_path = depth_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def read_depth_artifacts(depth_path: Path, start_idx: int = 0, end_idx: int = -1) -> tuple[np.ndarray, dict]:
    """Read depth from single zlib-compressed binary file."""
    with open(depth_path, 'rb') as f:
        metadata_len = struct.unpack('I', f.read(4))[0]
        metadata_json = f.read(metadata_len)
        metadata = json.loads(metadata_json.decode('utf-8'))
        compressed_bytes = f.read()
    
    depth_bytes = zlib.decompress(compressed_bytes)
    T, H, W = metadata["shape"]
    depth_array = np.frombuffer(depth_bytes, dtype=np.float16).reshape(T, H, W).astype(np.float32)
    
    if end_idx < 0:
        end_idx = T
    
    return depth_array[start_idx:end_idx], metadata


def read_instance_artifacts(
    zip_file_path: Path,
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read instance mask from zipped PNG files.
    """
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                yield frame_idx, torch.from_numpy(mask.copy()).byte()


def read_instance_phrases(instance_phrase_path: Path) -> dict[int, str]:
    """
    Read instance phrases from txt file.
    """
    instance_phrases = {}
    with instance_phrase_path.open("r") as f:
        for line in f.readlines():
            idx, phrase = line.split(":")
            instance_phrases[int(idx)] = phrase.strip()
    return instance_phrases


def save_binary_artifacts(
    out_path: ArtifactPath,
    cached_final_stream: VideoStream,
    rgb_format: str = "webp",
    rgb_quality: int = 95,
) -> None:
    """Save RGB, depth, poses, and intrinsics in a single binary file with JSON header."""
    # Collect all frame data
    rgb_list = []
    depth_list = []
    pose_list = []
    intrinsics_list = []
    
    for frame_data in cached_final_stream:
        assert isinstance(frame_data, VideoFrame)
        
        # RGB
        rgb = (frame_data.rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_list.append(rgb)
        
        # Depth
        if frame_data.metric_depth is not None:
            depth = frame_data.metric_depth.cpu().numpy().astype(np.float16)
        else:
            depth = np.full(rgb.shape[:2], float('nan'), dtype=np.float16)
        depth_list.append(depth)
        
        # Pose
        if frame_data.pose is not None:
            pose = frame_data.pose.matrix().cpu().numpy().astype(np.float32)
        else:
            pose = np.eye(4, dtype=np.float32)
        pose_list.append(pose)
        
        # Intrinsics - convert from [fx, fy, cx, cy] to 3x3 matrix
        if frame_data.intrinsics is not None:
            intr = frame_data.intrinsics.cpu().numpy()
            fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
            intrinsics_mat = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            intrinsics_mat = np.eye(3, dtype=np.float64)
        intrinsics_list.append(intrinsics_mat)
    
    if len(rgb_list) == 0:
        logger.warning("No frames to save in binary format")
        return
    
    # Stack all arrays
    rgb_array = np.stack(rgb_list, axis=0)  # [T, H, W, 3]
    depth_array = np.stack(depth_list, axis=0)  # [T, H, W]
    intrinsics_array = np.stack(intrinsics_list, axis=0)  # [T, 3, 3]
    poses_array = np.stack(pose_list, axis=0)  # [T, 4, 4]
    
    # Compute inverse poses
    poses_inv_array = np.linalg.inv(poses_array).astype(np.float32)  # [T, 4, 4]
    
    # Compute metadata
    T, H, W = rgb_array.shape[:3]
    depth_valid = depth_array[~np.isnan(depth_array)]
    depth_range = [float(depth_valid.min()), float(depth_valid.max())] if len(depth_valid) > 0 else [0.0, 1.0]
    
    # Compute FOV from first frame intrinsics
    fx, fy = intrinsics_array[0, 0, 0], intrinsics_array[0, 1, 1]
    cx, cy = intrinsics_array[0, 0, 2], intrinsics_array[0, 1, 2]
    fov_x = float(2 * np.arctan(W / (2 * fx)) * 180 / np.pi)
    fov_y = float(2 * np.arctan(H / (2 * fy)) * 180 / np.pi)
    
    # Get FPS
    fps = cached_final_stream.fps()
    
    data_blob = b""
    offsets = {}
    
    def compress_array(data: bytes) -> bytes:
        if HAS_ZSTD:
            return zstd.ZstdCompressor(level=19, threads=-1).compress(data)
        return gzip.compress(data, compresslevel=9)
    
    # RGB compression
    rgb_format_lower = rgb_format.lower()
    rgb_compressed_frames = []
    
    if rgb_format_lower == "webp":
        try:
            success, _ = cv2.imencode('.webp', rgb_array[0], [cv2.IMWRITE_WEBP_QUALITY, rgb_quality])
            if not success:
                raise RuntimeError("WebP not available")
            for i in range(T):
                _, buf = cv2.imencode('.webp', rgb_array[i], [cv2.IMWRITE_WEBP_QUALITY, rgb_quality])
                rgb_compressed_frames.append(buf.tobytes())
            compression_type = "webp"
        except:
            rgb_format_lower = "jpeg"
    
    if rgb_format_lower == "jpeg":
        for i in range(T):
            _, buf = cv2.imencode('.jpg', rgb_array[i], [cv2.IMWRITE_JPEG_QUALITY, rgb_quality])
            rgb_compressed_frames.append(buf.tobytes())
        compression_type = "jpeg"
    elif rgb_format_lower == "png":
        for i in range(T):
            _, buf = cv2.imencode('.png', rgb_array[i], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            rgb_compressed_frames.append(buf.tobytes())
        compression_type = "png"
    
    offsets["rgb"] = {
        "offset": len(data_blob),
        "dtype": "uint8",
        "shape": [T, H, W, 3],
        "compression": compression_type,
        "quality": rgb_quality if compression_type != "png" else None,
        "frame_offsets": []
    }
    
    for frame_data in rgb_compressed_frames:
        offsets["rgb"]["frame_offsets"].append(len(data_blob))
        data_blob += struct.pack('<I', len(frame_data))  # Frame size
        data_blob += frame_data
    offsets["rgb"]["length"] = len(data_blob) - offsets["rgb"]["offset"]
    
    # Depth
    offsets["depth"] = {"offset": len(data_blob), "dtype": "float16", "shape": [T, H, W]}
    depth_bytes = compress_array(depth_array.tobytes())
    offsets["depth"]["length"] = len(depth_bytes)
    data_blob += depth_bytes
    
    # Intrinsics
    offsets["intrinsics"] = {"offset": len(data_blob), "dtype": "float64", "shape": [T, 3, 3]}
    intrinsics_bytes = compress_array(intrinsics_array.tobytes())
    offsets["intrinsics"]["length"] = len(intrinsics_bytes)
    data_blob += intrinsics_bytes
    
    # Poses (delta encoded)
    offsets["poses"] = {"offset": len(data_blob), "dtype": "float32", "shape": [T, 4, 4], "encoding": "delta"}
    poses_delta = np.diff(poses_array, axis=0, prepend=poses_array[:1])
    poses_bytes = compress_array(poses_delta.tobytes())
    offsets["poses"]["length"] = len(poses_bytes)
    data_blob += poses_bytes
    
    # Poses inverse (delta encoded)
    offsets["poses_inv"] = {"offset": len(data_blob), "dtype": "float32", "shape": [T, 4, 4], "encoding": "delta"}
    poses_inv_delta = np.diff(poses_inv_array, axis=0, prepend=poses_inv_array[:1])
    poses_inv_bytes = compress_array(poses_inv_delta.tobytes())
    offsets["poses_inv"]["length"] = len(poses_inv_bytes)
    data_blob += poses_inv_bytes
    
    header = {
        **offsets,
        "meta": {
            "depth_range": depth_range,
            "total_frames": T,
            "resolution": [W, H],
            "base_fps": fps,
            "fov": fov_y,
            "fov_x": fov_x,
            "original_aspect_ratio": float(W / H),
            "fixed_aspect_ratio": float(W / H),
        }
    }
    
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    out_path.binary_path.parent.mkdir(exist_ok=True, parents=True)
    
    with out_path.binary_path.open('wb') as f:
        f.write(struct.pack('<I', len(header_json)))
        f.write(header_json)
        f.write(data_blob)


def save_manifest(
    out_path: ArtifactPath, 
    cached_final_stream: VideoStream
) -> None:
    """Save manifest.json with metadata and file structure."""
    
    depth_format_info = {
        "format": "fp16_zlib",
        "file": f"depth/{out_path.artifact_name}.bin.zlib",
        "frame_count": 0,
        "dtype": "float16",
        "units": "meters",
        "description": "Single-file zlib-compressed fp16 binary"
    }
    
    manifest = {
        "format_version": "1.0",
        "data": {
            "rgb": {
                "format": "webp_frames",
                "directory": f"rgb/{out_path.artifact_name}/",
                "frame_count": 0,
                "resolution": [0, 0],
                "description": "Individual WebP frames (00000.webp, 00001.webp, ...)"
            },
            "depth": depth_format_info,
            "poses": {
                "file": "pose/poses.bin",
                "meta_file": "pose/poses_meta.json",
                "dtype": "float32",
                "shape": [0, 4, 4],
                "byte_order": "little",
                "description": "cam2world transforms (OpenCV convention) - raw binary float32"
            },
            "poses_inv": {
                "file": "pose/poses_inv.bin",
                "meta_file": "pose/poses_meta.json",
                "dtype": "float32",
                "shape": [0, 4, 4],
                "byte_order": "little",
                "description": "world2cam transforms (computed as inverse) - raw binary float32"
            },
            "intrinsics": {
                "file": "intrinsics/intrinsics.bin",
                "meta_file": "intrinsics/intrinsics_meta.json",
                "dtype": "float32",
                "shape": [0, 4],
                "byte_order": "little",
                "description": "camera intrinsics per frame [fx, fy, cx, cy] - raw binary float32"
            }
        },
        "metadata": {}
    }
    
    # Collect frame data to compute metadata
    frames = list(cached_final_stream)
    if not frames:
        return
    
    T = len(frames)
    first_frame = frames[0]
    H, W = first_frame.rgb.shape[0], first_frame.rgb.shape[1]
    
    # Update frame counts and resolution
    manifest["data"]["rgb"]["frame_count"] = T
    manifest["data"]["rgb"]["resolution"] = [W, H]
    manifest["data"]["depth"]["frame_count"] = T
    manifest["data"]["poses"]["shape"] = [T, 4, 4]
    manifest["data"]["poses_inv"]["shape"] = [T, 4, 4]
    manifest["data"]["intrinsics"]["shape"] = [T, 4]
    
    # Compute metadata
    depth_values = []
    for frame in frames:
        if frame.metric_depth is not None:
            depth_valid = frame.metric_depth[~torch.isnan(frame.metric_depth)]
            if len(depth_valid) > 0:
                depth_values.extend(depth_valid.cpu().numpy().tolist())
    
    if depth_values:
        manifest["metadata"]["depth_range"] = [float(min(depth_values)), float(max(depth_values))]
    else:
        manifest["metadata"]["depth_range"] = [0.0, 1.0]
    
    manifest["metadata"]["total_frames"] = T
    manifest["metadata"]["resolution"] = [W, H]
    manifest["metadata"]["base_fps"] = cached_final_stream.fps()
    
    # Compute FOV from first frame intrinsics
    if first_frame.intrinsics is not None:
        intr = first_frame.intrinsics.cpu().numpy()
        fx, fy = intr[0], intr[1]
        fov_x = float(2 * np.arctan(W / (2 * fx)) * 180 / np.pi)
        fov_y = float(2 * np.arctan(H / (2 * fy)) * 180 / np.pi)
        manifest["metadata"]["fov_x"] = fov_x
        manifest["metadata"]["fov_y"] = fov_y
    
    manifest["metadata"]["aspect_ratio"] = float(W / H)
    
    # Save manifest
    manifest_path = out_path.base_path / "manifest.json"
    manifest_path.parent.mkdir(exist_ok=True, parents=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Saved manifest to {manifest_path}")


def save_artifacts(
    out_path: ArtifactPath, 
    cached_final_stream: VideoStream,
    zlib_level: int = 6,
    rgb_quality: int = 95
) -> None:
    save_pose_artifacts(out_path, cached_final_stream)
    save_intrinsics_artifacts(out_path, cached_final_stream)
    save_rgb_frames_as_webp(out_path, cached_final_stream, quality=rgb_quality)
    save_depth_artifacts(out_path, cached_final_stream, zlib_level=zlib_level)
    save_manifest(out_path, cached_final_stream)

    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_final_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_final_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")
