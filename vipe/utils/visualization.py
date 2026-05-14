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

from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycg import image

from vipe.ext.lietorch import SE3
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

from .geometry import project_points_to_panorama

rng = np.random.RandomState(200)
_palette = ((rng.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette

POINTS_STENCIL: np.ndarray
points_stencil_grid = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
POINTS_STENCIL = np.stack(points_stencil_grid, axis=-1).reshape(-1, 2)
POINTS_STENCIL = POINTS_STENCIL[np.max(np.abs(POINTS_STENCIL), axis=-1) > 1]
POINTS_STENCIL = np.pad(POINTS_STENCIL, ((0, 1), (0, 0)), constant_values=0)


class VideoWriter:
    """
    Simple video writer class (use h264 codec).

    Usage:
    ```
    with VideoWriter("output.mp4", 30) as vw:
        for frame in frames:
            vw.write(frame)
    ```
    """

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.vw: Any = None

    def __enter__(self):
        return self

    def write(self, frame: np.ndarray):
        if self.vw is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", macro_block_size=None)

        if frame.dtype in [np.float32, np.float64]:
            frame = (frame * 255).astype(np.uint8)

        assert self.vw is not None
        self.vw.append_data(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vw is not None:
            self.vw.close()


def bbox_with_size(pcd_xyz: torch.Tensor, quantile: float = 0.98):
    from pycg import vis

    low_quantile, high_quantile = (1 - quantile) / 2, 1 - (1 - quantile) / 2
    pcd_min = torch.quantile(pcd_xyz, low_quantile, dim=0, keepdim=True)
    pcd_max = torch.quantile(pcd_xyz, high_quantile, dim=0, keepdim=True)
    pcd_min_vec = pcd_min[0]
    pcd_max_vec = pcd_max[0]

    x_length = pcd_max_vec[0] - pcd_min_vec[0]
    x_length_pos = pcd_min_vec + pcd_min_vec.new_tensor([x_length.item() / 2, 0.0, 0.0])
    y_length = pcd_max_vec[1] - pcd_min_vec[1]
    y_length_pos = pcd_min_vec + pcd_min_vec.new_tensor([0.0, y_length.item() / 2, 0.0])
    z_length = pcd_max_vec[2] - pcd_min_vec[2]
    z_length_pos = pcd_min_vec + pcd_min_vec.new_tensor([0.0, 0.0, z_length.item() / 2])

    return [
        vis.wireframe_bbox(pcd_min_vec.detach().cpu().numpy(), pcd_max_vec.detach().cpu().numpy(), ucid=-1),
        vis.text(f"{x_length.item():.2f}m", x_length_pos.detach().cpu().numpy()),
        vis.text(f"{y_length.item():.2f}m", y_length_pos.detach().cpu().numpy()),
        vis.text(f"{z_length.item():.2f}m", z_length_pos.detach().cpu().numpy()),
    ]


def colorize_mask(pred_mask: np.ndarray):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode="RGB")
    return np.array(save_mask)


def colorize_depth(
    depth: np.ndarray,
    normalize: bool = False,
    clip_depth: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 1e4,
):
    if clip_depth:
        depth = np.clip(depth, a_min=min_depth, a_max=max_depth)

    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth


def draw_points_batch(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: np.ndarray | tuple | None = None,
    stencil: np.ndarray | None = None,
):
    if stencil is None:
        stencil = np.array([[0, 0]])

    for rel_pos in stencil:
        pos = (pts + rel_pos[None]).astype(int)
        in_bound = (pos[:, 0] >= 0) & (pos[:, 0] < canvas.shape[1]) & (pos[:, 1] >= 0) & (pos[:, 1] < canvas.shape[0])
        pos = pos[in_bound]
        if isinstance(color, np.ndarray):
            canvas[pos[:, 1], pos[:, 0]] = color[in_bound]
        else:
            canvas[pos[:, 1], pos[:, 0]] = color or (0, 255, 0)
    return canvas


def draw_lines_batch(
    canvas: np.ndarray,
    lines_start: np.ndarray,
    lines_end: np.ndarray,
    color: tuple | None = None,
):
    if lines_start.shape[0] == 0:
        return canvas
    lines = np.stack([lines_start, lines_end], axis=1).astype(int)
    return cv2.polylines(
        canvas.copy(),
        [line for line in lines],
        isClosed=False,
        color=color or (0, 255, 0),
        thickness=1,
    )


def draw_tracks(canvas: np.ndarray, tracks: np.ndarray, valid: np.ndarray):
    """
    Args:
        canvas: The image to draw the tracks on. (H, W, 3) uint8
        tracks: The tracks to draw. (length, n_tracks, 2)
            To draw tracks of different lengths, please call this function multiple times.
        valid: The validity of the tracks. (length, n_tracks)
    """
    for track_idx in range(tracks.shape[0]):
        uv, uv_valid = tracks[track_idx], valid[track_idx]
        canvas = draw_points_batch(canvas, uv[uv_valid], (0, 255 - 20 * track_idx, 0), stencil=POINTS_STENCIL)
    for segment_idx in range(tracks.shape[0] - 1):
        uv_start, start_valid = tracks[segment_idx], valid[segment_idx]
        uv_end, end_valid = tracks[segment_idx + 1], valid[segment_idx + 1]
        all_valid = start_valid & end_valid
        canvas = draw_lines_batch(canvas, uv_start[all_valid], uv_end[all_valid], (0, 255 - 20 * segment_idx, 0))
    return canvas


def project_points_panorama(
    xyz: np.ndarray,
    pose: SE3,
    frame_size: tuple[int, int],
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    uv_tensor = project_points_to_panorama(torch.from_numpy(local_xyz), return_depth=False)
    uv_tensor[:, 0] *= frame_size[1]
    uv_tensor[:, 1] *= frame_size[0]
    uv = (uv_tensor - 0.5).round().int().cpu().numpy()

    if color is not None:
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv, color, stencil=POINTS_STENCIL)


def project_points(
    xyz: np.ndarray,
    intrinsics: np.ndarray,
    camera_type: CameraType,
    pose: SE3,
    frame_size: tuple[int, int],
    subsample_factor: int,
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    camera_model = camera_type.build_camera_model(torch.from_numpy(intrinsics)).scaled(1.0 / subsample_factor)
    xyz_h = torch.cat([torch.from_numpy(local_xyz), torch.ones((local_xyz.shape[0], 1))], dim=1)
    uv, _, _ = camera_model.proj_points(xyz_h)
    in_bound = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < frame_size[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < frame_size[0])
        & torch.from_numpy(local_xyz[:, 2] > 0)
    )
    uv = uv[in_bound]
    uv_np = (uv - 0.5).round().int().cpu().numpy()

    # uv, in_bound = project_points_to_pinhole(
    #     torch.from_numpy(local_xyz),
    #     torch.from_numpy(intrinsics),
    #     frame_size,
    #     return_depth=False,
    # )
    # uv = uv[in_bound]
    # uv[:, 0] *= frame_size[1]
    # uv[:, 1] *= frame_size[0]
    # uv = (uv - 0.5).round().int().cpu().numpy()

    if color is not None:
        color = color[in_bound.cpu().numpy()]
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv_np, color, stencil=POINTS_STENCIL)


def image_above_text(img: np.ndarray, text: str = "<TEXT>") -> Image.Image:
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    width, height = image.size
    text_height = max(20, height // 10)

    new_height = height + int(text_height * 1.5)
    new_image = Image.new("RGB", (width, new_height), color=(255, 255, 255))
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    try:
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont = ImageFont.truetype(
            "arial.ttf", text_height
        )  # You can change the font size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial is not available

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height + int(text_height * 0.2)

    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  # Black text
    return new_image


def save_projection_video(
    video_path: Path,
    video_stream: VideoStream,
    slam_output: SLAMOutput | None,
    subsample_factor: int,
    attributes: list[list[str]],
):
    assert isinstance(video_stream, CachedVideoStream)

    img_h, img_w = video_stream.frame_size()
    img_h //= subsample_factor
    img_w //= subsample_factor

    na_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    text_img = image.text("N/A")
    na_img = image.place_image(
        text_img,
        na_img,
        img_w // 2 - text_img.shape[1] // 2,
        img_h // 2 - text_img.shape[0] // 2,
    )
    na_img = (na_img[..., :3] * 255).astype(np.uint8)

    def get_depth_imgs():
        depth_range = [np.inf, -np.inf]

        # Run first to obtain depth range
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)

            if (depth_data := frame_data.metric_depth) is None:
                continue
            depth_data = depth_data.reciprocal()

            # Remove sky regions if any
            depth_data = depth_data[~frame_data.sky_mask & torch.isfinite(depth_data)]

            depth_min_q, depth_max_q = torch.quantile(depth_data, torch.tensor([0.05, 0.95], device=depth_data.device))
            depth_range[0] = min(depth_range[0], depth_min_q.item())
            depth_range[1] = max(depth_range[1], depth_max_q.item())
        depth_middle = (depth_range[0] + depth_range[1]) / 2
        depth_scale = depth_range[1] - depth_range[0]
        depth_min = depth_middle - depth_scale / 2 * 1.3
        depth_max = depth_middle + depth_scale / 2 * 1.3

        # Then output normalized depth
        for frame_data in video_stream:
            if (depth_data := frame_data.metric_depth) is None:
                yield na_img
                continue

            depth_data = depth_data.reciprocal()
            depth_data[frame_data.sky_mask] = depth_min
            depth_data[~torch.isfinite(depth_data)] = depth_min

            depth_data = depth_data[::subsample_factor, ::subsample_factor]
            depth_img = depth_data.cpu().numpy().astype(float)
            depth_img = (depth_img - depth_min) / (depth_max - depth_min)
            depth_img = np.clip(depth_img, 0, 1)
            yield colorize_depth(depth_img)

    def get_pcd_imgs():
        assert slam_output is not None, "SLAM output is required!"
        slam_map = unpack_optional(slam_output.slam_map)
        pcd_xyz = slam_map.dense_disp_xyz.cpu().numpy()
        pcd_rgb = slam_map.dense_disp_rgb.cpu().numpy()
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)
            rgb_img = frame_data.rgb.cpu().numpy().astype(float)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            rgb_img = cv2.resize(rgb_img, (img_w, img_h))
            intrinsics = unpack_optional(frame_data.intrinsics)
            if torch.sum(intrinsics) < 1e-6:
                pcd_img = project_points_panorama(
                    pcd_xyz,
                    frame_data.pose,
                    frame_size=(img_h, img_w),
                    color=pcd_rgb,
                )
            else:
                pcd_img = project_points(
                    pcd_xyz,
                    frame_data.intrinsics.cpu().numpy(),
                    camera_type=frame_data.camera_type,
                    pose=frame_data.pose,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=pcd_rgb,
                )
            yield cv2.addWeighted(rgb_img, 0.2, pcd_img, 0.8, 0)

    def get_rectified_imgs():
        # Obtain rectification map
        for frame_data in video_stream:
            original_intr = frame_data.camera_type.build_camera_model(frame_data.intrinsics).scaled(
                1 / subsample_factor
            )
            pinhole_intr = original_intr.pinhole()
            device = pinhole_intr.intrinsics.device
            y, x = torch.meshgrid(torch.arange(img_h).float(), torch.arange(img_w).float(), indexing="ij")
            y, x = y.to(device), x.to(device)
            pts, _, _ = pinhole_intr.iproj_disp(torch.ones_like(x), x, y)
            coords, _, _ = original_intr.proj_points(pts)
            coords_norm = 2.0 * coords / torch.tensor([img_w, img_h], device=coords.device) - 1.0
            coords_norm = coords_norm.reshape(1, img_h, img_w, 2)
            break
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)
            img = frame_data.rgb.permute(2, 0, 1).unsqueeze(0)
            img = torch.nn.functional.grid_sample(
                img,
                coords_norm,
                mode="bilinear",
                align_corners=False,
            )[0].float()
            img = img.permute(1, 2, 0).cpu().numpy()
            yield (img * 255).astype(np.uint8)

    def get_rgb_imgs():
        for frame_data in video_stream:
            rgb_img = frame_data.rgb.cpu().numpy().astype(float)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            rgb_img = cv2.resize(rgb_img, (img_w, img_h))
            yield rgb_img

    def get_instance_imgs():
        for frame_data, rgb_img in zip(video_stream, get_rgb_imgs()):
            assert isinstance(frame_data, VideoFrame)
            if frame_data.instance is None:
                yield na_img
                continue
            instance_img = (inst_np := frame_data.instance.cpu().numpy()).astype(float)
            instance_img = colorize_mask(instance_img)

            if frame_data.instance_phrases is not None:
                for instance_id, instance_phrase in frame_data.instance_phrases.items():
                    if instance_id <= 0:
                        continue
                    text_img = image.text(instance_phrase)
                    inst_mask = inst_np == instance_id
                    try:
                        h_min, h_max = np.where(np.any(inst_mask, axis=1))[0][[0, -1]]
                        w_min, w_max = np.where(np.any(inst_mask, axis=0))[0][[0, -1]]
                        instance_img = image.place_image(
                            text_img,
                            instance_img,
                            (w_min + w_max) // 2,
                            (h_min + h_max) // 2,
                        )
                    except IndexError:
                        pass

            if instance_img.dtype == np.float64:
                instance_img = (instance_img[..., :3] * 255).astype(np.uint8)
            instance_img = cv2.resize(instance_img, (img_w, img_h))
            yield cv2.addWeighted(rgb_img, 0.5, instance_img, 0.5, 0)

    def get_empty_imgs():
        for _ in range(len(video_stream)):
            yield na_img

    img_iterators = [
        [
            {
                "rgb": get_rgb_imgs(),
                "depth": get_depth_imgs(),
                "pcd": get_pcd_imgs(),
                "instance": get_instance_imgs(),
                "rectified": get_rectified_imgs(),
                "empty": get_empty_imgs(),
            }[t]
            for t in t_arr
        ]
        for t_arr in attributes
    ]
    with VideoWriter(video_path, video_stream.fps()) as vw:
        trajectory_length = 0.0
        last_pose = unpack_optional(video_stream[0].pose)
        for frame_idx, frame_data in pbar(enumerate(video_stream), total=len(video_stream), desc="Writing viz video"):
            img_rows = []
            for img_iterator in img_iterators:
                img_row = []
                for img in img_iterator:
                    img_row.append(next(img))
                img_rows.append(np.concatenate(img_row, axis=1))
            img_final = np.concatenate(img_rows, axis=0)
            text_desc = f"Frame {frame_idx:03d}"
            # text_desc += f" | BA {slam_output.ba_residual:.4f}"
            if frame_data.intrinsics is not None:
                focal = frame_data.intrinsics[0].item()
                if focal > 1e-6:  # Pano has focal 0
                    fov_y = 2 * np.arctan(frame_data.size()[0] / (2 * focal))
                    fov_y = np.rad2deg(fov_y)
                    text_desc += f" | fovY {fov_y:.2f}"
            current_pose = unpack_optional(frame_data.pose)
            trajectory_length += float(np.linalg.norm((last_pose.inv() * current_pose).translation()[:3].cpu().numpy()))
            last_pose = current_pose
            text_desc += f" | Traj {trajectory_length:.4f}"
            if len(frame_data.information) > 0:
                text_desc += f" | {frame_data.information}"
            img_text = image.text(text_desc)
            img_final = image.place_image(img_text, img_final, 0, 0)
            vw.write(img_final)
