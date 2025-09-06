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

import cv2
import torch

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream


class FrameDirStream(VideoStream):
    """
    A video stream from a directory of frame images.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None) -> None:
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.name

        # Find all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.frame_files = []
        for ext in image_extensions:
            self.frame_files.extend(sorted(path.glob(f'*{ext}')))
            self.frame_files.extend(sorted(path.glob(f'*{ext.upper()}')))
        
        self.frame_files = sorted(list(set(self.frame_files)))
        
        if not self.frame_files:
            raise ValueError(f"No image files found in directory: {path}")

        # Read metadata from first frame
        first_frame = cv2.imread(str(self.frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_files[0]}")
        
        self._height, self._width = first_frame.shape[:2]
        
        # Assume 30 fps for frame directories (this is just for compatibility)
        self._fps = 30.0
        _n_frames = len(self.frame_files)

        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step
        self._fps = self._fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.current_frame_idx = -1
        return self

    def __next__(self) -> VideoFrame:
        self.current_frame_idx += 1
        
        if self.current_frame_idx >= self.end:
            raise StopIteration

        if self.current_frame_idx < self.start:
            return self.__next__()

        if (self.current_frame_idx - self.start) % self.step != 0:
            return self.__next__()

        # Load the frame
        frame_path = self.frame_files[self.current_frame_idx]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        return VideoFrame(raw_frame_idx=self.current_frame_idx, rgb=frame_rgb)


class FrameDirStreamList(StreamList):
    def __init__(self, base_path: str, frame_start: int, frame_end: int, frame_skip: int, cached: bool = False) -> None:
        super().__init__()
        base_path_obj = Path(base_path)
        
        if base_path_obj.is_dir():
            # Single directory of frames
            self.frame_directories = [base_path_obj]
        else:
            # Look for subdirectories that might contain frames
            if base_path_obj.parent.exists():
                self.frame_directories = [d for d in base_path_obj.parent.iterdir() if d.is_dir() and d.name == base_path_obj.name]
            else:
                raise ValueError(f"Directory not found: {base_path}")
                
        if not self.frame_directories:
            raise ValueError(f"No frame directories found at: {base_path}")
            
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached

    def __len__(self) -> int:
        return len(self.frame_directories)

    def __getitem__(self, index: int) -> VideoStream:
        stream: VideoStream = FrameDirStream(self.frame_directories[index], seek_range=self.frame_range)
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return self.frame_directories[index].name

