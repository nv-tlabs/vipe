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
import re

import cv2
import torch
from PIL import Image
import numpy as np

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream


class RawImageSequenceStream(VideoStream):
    """
    A video stream from a sequence of images in a folder.
    Supports common image formats: jpg, jpeg, png, bmp, tiff.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None, fps: float = 30.0) -> None:
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.name
        self._fps = fps

        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files in the directory
        self.image_files = []
        if path.is_dir():
            for ext in self.supported_extensions:
                self.image_files.extend(path.glob(f'*{ext}'))
                self.image_files.extend(path.glob(f'*{ext.upper()}'))
        else:
            raise ValueError(f"Path {path} is not a directory")
        
        if not self.image_files:
            raise ValueError(f"No image files found in {path}")
        
        # Sort files naturally (handle numeric sequences like img_001.jpg, img_002.jpg)
        self.image_files = self._natural_sort(self.image_files)
        
        # Read first image to get dimensions
        first_image = Image.open(self.image_files[0])
        self._width = first_image.width
        self._height = first_image.height
        first_image.close()
        
        _n_frames = len(self.image_files)
        
        self.start = seek_range.start if seek_range.start >= 0 else 0
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step if seek_range.step > 0 else 1
        
        # Adjust fps based on step
        self._fps = fps / self.step

    def _natural_sort(self, file_list):
        """Sort files naturally, handling numeric sequences properly."""
        def natural_key(path):
            # Extract numbers from filename for proper sorting
            parts = re.split(r'(\d+)', path.stem)
            return [int(part) if part.isdigit() else part for part in parts]
        
        return sorted(file_list, key=natural_key)

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> VideoFrame:
        if self.current_idx >= len(self):
            raise StopIteration
        
        # Calculate actual frame index
        actual_frame_idx = self.start + self.current_idx * self.step
        
        if actual_frame_idx >= len(self.image_files):
            raise StopIteration
        
        # Load image
        image_path = self.image_files[actual_frame_idx]
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        frame_rgb = torch.as_tensor(np.array(image)).float() / 255.0
        frame_rgb = frame_rgb.cuda()
        
        image.close()
        
        self.current_idx += 1
        
        return VideoFrame(raw_frame_idx=actual_frame_idx, rgb=frame_rgb)


class RawImageSequenceStreamList(StreamList):
    def __init__(self, base_path: str, frame_start: int, frame_end: int, frame_skip: int, fps: float = 30.0, cached: bool = False) -> None:
        super().__init__()
        base_path = Path(base_path)
        
        if base_path.is_dir():
            # Check if it's a single image sequence directory
            supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            for ext in supported_extensions:
                image_files.extend(base_path.glob(f'*{ext}'))
                image_files.extend(base_path.glob(f'*{ext.upper()}'))
            
            if image_files:
                # Single image sequence
                self.image_sequences = [base_path]
            else:
                # Multiple subdirectories, each containing an image sequence
                self.image_sequences = []
                for subdir in base_path.iterdir():
                    if subdir.is_dir():
                        subdir_images = []
                        for ext in supported_extensions:
                            subdir_images.extend(subdir.glob(f'*{ext}'))
                            subdir_images.extend(subdir.glob(f'*{ext.upper()}'))
                        if subdir_images:
                            self.image_sequences.append(subdir)
                
                if not self.image_sequences:
                    raise ValueError(f"No image sequences found in {base_path}")
        else:
            raise ValueError(f"Path {base_path} is not a directory")
        
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.fps = fps
        self.cached = cached

    def __len__(self) -> int:
        return len(self.image_sequences)

    def __getitem__(self, index: int) -> VideoStream:
        stream: VideoStream = RawImageSequenceStream(
            self.image_sequences[index], 
            seek_range=self.frame_range,
            fps=self.fps
        )
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading image sequence", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return self.image_sequences[index].name