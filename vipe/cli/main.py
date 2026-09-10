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

import click

from vipe import make_pipeline
from vipe.config import parse_typed_config
from vipe.pipeline.pose_only_long import PoseOnlyLongAnnotationPipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing image frames",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
def infer(video: Path | None, image_dir: Path | None, output: Path, pipeline: str, visualize: bool):
    """Run inference on a video file, a directory of videos, or a directory of images.

    VIDEO may be a single .mp4 file or a directory containing .mp4 files -- every .mp4 in the
    directory is processed in turn, through one pipeline instance (so its models are loaded once).
    """

    logger = configure_logging()

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file/directory or --image-dir", err=True)
        raise click.Abort()

    if video and image_dir:
        click.echo("Error: Cannot provide both video file/directory and --image-dir", err=True)
        raise click.Abort()

    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    # Set up stream configuration based on input type
    video_paths: list[Path] = []
    if image_dir:
        overrides.extend(["streams=frame_dir_stream", f"streams.base_path={image_dir}"])
        input_desc = f"image directory {image_dir}"
    elif video.is_dir():
        video_paths = sorted(video.glob("*.mp4"))
        if not video_paths:
            click.echo(f"Error: no .mp4 files found in directory {video}", err=True)
            raise click.Abort()
        input_desc = f"{len(video_paths)} video(s) in directory {video}"
    else:
        video_paths = [video]
        input_desc = f"video {video}"

    args = parse_typed_config("default", hydra_args=overrides)

    logger.info(f"Processing {input_desc}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    # The long-sequence pipeline reads its input stream in a single bounded pass (see
    # PoseOnlyLongAnnotationPipeline / _StreamingProcessedVideoStream) and does not need --
    # or want -- the whole video materialized in host memory up front. Eagerly `.cache()`-ing
    # here anyway defeats that: it forces full decode-and-buffer of every frame before the
    # pipeline even starts, which is O(sequence length) host memory on exactly the inputs
    # this pipeline exists to handle (it OOM'd a 5080-frame video at ~119GB RSS). Every other
    # pipeline still gets the eager cache, since they rely on it to materialize a correct
    # frame count for malformed videos before iterating.
    is_long_sequence = isinstance(vipe_pipeline, PoseOnlyLongAnnotationPipeline)

    if image_dir:
        # Use frame directory stream
        video_stream = ProcessedVideoStream(FrameDirStream(image_dir), [])
        if not is_long_sequence:
            video_stream = video_stream.cache(desc="Reading image frames")
        vipe_pipeline.run(video_stream)
    else:
        # Process each video with the same pipeline instance, so its cached models
        # (depth, GeoCalib, TrackAnything networks) are only loaded once. Each video writes
        # its own artifacts (named after the video) into the shared output directory.
        for idx, video_path in enumerate(video_paths):
            logger.info(f"Processing {video_path} ({idx + 1} / {len(video_paths)})")
            # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
            video_stream = ProcessedVideoStream(RawMp4Stream(video_path), [])
            if not is_long_sequence:
                video_stream = video_stream.cache(desc="Reading video stream")
            vipe_pipeline.run(video_stream)
            logger.info(f"Finished processing {video_path}")

    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option(package_name="nvidia-vipe")
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
