import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.config import validate_typed_config
    from vipe.streams.base import StreamList

    typed_args = validate_typed_config(args)

    # Gather all video streams
    stream_list = StreamList.make(typed_args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})")
        pipeline = make_pipeline(typed_args.pipeline)
        pipeline.run(video_stream)
        logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
