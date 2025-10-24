#!/usr/bin/env python3
"""
End-to-end test for hybrid metric depth implementation.
This script runs the full ViPE pipeline with the hybrid_metric configuration.
"""

import sys
import logging
from pathlib import Path

# Ensure ViPE modules are discoverable
sys.path.insert(0, str(Path(__file__).parent))

from hydra import compose, initialize_config_dir
from vipe import get_config_path, make_pipeline
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging as vipe_configure_logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("ViPE Hybrid Metric Depth - End-to-End Test")
    logger.info("=" * 80)
    
    # Configure ViPE logging
    vipe_configure_logging()
    
    # Set up paths
    video_path = Path(__file__).parent / "assets/examples/cosmos-example.mp4"
    output_dir = Path(__file__).parent / "vipe_results_hybrid_e2e"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    logger.info(f"Input video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load Hydra config for hybrid_metric pipeline
    config_path = str(get_config_path())
    logger.info(f"Config path: {config_path}")
    
    overrides = [
        "pipeline=hybrid_metric",
        f"pipeline.output.path={output_dir}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=true",
        "pipeline.output.save_slam_map=false",
    ]
    
    logger.info("Loading configuration with overrides:")
    for override in overrides:
        logger.info(f"  - {override}")
    
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="default", overrides=overrides)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Depth model: {cfg.pipeline.slam.keyframe_depth}")
    
    # Create video stream
    logger.info("Creating video stream...")
    video_stream = RawMp4Stream(video_path)
    logger.info("Video stream created successfully")
    
    # Create and run pipeline
    logger.info("Creating pipeline...")
    pipeline = make_pipeline(cfg.pipeline)
    logger.info("Pipeline created successfully")
    
    logger.info("=" * 80)
    logger.info("Running pipeline (this may take a few minutes)...")
    logger.info("=" * 80)
    
    try:
        result = pipeline.run(video_stream)
        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        
        # List generated files
        output_files = list(output_dir.rglob("*"))
        output_files = [f for f in output_files if f.is_file()]
        
        if output_files:
            logger.info(f"\nGenerated {len(output_files)} output files:")
            for file_path in sorted(output_files):
                rel_path = file_path.relative_to(output_dir)
                size = file_path.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
                logger.info(f"  {rel_path} ({size_str})")
        else:
            logger.warning("No output files generated!")
        
        return 0
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ Pipeline failed: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

