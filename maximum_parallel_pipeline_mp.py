#!/usr/bin/env python3
"""
MAXIMUM PARALLELIZATION - MULTIPROCESSING VERSION
GPU 0: Frame processing + SLAM + Alignment (separate process)
GPU 1: GC models + GC inference (separate process)
Uses multiprocessing instead of multithreading for true parallelism!
OPTIMIZATIONS:
- Lazy imports: torch/tqdm imported inside workers to minimize main process overhead
- Forkserver default: faster process spawning while maintaining CUDA safety
STARTUP METHODS (can be controlled via MP_START_METHOD env var):
- 'forkserver' (DEFAULT): ~1-2s startup with lazy imports, safe with CUDA
- 'spawn': ~2-3s startup, safest but slower
- 'fork': ~0.1s startup, fastest but may cause CUDA issues
Usage:
  # Use default (forkserver with lazy imports - recommended)
  python maximum_parallel_pipeline_mp.py video.mp4
  
  # Use fork for fastest startup (risky)
  MP_START_METHOD=fork python maximum_parallel_pipeline_mp.py video.mp4
  
  # Use spawn for safest (slower)
  MP_START_METHOD=spawn python maximum_parallel_pipeline_mp.py video.mp4
"""

import os
import time
import logging
import multiprocessing as mp
from pathlib import Path

# NOTE: torch, pickle, tqdm imported lazily inside workers to avoid ~4s import overhead in main process

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def unpack_optional(val):
    assert val is not None
    return val


def gpu1_worker(frame_data_queue, result_queue):
    """GPU 1 Process: Load GC models, run GC inference, send results to GPU 0"""
    # Lazy imports to avoid overhead in main process
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        # Set up logger for this process
        logger = logging.getLogger(__name__)

        use_flashpack = True

        def load_geometrycrafter_with_flashpack(
            model_class,
            flashpack_class,
            repo_id: str,
            subfolder: str,
            cache_name: str,
            use_flashpack: bool = True,
            **load_kwargs
        ):
            """Helper function to load GeometryCrafter models with FlashPack caching."""
            if not use_flashpack:
                return model_class.from_pretrained(repo_id, subfolder=subfolder, **load_kwargs)

            # Setup flashpack cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vipe_geometrycrafter_flashpack")
            os.makedirs(cache_dir, exist_ok=True)

            save_dir = os.path.join(cache_dir, cache_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "model.flashpack")

            # If flashpack doesn't exist, create it
            if not os.path.exists(model_path):
                print(f"Creating flashpack for {cache_name}...")
                start = time.time()
                # Remove cache_dir from load_kwargs to avoid conflict
                filtered_kwargs = {k: v for k, v in load_kwargs.items() if k != 'cache_dir'}
                initial_model = flashpack_class.from_pretrained(repo_id, subfolder=subfolder, **filtered_kwargs)

                # Save model as flashpack
                initial_model.save_pretrained_flashpack(
                    save_dir,
                    target_dtype=load_kwargs.get('torch_dtype', torch.float16)
                )
                print(f"Flashpack creation took {time.time() - start:.2f}s")
                del initial_model
                torch.cuda.empty_cache()

            # Load from flashpack using the built-in method
            print(f"Loading {cache_name} from flashpack...")
            start = time.time()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = flashpack_class.from_pretrained_flashpack(
                save_dir,
                device=device,
                silent=False
            )

            # Ensure gradient_checkpointing is initialized for UNet (not VAE)
            # Only for UNetSpatioTemporal models which have the custom implementation
            if 'unet' in cache_name.lower() and hasattr(model, 'disable_gradient_checkpointing'):
                try:
                    model.disable_gradient_checkpointing()
                except TypeError:
                    # Some models have different gradient checkpointing interface
                    if not hasattr(model, 'gradient_checkpointing'):
                        model.gradient_checkpointing = False

            print(f"Flashpack loading took {time.time() - start:.2f}s")
            return model

        from vipe.priors.depth.geometrycrafter import (
            GeometryCrafterDiffPipeline,
            PMapAutoencoderKLTemporalDecoder,
            UNetSpatioTemporalConditionModelVid2vid
        )
        from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin
        from vipe.priors.depth.moge import MogeModel

        # FlashPack-enabled GeometryCrafter models
        class FlashPackUNetSpatioTemporal(UNetSpatioTemporalConditionModelVid2vid, FlashPackDiffusersModelMixin):
            """FlashPack-enabled UNet for GeometryCrafter."""
            pass

        class FlashPackPMapVAE(PMapAutoencoderKLTemporalDecoder, FlashPackDiffusersModelMixin):
            """FlashPack-enabled Point Map VAE for GeometryCrafter."""
            pass

        logger.info("[GPU1]: Starting...")

        # ==== GC PHASE ====
        logger.info("[GPU1]: Loading GC models...")
        gc_start = time.time()
        cache_dir = "/home/afridi/Depth/GeometryCrafter/workspace/cache"
        model_type = "diff"

        with torch.cuda.device(1):
            torch.set_default_dtype(torch.float32)  # Ensure float32 default
            load_start = time.time()
            unet = load_geometrycrafter_with_flashpack(
                model_class=UNetSpatioTemporalConditionModelVid2vid,
                flashpack_class=FlashPackUNetSpatioTemporal,
                repo_id='TencentARC/GeometryCrafter',
                subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
                cache_name=f'unet_{model_type}',
                use_flashpack=use_flashpack,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            ).requires_grad_(False).to("cuda:1", dtype=torch.float16)
            logger.info(f"[GPU1]: Time taken to load UNet: {time.time() - load_start} seconds")

            start_time = time.time()
            point_map_vae = load_geometrycrafter_with_flashpack(
                model_class=PMapAutoencoderKLTemporalDecoder,
                flashpack_class=FlashPackPMapVAE,
                repo_id='TencentARC/GeometryCrafter',
                subfolder='point_map_vae',
                cache_name='point_map_vae',
                use_flashpack=True,
                low_cpu_mem_usage=False,
                torch_dtype=torch.float32,
                cache_dir=cache_dir
            ).requires_grad_(False).to("cuda:1", dtype=torch.float32)
            logger.info(f"[GPU1]: Time taken to load Point Map VAE: {time.time() - start_time} seconds")

            start_time = time.time()
            prior_model = MogeModel(cache_dir=cache_dir,
            use_flashpack=True).requires_grad_(False).to('cuda:1', dtype=torch.float32)
            logger.info(f"[GPU1]: Time taken to load MogeModel: {time.time() - start_time} seconds")

            start_time = time.time()
            video_depth_model = GeometryCrafterDiffPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", unet=unet,
                torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir
            ).to("cuda:1")
            logger.info(f"[GPU1]: Time taken to load GeometryCrafterDiffPipeline: {time.time() - start_time} seconds")
            logger.info(f"[GPU1]: Time taken to load entire GC pipeline: {time.time() - load_start} seconds")

            # Wait for frames from queue
            frame_list = []
            frame_data_list = []
            while True:
                item = frame_data_queue.get()
                if item is None:
                    break
                frame_list.append(item[0])
                frame_data_list.append(item[1])

            logger.info(f"GPU 1: Received {len(frame_list)} frames")

            # Run GC inference
            frames_tensor = torch.stack(frame_list).permute(0, 3, 1, 2)
            h, w = frame_data_list[0].rgb.shape[:2]

            inf_start = time.time()
            with torch.inference_mode():
                rec_point_map, _ = video_depth_model(
                    frames_tensor.to("cuda:1"), point_map_vae, prior_model,
                    height=h, width=w, num_inference_steps=5, guidance_scale=1.0,
                    window_size=110, decode_chunk_size=8, overlap=25,
                    force_projection=True, force_fixed_focal=True,
                    use_extract_interp=False, track_time=True, low_memory_usage=False
                )

            rec_depth = 1 / (rec_point_map[:, :, :, 2])
            logger.info(f"GPU 1: GC inference done in {time.time() - inf_start:.2f}s")

        # Send depth tensor frame-by-frame through queue
        rec_depth_cpu = rec_depth.cpu()
        gc_time = time.time() - gc_start

        # Send metadata first
        result_queue.put({
            'type': 'metadata',
            'success': True,
            'shape': list(rec_depth_cpu.shape),
            'dtype': str(rec_depth_cpu.dtype),
            'num_frames': len(rec_depth_cpu),
            'time': gc_time
        })

        # Send depth frame by frame to avoid large tensor serialization issues
        for i, depth_frame in enumerate(rec_depth_cpu):
            result_queue.put({
                'type': 'depth_frame',
                'index': i,
                'data': depth_frame.numpy()  # Convert to numpy for better serialization
            })

        # Send completion signal
        result_queue.put({'type': 'complete'})

        logger.info(f"[GPU1]: Completed in {gc_time:.2f}s, sent {len(rec_depth_cpu)} depth frames to GPU 0")

    except Exception as e:
        logger.error(f"GPU 1 error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({'success': False, 'error': str(e)})


def gpu0_worker(video_path, cfg_dict, frame_data_queue, gc_result_queue):
    """GPU 0 Process: Process frames, run SLAM, load alignment models, run alignment"""
    # Lazy imports to avoid overhead in main process
    import torch
    import pickle
    from tqdm import tqdm as pbar

    try:
        # Set up logger for this process
        logger = logging.getLogger(__name__)

        # Set default dtype to avoid mixed precision issues
        torch.set_default_dtype(torch.float32)
        start_time = time.time()
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from vipe import get_config_path
        from vipe.streams.base import StreamList, ProcessedVideoStream, FrameAttribute, VideoStream, CachedVideoStream
        from vipe.pipeline.processors import GeoCalibIntrinsicsProcessor, TrackAnythingProcessor
        from vipe.utils.cameras import CameraType
        from vipe.utils import io
        logger.info(f"GPU 0: Time taken to imports: {time.time() - start_time} seconds")
        # Initialize Hydra
        initialize_config_dir(config_dir=str(get_config_path()), version_base=None)
        cfg = compose(config_name="default", overrides=["pipeline=default"])

        cfg.streams.base_path = video_path
        stream_list = StreamList.make(cfg.streams)
        video_stream = stream_list[0]

        out_path = Path(cfg.pipeline.output.path)
        out_path.mkdir(exist_ok=True, parents=True)
        artifact_path = io.ArtifactPath(out_path, video_stream.name())

        camera_type = CameraType(cfg.pipeline.init.camera_type)

        # Process frames
        logger.info("GPU 0: Processing frames...")
        init_processors = []
        start_time = time.time()
        use_flashpack = cfg.pipeline.init.get("use_flashpack", True)
        with torch.cuda.device(0):
            init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=camera_type, use_flashpack=use_flashpack))
        logger.info(f"GPU 0: GeoCalibIntrinsicsProcessor loaded in {time.time() - start_time:.2f}s")
        start_time = time.time()
        if cfg.pipeline.init.instance is not None:
            logger.info("[GPU0]: Adding TrackAnything...")
            with torch.cuda.device(0):
                init_processors.append(
                    TrackAnythingProcessor(
                        cfg.pipeline.init.instance.phrases,
                        add_sky=cfg.pipeline.init.instance.add_sky,
                        sam_run_gap=int(video_stream.fps() * cfg.pipeline.init.instance.kf_gap_sec),
                        use_flashpack=True,
                    )
                )
        logger.info(f"GPU 0: TrackAnything loaded in {time.time() - start_time:.2f}s")
        slam_stream = ProcessedVideoStream(video_stream, init_processors).cache("process", online=True)

        # Collect frames, keep local copy, and send to GPU 1
        frame_data_list = []
        for frame in slam_stream:
            frame_cpu = frame.cpu()
            frame_data_list.append(frame_cpu)  # Keep local copy
            frame_data_queue.put((frame.rgb, frame_cpu))
        frame_data_queue.put(None)  # Signal end

        logger.info(f"GPU 0: {len(frame_data_list)} frames processed and sent to GPU 1")

        # Run SLAM
        from vipe.slam.system import SLAMSystem

        logger.info("GPU 0: Starting SLAM...")
        slam_start = time.time()
        with torch.cuda.device(0):
            slam_system = SLAMSystem(device=torch.device("cuda:0"), config=cfg.pipeline.slam)
            slam_output = slam_system.run([slam_stream], camera_type=camera_type)

        slam_time = time.time() - slam_start
        logger.info(f"GPU 0: SLAM completed in {slam_time:.2f}s")

        # ==== REUSE DEPTH MODEL FROM SLAM + LOAD PROMPT MODEL (while GC runs!) ====
        logger.info("GPU 0: Loading alignment models (while GC runs)...")
        from vipe.priors.depth.priorda import PriorDAModel

        from vipe.priors.depth import make_depth_model, DepthEstimationInput
        from vipe.priors.depth.alignment import align_inv_depth_to_depth
        from vipe.utils.visualization import save_projection_video
        model_start = time.time()

        # Reuse the depth model from SLAM instead of loading it again
        if hasattr(slam_system, 'metric_depth') and slam_system.metric_depth is not None:
            depth_model = slam_system.metric_depth
            logger.info("GPU 0: Reusing depth model from SLAM (avoiding duplicate load!)")
        else:
            # Fallback: load if SLAM didn't use one
            model_name = cfg.pipeline.post.depth_align_model
            try:
                prefix, metric_model, _ = model_name.split("_")
            except:
                prefix, metric_model = model_name.split("_")

            with torch.cuda.device(0):
                depth_model = make_depth_model(metric_model, use_flashpack=True)
            logger.info(f"GPU 0: Loaded depth model {metric_model}")

        # Only need to load prompt model
        with torch.cuda.device(0):
            prompt_model = PriorDAModel()

        model_time = time.time() - model_start
        logger.info(f"GPU 0: Alignment models ready in {model_time:.2f}s")

        # ==== WAIT FOR GC DEPTH FROM GPU 1 ====
        logger.info("GPU 0: Waiting for GC depth from GPU 1...")
        wait_start = time.time()

        # Receive metadata
        metadata = gc_result_queue.get()
        if metadata['type'] != 'metadata':
            raise RuntimeError(f"Expected metadata, got {metadata}")
        if not metadata['success']:
            raise RuntimeError(f"GPU 1 failed: {metadata.get('error', 'Unknown error')}")

        gc_time = metadata['time']
        num_frames = metadata['num_frames']

        logger.info(f"GPU 0: Receiving {num_frames} depth frames from GPU 1...")

        # Receive depth frames one by one
        import numpy as np
        depth_frames = []
        for i in range(num_frames):
            frame_msg = gc_result_queue.get()
            if frame_msg['type'] != 'depth_frame':
                raise RuntimeError(f"Expected depth_frame, got {frame_msg['type']}")
            depth_frames.append(frame_msg['data'])

        # Wait for completion signal
        completion = gc_result_queue.get()
        if completion['type'] != 'complete':
            raise RuntimeError(f"Expected complete signal, got {completion['type']}")

        wait_time = time.time() - wait_start
        logger.info(f"GPU 0: Received all depth frames from GPU 1, waited {wait_time:.2f}s")

        # Reconstruct depth tensor and move to GPU 0
        rec_depth_np = np.stack(depth_frames, axis=0)
        rec_depth = torch.from_numpy(rec_depth_np).to("cuda:0")
        logger.info(f"GPU 0: Reconstructed depth tensor with shape {rec_depth.shape}")

        # ==== RUN ALIGNMENT ON GPU 0 ====
        logger.info("GPU 0: Starting alignment...")
        align_start = time.time()

        cache_scale_bias = None
        min_uv_score = 1.0
        update_momentum = 0.99
        view_idx = 0
        infill_target_pose = slam_output.get_view_trajectory(view_idx)

        def _compute_uv_score(depth, patch_count=10):
            h_shape = depth.size(0) // patch_count
            w_shape = depth.size(1) // patch_count
            depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
            depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
            return depth_crop.any(dim=(1, 3)).float().mean().item()

        aligned_frames = []

        for frame_idx, frame in pbar(enumerate(frame_data_list), desc="Aligning on GPU 0"):
            frame = frame.cuda()

            # UV calculation
            if frame_idx == 0:
                slam_intrinsics = slam_output.intrinsics[view_idx]
                for test_idx in range(slam_output.trajectory.shape[0]):
                    if test_idx % 10 != 0:
                        continue
                    depth_infilled = slam_output.slam_map.project_map(
                        test_idx, 0, frame.size(), slam_intrinsics,
                        infill_target_pose[test_idx],
                        unpack_optional(frame.camera_type), infill=False
                    )
                    uv_score = _compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score
                logger.info(f"GPU 0: Minimum UV score: {min_uv_score:.4f}")

            # Prompt depth
            if min_uv_score < 0.3:
                prompt_result = depth_model.estimate(
                    DepthEstimationInput(rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type)
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = slam_output.slam_map.project_map(
                    frame_idx, 0, frame.size(), unpack_optional(frame.intrinsics),
                    infill_target_pose[frame_idx], unpack_optional(frame.camera_type), infill=False
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()

                prompt_result = prompt_model.estimate(
                    DepthEstimationInput(rgb=frame.rgb.float().cuda(), prompt_metric_depth=depth_map)
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            # Align
            video_inv_depth = rec_depth[frame_idx]
            align_mask = video_inv_depth > 1e-3
            if frame.mask is not None:
                align_mask = align_mask & frame.mask & (~frame.sky_mask)

            try:
                _, scale, bias = align_inv_depth_to_depth(video_inv_depth, prompt_result, align_mask)
            except RuntimeError:
                scale, bias = cache_scale_bias if cache_scale_bias else (1.0, 0.0)

            if cache_scale_bias is None:
                cache_scale_bias = (scale, bias)
            scale = cache_scale_bias[0] * update_momentum + scale * (1 - update_momentum)
            bias = cache_scale_bias[1] * update_momentum + bias * (1 - update_momentum)
            cache_scale_bias = (scale, bias)

            video_inv_depth_aligned = video_inv_depth * scale + bias
            video_inv_depth_aligned[video_inv_depth_aligned < 1e-3] = 1e-3
            frame.metric_depth = video_inv_depth_aligned.reciprocal()

            frame.pose = slam_output.get_view_trajectory(view_idx)[frame_idx]
            frame.intrinsics = slam_output.intrinsics[view_idx]

            aligned_frames.append(frame)

        align_time = time.time() - align_start
        logger.info(f"GPU 0: Alignment completed in {align_time:.2f}s")

        # ==== SAVE VISUALIZATION ====
        logger.info("Saving visualization...")
        save_start = time.time()

        class AlignedVideoStream(VideoStream):
            def __init__(self, frames, fps_val, name_val):
                super().__init__()
                self.frames = frames
                self._fps = fps_val
                self._name = name_val

            def __iter__(self):
                return iter(self.frames)

            def __len__(self):
                return len(self.frames)

            def frame_size(self):
                return self.frames[0].size() if self.frames else (576, 1024)

            def fps(self):
                return self._fps

            def name(self):
                return self._name

            def attributes(self):
                return {FrameAttribute.METRIC_DEPTH, FrameAttribute.POSE, FrameAttribute.INTRINSICS, 
                        FrameAttribute.CAMERA_TYPE, FrameAttribute.MASK, FrameAttribute.INSTANCE}

            def cache(self, desc="Caching", online=False):
                vs = CachedVideoStream(self, desc)
                if not online:
                    _ = vs[len(vs) - 1]
                return vs

        output_stream = AlignedVideoStream(
            [f.cuda() for f in aligned_frames],
            video_stream.fps(),
            video_stream.name()
        ).cache("depth", online=True)

        artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)

        # if cfg.pipeline.output.save_viz:
        #     save_projection_video(
        #         artifact_path.meta_vis_path, output_stream, slam_output,
        #         cfg.pipeline.output.viz_downsample, cfg.pipeline.output.viz_attributes
        #     )
        #     logger.info(f"✓ Visualization saved: {artifact_path.meta_vis_path}")
        logger.info(f"Saving artifacts: {cfg.pipeline.output.save_artifacts}")
        if cfg.pipeline.output.save_artifacts:
            io.save_artifacts(artifact_path, output_stream)
            with artifact_path.meta_info_path.open("wb") as f:
                pickle.dump({"ba_residual": slam_output.ba_residual}, f)

        save_time = time.time() - save_start

        # Return results
        result = {
            'success': True,
            'slam_time': slam_time,
            'model_time': model_time,
            'align_time': align_time,
            'gc_time': gc_time,
            'save_time': save_time,
            'artifact_path': str(artifact_path.meta_vis_path)
        }

        GlobalHydra.instance().clear()
        return result

    except Exception as e:
        logger.error(f"GPU 0 error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_maximum_parallel_pipeline(video_path: str, start_method: str = 'forkserver'):
    """Main function that coordinates multiprocessing execution
    
    Args:
        video_path: Path to input video
        start_method: 'spawn' (safest, ~6s), 'forkserver' (balanced, ~2-3s), or 'fork' (fastest, ~0.1s but risky)
    """
    total_start = time.time()

    logger.info("="*60)
    logger.info("MAXIMUM PARALLEL PIPELINE - MULTIPROCESSING VERSION")
    logger.info("="*60)
    logger.info(f"Video: {video_path}")

    # Create multiprocessing queues
    # STARTUP METHOD OPTIONS (trade-off: speed vs safety):
    # - 'forkserver': Balanced ~1-2s with lazy imports, safe with CUDA (DEFAULT)
    # - 'spawn': Safest ~2-3s, but slower
    # - 'fork': Fastest ~0.1s, but can cause CUDA deadlocks (use at your own risk)

    # Allow override via environment variable
    start_method = os.environ.get('MP_START_METHOD', start_method)

    try:
        mp.set_start_method(start_method, force=True)
        overhead = {'spawn': '~2-3s', 'forkserver': '~1-2s', 'fork': '~0.1s'}.get(start_method, '?')
        logger.info(f"Using '{start_method}' method (startup overhead: {overhead})")
    except Exception as e:
        logger.warning(f"Could not set '{start_method}' method: {e}, falling back to spawn")
        mp.set_start_method('spawn', force=True)
        logger.info("Using 'spawn' method (safest but ~6s startup overhead)")

    # Unbounded queue since GPU1 consumes all frames after model loading
    # (GPU0 would block at 100 frames otherwise)
    frame_data_queue = mp.Queue()
    gc_result_queue = mp.Queue()

    # Start GPU 1 process
    gpu1_process = mp.Process(
        target=gpu1_worker,
        args=(frame_data_queue, gc_result_queue),
        name="GPU1-Worker"
    )

    # Start GPU 0 process
    gpu0_process = mp.Process(
        target=gpu0_worker,
        args=(video_path, {}, frame_data_queue, gc_result_queue),
        name="GPU0-Worker"
    )

    logger.info("🚀 Starting GPU 0 and GPU 1 processes...")
    spawn_start = time.time()

    gpu0_process.start()
    gpu1_process.start()

    logger.info(f"   Processes created in {time.time() - spawn_start:.2f}s, initializing...")

    # Wait for both processes to complete
    gpu0_process.join()
    gpu1_process.join()

    # Check exit codes
    if gpu0_process.exitcode != 0:
        logger.error(f"GPU 0 process failed with exit code {gpu0_process.exitcode}")
        return False

    if gpu1_process.exitcode != 0:
        logger.error(f"GPU 1 process failed with exit code {gpu1_process.exitcode}")
        return False

    total_time = time.time() - total_start

    # Note: In multiprocessing, we can't easily get the return value from gpu0_worker
    # The results are logged within each process
    logger.info("="*60)
    logger.info("MULTIPROCESSING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"TOTAL TIME: {total_time:.2f}s")
    logger.info("="*60)

    return True


def main():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    args = parser.parse_args()
    success = run_maximum_parallel_pipeline(args.video_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
