import argparse
import itertools
import shutil
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run a TSDF hyperparameter ablation study using ViPE.")
    parser.add_argument("video_path", type=str, help="Path to the input video file (e.g., input.mp4)")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Directory to store the resulting .ply files")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem

    # Define the parameter grid to test
    grid = {
        "voxel_length": [0.005, 0.01],
        "pruning_threshold": [None, 0.05, 0.10, 0.20], # None means apply_edge_pruning=False
        "dynamic_mask": [True, False]
    }

    keys = grid.keys()
    combinations = list(itertools.product(*grid.values()))
    
    print(f"Starting ablation study for {video_name}.")
    print(f"Total combinations to test: {len(combinations)}
")

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        # Construct the unique identifier for this run
        voxel_str = f"Voxel{params['voxel_length']}"
        prune_str = "NoPrune" if params['pruning_threshold'] is None else f"Prune{params['pruning_threshold']}"
        mask_str = "MaskOn" if params['dynamic_mask'] else "MaskOff"
        run_id = f"{voxel_str}_{prune_str}_{mask_str}"
        
        # Temporary output directory for this specific vipe run
        run_out_dir = output_dir / f"tmp_{run_id}"
        
        print(f"[{idx}/{len(combinations)}] Running: {run_id}")
        
        # Build Hydra overrides
        cmd = [
            "vipe",
            "--video_path", str(video_path),
            "pipeline.output.save_tsdf_ply=true",
            f"pipeline.output.path={run_out_dir}",
            f"pipeline.output.tsdf_voxel_length={params['voxel_length']}",
            # Rule of thumb: sdf_trunc should be ~4x voxel_length
            f"pipeline.output.tsdf_sdf_trunc={params['voxel_length'] * 4.0}",
            f"pipeline.output.tsdf_apply_dynamic_mask={str(params['dynamic_mask']).lower()}"
        ]
        
        if params['pruning_threshold'] is None:
            cmd.append("pipeline.output.tsdf_apply_edge_pruning=false")
        else:
            cmd.append("pipeline.output.tsdf_apply_edge_pruning=true")
            cmd.append(f"pipeline.output.tsdf_pruning_threshold={params['pruning_threshold']}")

        # Execute ViPE
        try:
            subprocess.run(cmd, check=True)
            
            # The resulting TSDF ply will be deeply nested in the run_out_dir:
            # run_out_dir / "vipe" / f"{video_name}_tsdf.ply" (based on default ArtifactPath)
            expected_ply = run_out_dir / video_name / "vipe" / f"{video_name}_tsdf.ply"
            
            # ViPE artifact paths often have a subfolder with the artifact_name (which is video_name)
            if not expected_ply.exists():
                 # Fallback search if artifact path structure differs
                 found_plys = list(run_out_dir.rglob("*_tsdf.ply"))
                 if found_plys:
                     expected_ply = found_plys[0]
            
            if expected_ply.exists():
                final_ply_path = output_dir / f"{video_name}_{run_id}.ply"
                shutil.copy(expected_ply, final_ply_path)
                print(f"  -> Saved: {final_ply_path.name}")
            else:
                print(f"  -> Warning: Could not find resulting .ply file for {run_id}")
                
        except subprocess.CalledProcessError as e:
            print(f"  -> Error executing ViPE for {run_id}: {e}")
        finally:
            # Cleanup the heavy temporary ViPE artifacts to save disk space
            if run_out_dir.exists():
                shutil.rmtree(run_out_dir)

    print("
✅ Ablation study complete!")
    print(f"All generated .ply files are available in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
