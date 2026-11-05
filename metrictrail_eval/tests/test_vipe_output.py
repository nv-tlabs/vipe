"""``--pred-format vipe``: a synthetic ViPE output directory gives the arrays of the
plain prediction path (``--pred-format npz``) and therefore the same scores.

Every test builds one sequence twice from the same arrays: as a plain
prediction (``<case>.npz`` + frontend depth cache) and as a ViPE
output directory (``pose/``, ``intrinsics/``, ``depth/<name>.zip``, run record).
The scorer runs on both and every per-sequence value must be equal.  Depth maps
are float16-representable, since ViPE stores depth as HALF.

Ground truth is synthetic; datasets whose preparation is attested by profile
digests or fixed frame counts get those constants patched to the synthetic
values.  Run from the repository root: ``python -m pytest tests``.
"""

from __future__ import annotations

import os

# Single-threaded Open3D ICP is reproducible, so point metrics can be compared exactly.  Forced (not a
# default): with a multi-threaded OMP_NUM_THREADS from the shell (e.g. 16, as for --compare-tol 0) the two
# ICP runs of one test differ in the last bits.  Must be set before Open3D loads its OpenMP runtime.
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from metrictrail_eval import (  # noqa: E402
    lamaria,
    oxford,
    points,
    score,
    sevenscenes,
    tum,
    vbr,
    vipe_output,
    wild_slam,
)
from metrictrail_eval.common import sha256_file, text_sha256  # noqa: E402

PROFILE = "synthetic-test-profile"
HAS_EVO = importlib.util.find_spec("evo") is not None
HAS_POINTS = all(importlib.util.find_spec(name) is not None for name in ("open3d", "cv2", "OpenEXR", "Imath"))
HAS_EXR = all(importlib.util.find_spec(name) is not None for name in ("OpenEXR", "Imath"))
needs_evo = pytest.mark.skipif(not HAS_EVO, reason="evo not installed")
needs_points = pytest.mark.skipif(not HAS_POINTS, reason="open3d / OpenCV / OpenEXR not installed")
needs_exr = pytest.mark.skipif(not HAS_EXR, reason="OpenEXR not installed")


# --------------------------------------------------------------------------- synthetic data


def trajectory(n: int, seed: int, radius: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, :3, 3] = np.stack([radius * np.cos(t), radius * np.sin(t), 0.3 * t], axis=1)
    poses[:, :3, 3] += rng.normal(scale=0.02, size=(n, 3))
    poses[:, :3, :3] = Rotation.from_euler("zyx", np.stack([t, 0.1 * np.sin(t), 0.05 * t], 1)).as_matrix()
    return poses


def estimate(gt: np.ndarray, seed: int, scale: float = 0.8) -> np.ndarray:
    """A Sim(3) copy of ``gt`` with centre and rotation noise (finite, proper rotations)."""

    rng = np.random.default_rng(seed)
    gt = np.where(np.isfinite(gt), gt, np.eye(4)[None])
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    out = np.repeat(np.eye(4)[None], len(gt), axis=0)
    noise = Rotation.from_rotvec(rng.normal(scale=0.01, size=(len(gt), 3))).as_matrix()
    out[:, :3, :3] = np.einsum("ij,njk,nkl->nil", rotation, gt[:, :3, :3], noise)
    out[:, :3, 3] = scale * gt[:, :3, 3] @ rotation.T + [1.0, -2.0, 0.5] + rng.normal(scale=0.05, size=(len(gt), 3))
    return out


def half(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float16).astype(np.float32)


def write_depth_zip(path: Path, depth_maps: np.ndarray, inds: np.ndarray | None = None) -> None:
    """As ``vipe.utils.io.save_depth_artifacts`` writes it: ``{frame:05d}.exr``, HALF channel ``Z``."""

    import Imath
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    inds = np.arange(len(depth_maps)) if inds is None else inds
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for frame_idx, metric_depth in zip(inds.tolist(), depth_maps):
            height, width = metric_depth.shape
            header = OpenEXR.Header(width, height)
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
            with tempfile.NamedTemporaryFile(suffix=".exr") as handle:
                exr = OpenEXR.OutputFile(handle.name, header)
                exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
                exr.close()
                archive.write(handle.name, f"{frame_idx:05d}.exr")


def write_vipe_output(
    root: Path,
    name: str,
    c2w: np.ndarray,
    *,
    inds: np.ndarray | None = None,
    k4: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    input_hw: tuple[int, int] | None = None,
    record: dict | None = None,
    pose_dir: str = "pose",
) -> None:
    """A ``pipeline=metrictrail`` output: pose / intrinsics as its ``_save_pose_artifacts``, depth, run record."""

    count = len(c2w)
    inds = np.arange(count) if inds is None else np.asarray(inds)
    (root / pose_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        root / pose_dir / f"{name}.npz",
        data=np.asarray(c2w, dtype=np.float64),
        inds=inds,
        keyframe_ids=inds[::4],
        runtime=np.float64(1.5),
        source_frame_ids=inds * 3 + 7,  # stream raw indices: not dataset frame ids, never mapped
    )
    if k4 is not None:
        (root / "intrinsics").mkdir(parents=True, exist_ok=True)
        np.savez(
            root / "intrinsics" / f"{name}.npz",
            data=np.repeat(np.asarray(k4, dtype=np.float32)[None], count, axis=0),
            inds=inds,
        )
        (root / "intrinsics" / f"{name}_camera.txt").write_text("".join(f"{i}: PINHOLE\n" for i in inds.tolist()))
    if depth is not None:
        write_depth_zip(root / "depth" / f"{name}.zip", depth, inds)
    if record is None:
        hw = list(input_hw) if input_hw is not None else None
        record = {
            "schema": "metrictrail_vipe_run_v1",
            "status": "success",
            "name": name,
            "tracking_status": "tracked",
            "input_raster": {"policy": "unchanged", "source_hw": hw, "prepared_hw": hw, "frames": count,
                             "frame_stride": 1},
            "tracking": {"num_frames": count, "input_hw": hw},
        }
    (root / "vipe").mkdir(parents=True, exist_ok=True)
    (root / "vipe" / f"{name}_metrictrail.json").write_text(json.dumps(record))


def write_npz_prediction(
    pred: Path,
    case: str,
    c2w: np.ndarray,
    *,
    frame_ids: np.ndarray | None = None,
    k4: np.ndarray | None = None,
    input_hw: tuple[int, int] | None = None,
    extra: dict | None = None,
) -> Path:
    """A plain prediction file ``<case>.npz`` (sequence, frame ids, poses, optional intrinsics and raster)."""

    count = len(c2w)
    fields = {
        "sequence": np.asarray(case),
        "frame_ids": np.arange(count, dtype=np.int32) if frame_ids is None else np.asarray(frame_ids, np.int32),
        "c2w": np.asarray(c2w, dtype=np.float64),
        "w2c": np.linalg.inv(c2w),
        "pose_camera_frame": np.asarray("cam0"),
        **(extra or {}),
    }
    if k4 is not None:
        fields["intrinsics"] = np.repeat(np.asarray(k4, dtype=np.float32)[None], count, axis=0)
    if input_hw is not None:
        fields["input_hw"] = np.asarray(input_hw, dtype=np.int32)
    pred.mkdir(parents=True, exist_ok=True)
    path = pred / f"{case}.npz"
    np.savez_compressed(path, **fields)
    return path


def write_cache(root: Path, depth: np.ndarray, mask: np.ndarray) -> None:
    """A frontend depth cache: raw float32 depth + depth-model mask, contract, SUCCESS."""

    root.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(depth, dtype=np.float32).tofile(root / "depth.f32")
    np.ascontiguousarray(mask, dtype=np.uint8).tofile(root / "mask.u8")
    count, height, width = depth.shape
    (root / "contract.json").write_text(json.dumps({"count": count, "input_hw": [height, width]}))
    (root / "SUCCESS").write_text(json.dumps({"status": "success", "completed_frames": count}))


def vipe_depth(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """What ``pipeline=metrictrail`` saves: the model depth where its mask is valid, else 0."""

    return np.where(mask, raw, np.float32(0.0)).astype(np.float32)


def run_score(tmp_path: Path, tag: str, argv: list[str]) -> dict:
    out = tmp_path / "scores" / f"{tag}.json"
    assert score.main([*argv, "--out", str(out)]) == 0
    return json.loads(out.read_text())


def assert_same_scores(npz: dict, vipe: dict, *, vipe_root: Path, name: str, stage_dir: str = "pose") -> None:
    assert vipe["pred_format"] == "vipe" and npz["pred_format"] == "npz"
    for key in ("dataset", "stage", "metric", "protocol", "units", "gt_root", "cases"):
        assert npz[key] == vipe[key], key
    (run_npz,), (run_vipe,) = npz["runs"], vipe["runs"]  # one --pred, one run
    assert run_vipe["layout"] == "vipe_output"
    assert run_npz["per_case"].keys() == run_vipe["per_case"].keys()
    assert run_npz["missing"] == run_vipe["missing"]
    for case, expected in run_npz["per_case"].items():
        got = run_vipe["per_case"][case]
        assert got["prediction"] == str(vipe_root.resolve() / stage_dir / f"{name}.npz")
        assert got["prediction_format"] == "vipe" and got["vipe"]["name"] == name
        for key, value in expected.items():
            if key == "prediction":
                continue
            if key == "depth_source":
                assert got[key] == str(vipe_root.resolve() / "depth" / f"{name}.zip")
                continue
            assert got[key] == value, (case, key, got[key], value)
    for key in ("mean_ate", "macro", "num_cases"):
        assert run_npz.get(key) == run_vipe.get(key), key


def sha_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2))
    return sha256_file(path)


# --------------------------------------------------------------------------- synthetic GT per dataset


def gt_kitti(root: Path, case: str, n: int) -> np.ndarray:
    gt = trajectory(n, 1, radius=30.0)
    root.mkdir(parents=True)
    np.savez(root / f"{case}.npz", c2w_gt=gt)
    return gt


def gt_lamaria(root: Path, case: str, n: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    monkeypatch.setitem(lamaria.LAMARIA_COUNTS, case, n)
    base = root / "training" / case
    stamps = 1_000_000_000 + np.arange(n, dtype=np.int64) * 50_000_000
    (base / "asl_folder" / case / "aria" / "cam0").mkdir(parents=True)
    (base / "asl_folder" / case / "aria" / "cam0" / "data.csv").write_text(
        "".join(f"{t},{t}.png\n" for t in stamps.tolist())
    )
    body = trajectory(n, 2)
    quats = Rotation.from_matrix(body[:, :3, :3]).as_quat()
    (base / "ground_truth" / "pGT").mkdir(parents=True)
    np.savetxt(base / "ground_truth" / "pGT" / f"{case}.txt",
               np.column_stack([stamps.astype(np.float64), body[:, :3, 3], quats]), fmt="%.17g")
    (base / "pinhole_calibrations").mkdir(parents=True)
    (base / "pinhole_calibrations" / f"{case}.json").write_text(json.dumps(
        {"cam0": {"T_b_s": {"tvec": [0.01, -0.02, 0.03], "qvec": [0.0, 0.0, 0.0998, 0.995]}}}
    ))
    return lamaria.load_gt(root, case)[0]


def gt_7scenes_poses(root: Path, case: str) -> np.ndarray:
    ids = sevenscenes.source_ids(case)
    sequence = root / sevenscenes.SOURCE_BY_KEY[case]
    sequence.mkdir(parents=True)
    gt = trajectory(len(ids), 3, radius=1.0)
    for frame_id, pose in zip(ids.tolist(), gt):
        np.savetxt(sequence / f"frame-{frame_id:06d}.pose.txt", pose, fmt="%.9f")
    return sevenscenes.load_gt(root, case, ids)[0].astype(np.float64)


def gt_tum(root: Path, case: str, n: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    hw = (388, 518)
    monkeypatch.setattr(tum, "PROFILE_SHA256", {hw: text_sha256(PROFILE)})
    scene = root / case
    scene.mkdir(parents=True)
    gt = trajectory(n, 4, radius=1.5)
    gt_valid = np.ones(n, dtype=bool)
    gt_valid[[3, 17]] = False
    gt[~gt_valid] = np.nan
    frame_ids = np.arange(n, dtype=np.int32)
    np.savez(scene / "sequence.npz", frame_ids=frame_ids, c2w_gt=gt.astype(np.float32), gt_valid=gt_valid,
             profile=np.asarray(PROFILE), scene=np.asarray(case), input_hw=np.asarray(hw, dtype=np.int32))
    manifest = {"status": "success", "profile": PROFILE, "scene": case, "frame_count": n,
                "output": {"image_inventory_sha256": "0" * 64}}
    manifest_sha = sha_json(scene / "manifest.json", manifest)
    contract = {
        "schema_version": 1, "dataset": "TUM-RGBD-Freiburg1-9", "profile": PROFILE, "scene": case,
        "input_hw": list(hw), "frame_count": n, "prepared_manifest_sha256": manifest_sha,
        "sequence_npz_sha256": sha256_file(scene / "sequence.npz"),
        "output_image_inventory_sha256": "0" * 64,
        "frame_ids_sha256_i64le": hashlib.sha256(frame_ids.astype("<i8").tobytes()).hexdigest(),
    }
    contract_sha = sha_json(scene / tum.INPUT_CONTRACT_FILENAME, contract)
    sha_json(scene / ".success", {"manifest_sha256": manifest_sha,
                                  "metadata_sha256": sha256_file(scene / "sequence.npz"),
                                  "input_contract_sha256": contract_sha})
    return tum.load_gt(root, case)["c2w_gt"]


def gt_vbr(root: Path, case: str, n: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    monkeypatch.setattr(vbr, "PROFILE_SHA256", text_sha256(PROFILE))
    monkeypatch.setitem(vbr.EXPECTED_SEQUENCE_FRAMES, case, n)
    sequence = root / case
    sequence.mkdir(parents=True)
    gt = trajectory(n, 5, radius=20.0)
    np.savez(sequence / "sequence.npz", frame_ids=np.arange(n, dtype=np.int64), c2w_gt=gt,
             sequence=np.asarray(case), profile=np.asarray(PROFILE),
             input_hw=np.asarray(vbr.TARGET_HW, dtype=np.int32))
    (sequence / "intrinsics.yaml").write_text("fx: 300\n")
    manifest_sha = sha_json(sequence / "manifest.json", {
        "schema_version": 1, "status": "success", "dataset": "vbr", "sequence": case, "profile": PROFILE,
        "frame_count": n, "intrinsics_yaml": {"sha256": sha256_file(sequence / "intrinsics.yaml")},
    })
    sha_json(sequence / ".success", {"status": "success", "sequence": case, "manifest_sha256": manifest_sha,
                                     "metadata_sha256": sha256_file(sequence / "sequence.npz")})
    return gt


def gt_wild_slam(root: Path, case: str, n: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    monkeypatch.setattr(wild_slam, "PROFILE_SHA256", text_sha256(PROFILE))
    sequence = root / case
    sequence.mkdir(parents=True)
    gt = trajectory(n, 6, radius=2.0)
    gt_valid = np.ones(n, dtype=bool)
    gt_valid[[0, 9, 10]] = False
    np.savez(sequence / "sequence.npz", frame_ids=np.arange(n, dtype=np.int32), c2w_gt=gt, gt_valid=gt_valid)
    sha_json(sequence / "manifest.json", {"status": "success", "profile": PROFILE, "sequence": case, "num_frames": n})
    (sequence / wild_slam.SUCCESS_MARKER).write_text("ok\n")
    return np.where(gt_valid[:, None, None], gt, np.nan)


def gt_oxford(root: Path, case: str, n: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    counts = {scene: n for scene in oxford.CASES}
    monkeypatch.setattr(oxford, "PROFILE_SHA256", text_sha256(PROFILE))
    monkeypatch.setattr(oxford, "EXPECTED_SCENE_FRAMES", counts)
    monkeypatch.setattr(oxford, "EXPECTED_TOTAL_FRAMES", sum(counts.values()))
    root.mkdir(parents=True)
    manifest_sha = sha_json(root / "manifest.json", {
        "status": "success", "dataset": oxford.DATASET, "profile": PROFILE, "scenes": list(oxford.CASES),
        "input_hw": list(oxford.TARGET_HW), "total_input_frames": sum(counts.values()),
    })
    sha_json(root / "prepare.SUCCESS.json", {
        "status": "success", "profile": PROFILE, "scenes": list(oxford.CASES),
        "total_input_frames": sum(counts.values()), "manifest_sha256": manifest_sha,
    })
    scene = root / case
    scene.mkdir()
    gt = trajectory(n, 7, radius=10.0)
    ids = np.arange(n, dtype=np.int32)
    stamps = 1.7e9 + np.arange(n) * 0.1
    np.savez(scene / "sequence.npz", frame_ids=ids, c2w_gt=gt, source_timestamps=stamps,
             source_processed_ordinals=ids, input_hw=np.asarray(oxford.TARGET_HW, dtype=np.int32),
             scene=np.asarray(case), profile=np.asarray(PROFILE))
    np.savez(scene / "rgb_timing.npz", frame_ids=ids, source_timestamps=stamps, source_processed_ordinals=ids)
    (scene / "intrinsics.yaml").write_text("fx: 300\n")
    hashes = {
        "manifest_sha256": sha_json(scene / "manifest.json", {"scene": case}),
        "sequence_npz_sha256": sha256_file(scene / "sequence.npz"),
        "rgb_timing_npz_sha256": sha256_file(scene / "rgb_timing.npz"),
    }
    contract_sha = sha_json(scene / oxford.INPUT_CONTRACT_FILENAME, {
        "profile": PROFILE, "scene": case, "frame_count": n, "input_hw": list(oxford.TARGET_HW),
        "frame_ids": ids.tolist(), "intrinsics_yaml_sha256": sha256_file(scene / "intrinsics.yaml"), **hashes,
    })
    sha_json(scene / "prepare.SUCCESS.json", {
        "status": "success", "profile": PROFILE, "scene": case, "frame_count": n,
        "input_contract_sha256": contract_sha, **hashes,
    })
    return gt


# --------------------------------------------------------------------------- trajectory


TRAJECTORY_CASES = {
    # dataset: (case, frames, GT writer, needs evo)
    "kitti": ("04", 120, lambda root, case, n, mp: gt_kitti(root, case, n), True),
    "lamaria": ("R_01_easy", 90, gt_lamaria, False),
    "7scenes": ("stairs__seq-01", 100, lambda root, case, n, mp: gt_7scenes_poses(root, case), True),
    "tum": ("desk", 80, gt_tum, True),
    "vbr": ("campus_train0", 150, gt_vbr, True),
    "wild_slam": ("ANYmal1", 70, gt_wild_slam, True),
    "oxford": ("keble-college-02", 60, gt_oxford, True),
}


@pytest.mark.parametrize("dataset", sorted(TRAJECTORY_CASES))
def test_trajectory_scores_equal(dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, n, write_gt, evo = TRAJECTORY_CASES[dataset]
    if evo and not HAS_EVO:
        pytest.skip("evo not installed")
    gt_root = tmp_path / "gt"
    gt = write_gt(gt_root, case, n, monkeypatch)
    pred = estimate(gt, seed=11)
    hw = {"oxford": oxford.TARGET_HW, "vbr": (280, 504)}.get(dataset, (240, 320))
    k4 = np.asarray([250.5, 251.25, hw[1] / 2, hw[0] / 2])
    write_npz_prediction(tmp_path / "npz", case, pred, k4=k4, input_hw=hw)
    write_vipe_output(tmp_path / "vipe", case, pred, k4=k4, input_hw=hw)
    common = ["--dataset", dataset, "--gt-root", str(gt_root), "--cases", case]
    npz = run_score(tmp_path, "npz", [*common, "--pred", str(tmp_path / "npz")])
    got = run_score(tmp_path, "vipe", [*common, "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"])
    assert_same_scores(npz, got, vipe_root=tmp_path / "vipe", name=case)
    assert got["runs"][0]["per_case"][case]["vipe"]["input_hw"] == list(hw)


@needs_evo
def test_sparse_inds_map_to_frame_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """VBR scores an exact ordinal subset; ViPE ``inds`` become ``frame_ids``."""

    case, n = "campus_train0", 150
    gt = gt_vbr(tmp_path / "gt", case, n, monkeypatch)
    keep = np.setdiff1d(np.arange(n), [5, 6, 40, 41, 42, 149])
    pred = estimate(gt, seed=12)[keep]
    write_npz_prediction(tmp_path / "npz", case, pred, frame_ids=keep)
    write_vipe_output(tmp_path / "vipe", case, pred, inds=keep, record={"status": "success"})
    common = ["--dataset", "vbr", "--gt-root", str(tmp_path / "gt"), "--cases", case]
    npz = run_score(tmp_path, "npz", [*common, "--pred", str(tmp_path / "npz")])
    got = run_score(tmp_path, "vipe", [*common, "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"])
    assert_same_scores(npz, got, vipe_root=tmp_path / "vipe", name=case)
    assert got["runs"][0]["per_case"][case]["coverage"] == "sparse_exact_ordinal_subset"


@needs_evo
def test_lc_stage_and_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--stage LC`` reads ``pose_lc/``; ``--vipe-name`` maps artifact names; ``--pred`` takes one run."""

    case, n = "desk", 80
    gt = gt_tum(tmp_path / "gt", case, n, monkeypatch)
    local, closed = estimate(gt, seed=21), estimate(gt, seed=31, scale=1.2)
    run = tmp_path / "npz"
    write_npz_prediction(run / "local" / "inference" / "predictions", case, local)
    write_npz_prediction(run / "LC" / "inference", case, closed)
    vipe = tmp_path / "vipe"
    write_vipe_output(vipe, "tum_desk_run", local, record={"status": "success"})
    write_vipe_output(vipe, "tum_desk_run", closed, pose_dir="pose_lc", record={"status": "success"})
    ates = {}
    for stage, stage_dir in (("local", "pose"), ("LC", "pose_lc")):
        common = ["--dataset", "tum", "--gt-root", str(tmp_path / "gt"), "--cases", case, "--stage", stage]
        npz = run_score(tmp_path, f"npz_{stage}", [*common, "--pred", str(run)])
        got = run_score(tmp_path, f"vipe_{stage}", [*common, "--pred", str(vipe), "--pred-format", "vipe",
                                                     "--vipe-name", f"{case}=tum_desk_run"])
        assert_same_scores(npz, got, vipe_root=vipe, name="tum_desk_run", stage_dir=stage_dir)
        ates[stage] = got["runs"][0]["per_case"][case]["ate"]
    assert ates["local"] != ates["LC"]
    # Exactly one --pred: a second directory, or --pred given twice, is an argparse error.
    for extra in ([str(run)], ["--pred", str(run)]):
        with pytest.raises(SystemExit) as error:
            score.main(["--dataset", "tum", "--gt-root", str(tmp_path / "gt"), "--pred", str(run), *extra])
        assert error.value.code == 2


# --------------------------------------------------------------------------- arrays


@needs_exr
def test_arrays_round_trip(tmp_path: Path) -> None:
    """Pose, intrinsics, raster and the scorer-visible depth equal the plain prediction's arrays."""

    rng = np.random.default_rng(0)
    n, hw = 12, (434, 574)
    c2w = estimate(trajectory(n, 8), seed=13)
    k4 = np.asarray([480.25, 479.5, 287.0, 217.0], dtype=np.float32)
    raw = half(rng.uniform(0.05, 9.0, size=(n, *hw)))
    raw[:, :3] = half(600.0)  # beyond the 7-Scenes cache range: dropped by both paths
    mask = rng.random((n, *hw)) > 0.1
    npz_file = write_npz_prediction(tmp_path / "npz", "chess__seq-03", c2w, k4=k4, input_hw=hw)
    write_cache(tmp_path / "npz" / "frontend-cache" / "seven_scenes" / "chess__seq-03",
                np.where(mask, raw, half(7.0)), mask)
    write_vipe_output(tmp_path / "vipe", "chess__seq-03", c2w, k4=k4, depth=vipe_depth(raw, mask), input_hw=hw)
    found = vipe_output.locate(tmp_path / "vipe", "local", "chess__seq-03")
    assert found is not None
    with vipe_output.materialize(found, "chess__seq-03", with_depth=True, work_dir=tmp_path / "work") as converted:
        with np.load(npz_file) as a, np.load(converted.prediction) as b:
            assert sorted(b.files) == ["c2w", "frame_ids", "input_hw", "intrinsics"]
            for key in b.files:
                assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), key
        npz_cache = points.find_depth_cache(npz_file, "chess__seq-03", ("seven_scenes",), None)
        vipe_cache = points.find_depth_cache(converted.prediction, "chess__seq-03", (), converted.depth_cache)
        assert npz_cache is not None and vipe_cache is not None
        depth_a, mask_a = points.open_depth_cache(npz_cache, n, hw)
        depth_b, mask_b = points.open_depth_cache(vipe_cache, n, hw)
        rules = {
            "7scenes": lambda d, m: (m != 0) & np.isfinite(d) & (d >= np.float32(0.2)) & (d <= np.float32(500.0)),
            "oxford": lambda d, m: (m != 0) & np.isfinite(d) & (d > oxford.POINT_PRED_MIN_DEPTH_M),
        }
        for name, rule in rules.items():
            seen_a = np.where(rule(depth_a, mask_a), depth_a, 0.0)
            seen_b = np.where(rule(depth_b, mask_b), depth_b, 0.0)
            assert np.array_equal(seen_a, seen_b), name
        assert converted.provenance["depth_raster_hw"] == list(hw)
        del depth_a, mask_a, depth_b, mask_b
    assert not any((tmp_path / "work").iterdir()), "scratch directory left behind"


# --------------------------------------------------------------------------- points


def gt_7scenes_depth(root: Path, case: str, k_native: np.ndarray) -> np.ndarray:
    """``depth.proj.png`` (uint16 mm) of a tilted plane in a window, 0 elsewhere; returns metres."""

    cv2 = points.cv2_module()
    ids = sevenscenes.source_ids(case)
    vv, uu = np.mgrid[0:480, 0:640]
    window = (vv >= 200) & (vv < 260) & (uu >= 280) & (uu < 360)
    depths = []
    for index, frame_id in enumerate(ids.tolist()):
        metres = 1.5 + 0.002 * uu + 0.001 * vv + 0.01 * index
        mm = np.where(window, np.round(metres * 1000.0), 0).astype(np.uint16)
        cv2.imwrite(str(root / sevenscenes.SOURCE_BY_KEY[case] / f"frame-{frame_id:06d}.depth.proj.png"), mm)
        depths.append(mm.astype(np.float32) / 1000.0)
    return np.stack(depths)


@needs_points
def test_points_7scenes_equal(tmp_path: Path) -> None:
    cv2 = points.cv2_module()
    case = "stairs__seq-01"
    gt_root = tmp_path / "gt"
    gt_c2w = gt_7scenes_poses(gt_root, case)
    gt_depth = gt_7scenes_depth(gt_root, case, sevenscenes.NATIVE_INTRINSICS)
    n, hw = len(gt_c2w), sevenscenes.INPUT_HW
    rng = np.random.default_rng(1)
    raw = half(np.stack([cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) for d in gt_depth])
               * 1.1 + rng.normal(scale=0.01, size=(n, *hw)))
    mask = raw > 0.05
    raw[:, 210, 260:270] = half(0.1)  # below the cache range, still "valid" for the depth model
    k4 = np.asarray([525.0 * 574 / 640, 525.0 * 434 / 480, 320.0 * 574 / 640, 240.0 * 434 / 480])
    pred = estimate(gt_c2w, seed=14, scale=1.0)
    write_npz_prediction(tmp_path / "npz", case, pred, k4=k4, input_hw=hw,
                     extra={"source_frame_ids": sevenscenes.source_ids(case)})
    write_cache(tmp_path / "npz" / "frontend-cache" / "seven_scenes" / case, np.where(mask, raw, 3.0), mask)
    write_vipe_output(tmp_path / "vipe", case, pred, k4=k4, depth=vipe_depth(raw, mask), input_hw=hw)
    common = ["--dataset", "7scenes", "--metric", "points", "--gt-root", str(gt_root), "--cases", case]
    npz = run_score(tmp_path, "npz", [*common, "--pred", str(tmp_path / "npz")])
    got = run_score(tmp_path, "vipe", [*common, "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe",
                                       "--work-dir", str(tmp_path / "work")])
    assert_same_scores(npz, got, vipe_root=tmp_path / "vipe", name=case)
    assert npz["runs"][0]["per_case"][case]["sim3_correspondences"] > 1000
    assert not any((tmp_path / "work").iterdir())


@needs_points
def test_points_oxford_equal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, n = "keble-college-02", 20
    monkeypatch.setattr(oxford, "EXPECTED_SCENE_FRAMES", {scene: n for scene in oxford.CASES})
    root = tmp_path / "points_gt"
    native_hw = (108, 144)
    gt_c2w = trajectory(n, 9, radius=8.0)
    processed = root / "processed_sequences" / case
    processed.mkdir(parents=True)
    np.savetxt(processed / "poses_c2w.txt", gt_c2w.reshape(n, 16), fmt="%.17g")
    np.savetxt(processed / "intrinsics.txt", [[100.0, 100.0, 71.5, 53.5, native_hw[1], native_hw[0]]], fmt="%.17g")
    frame_ids = np.arange(0, n, oxford.POINT_INTERVAL)
    pc = root / "pc" / case
    (pc / "depth").mkdir(parents=True)
    (pc / "DONE.json").write_text(json.dumps({"source_frame_count": n, "width": native_hw[1],
                                              "height": native_hw[0]}))
    np.savetxt(pc / "frame_ids.txt", frame_ids, fmt="%d")
    np.savetxt(pc / "poses_c2w.txt", gt_c2w[frame_ids].reshape(-1, 16), fmt="%.17g")
    vv, uu = np.mgrid[0:native_hw[0], 0:native_hw[1]]
    for index, frame_id in enumerate(frame_ids.tolist()):
        depth = (6.0 + 0.05 * uu + 0.03 * vv + index).astype(np.float32)
        depth[:10] = 0.0
        np.save(pc / "depth" / f"{frame_id:06d}.npy", depth)
    hw = oxford.TARGET_HW
    pv, pu = np.mgrid[0:hw[0], 0:hw[1]]
    rng = np.random.default_rng(2)
    raw = half(np.stack([6.5 + 0.014 * pu + 0.008 * pv + 0.025 * i for i in range(n)])
               + rng.normal(scale=0.02, size=(n, *hw)))
    mask = np.ones((n, *hw), dtype=bool)
    mask[:, :20] = False
    k4 = np.asarray([360.0, 360.0, 258.5, 188.5])
    pred = estimate(gt_c2w, seed=15, scale=1.0)
    write_npz_prediction(tmp_path / "npz", case, pred, k4=k4, input_hw=hw)
    write_cache(tmp_path / "npz" / "frontend-cache" / "oxford" / case, np.where(mask, raw, 9.0), mask)
    write_vipe_output(tmp_path / "vipe", case, pred, k4=k4, depth=vipe_depth(raw, mask), input_hw=hw)
    common = ["--dataset", "oxford", "--metric", "points", "--gt-root", str(root), "--cases", case]
    npz = run_score(tmp_path, "npz", [*common, "--pred", str(tmp_path / "npz")])
    got = run_score(tmp_path, "vipe", [*common, "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"])
    assert_same_scores(npz, got, vipe_root=tmp_path / "vipe", name=case)
    assert npz["runs"][0]["per_case"][case]["num_alignment_correspondences"] > 1000


# --------------------------------------------------------------------------- refusals


def _vipe_7scenes(tmp_path: Path, **kwargs) -> tuple[Path, list[str]]:
    case = "stairs__seq-01"
    gt = gt_7scenes_poses(tmp_path / "gt", case)
    write_vipe_output(tmp_path / "vipe", case, estimate(gt, seed=16), **kwargs)
    return tmp_path / "vipe", ["--dataset", "7scenes", "--gt-root", str(tmp_path / "gt"), "--cases", case,
                               "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"]


def _refused(tmp_path: Path, argv: list[str], text: str) -> None:
    with pytest.raises(SystemExit) as raised:
        score.main([*argv, "--out", str(tmp_path / "refused.json")])
    assert text in str(raised.value), raised.value


def test_refuses_frame_stride(tmp_path: Path) -> None:
    record = {"status": "success", "input_raster": {"policy": "cover_crop", "frame_stride": 2,
                                                    "source_hw": [480, 640], "prepared_hw": [280, 504]}}
    _, argv = _vipe_7scenes(tmp_path, record=record)
    _refused(tmp_path, argv, "kept one frame in 2")


@needs_exr
def test_refuses_points_on_cropped_raster(tmp_path: Path) -> None:
    hw = (280, 504)
    record = {"status": "success", "tracking": {"num_frames": 100, "input_hw": list(hw)},
              "input_raster": {"policy": "cover_crop", "frame_stride": 1, "source_hw": [480, 640],
                               "prepared_hw": list(hw)}}
    _, argv = _vipe_7scenes(tmp_path, k4=np.ones(4), depth=np.ones((100, *hw), np.float32), record=record)
    _refused(tmp_path, [*argv, "--metric", "points"], "was cropped from")


def test_accepts_points_on_plain_resize_raster(tmp_path: Path) -> None:
    """7-Scenes: 480x640 stretched to 434x574 (policy resize) is pure scaling, so points pass the record check."""
    from metrictrail_eval import vipe_output

    record = {"status": "success", "tracking": {"num_frames": 100, "input_hw": [434, 574]},
              "input_raster": {"policy": "resize", "frame_stride": 1, "source_hw": [480, 640],
                               "prepared_hw": [434, 574]}}
    summary = vipe_output.check_run_record(record, 100, points=True, source=tmp_path / "record.json")
    assert summary["input_raster"]["policy"] == "resize"
    cropped = {**record, "input_raster": {**record["input_raster"], "policy": "cover_crop"}}
    with pytest.raises(vipe_output.VipeOutputError, match="was cropped from"):
        vipe_output.check_run_record(cropped, 100, points=True, source=tmp_path / "record.json")

def test_refuses_frame_count_and_failed_run(tmp_path: Path) -> None:
    _, argv = _vipe_7scenes(tmp_path, record={"status": "success", "tracking": {"num_frames": 99}})
    _refused(tmp_path, argv, "tracked 99 frames")
    (tmp_path / "vipe" / "vipe" / "stairs__seq-01_metrictrail.json").write_text(json.dumps({"status": "failed"}))
    _refused(tmp_path, argv, "run status is 'failed'")


def test_refuses_points_without_depth_or_intrinsics(tmp_path: Path) -> None:
    _, argv = _vipe_7scenes(tmp_path, k4=np.ones(4))
    _refused(tmp_path, [*argv, "--metric", "points"], "save_metric_depth=off")
    (tmp_path / "vipe" / "intrinsics" / "stairs__seq-01.npz").unlink()
    _refused(tmp_path, [*argv, "--metric", "points"], "needed for point metrics")


def test_refuses_options(tmp_path: Path) -> None:
    root, argv = _vipe_7scenes(tmp_path)
    _refused(tmp_path, [*argv, "--depth-cache", str(tmp_path)], "--depth-cache applies to --pred-format npz")
    _refused(tmp_path, [*argv, "--vipe-name", "nope=x"], "expected CASE=NAME")
    _refused(tmp_path, [*argv, "--vipe-name", "stairs__seq-01=../x"], "expected CASE=NAME")
    _refused(tmp_path, [a for a in argv if a != "vipe" and a != "--pred-format"] + ["--vipe-name", "a=b"],
             "--vipe-name applies to --pred-format vipe")
    _refused(tmp_path, [*argv[:-3], str(tmp_path / "gt"), "--pred-format", "vipe"], "not a ViPE output directory")
    _refused(tmp_path, [*argv, "--stage", "LC"], "no prediction under")


@needs_evo
def test_refuses_historical_layout_protocol(tmp_path: Path) -> None:
    gt = gt_kitti(tmp_path / "gt", "04", 50)
    write_vipe_output(tmp_path / "vipe", "04", estimate(gt, seed=17), record={"status": "success"})
    _refused(tmp_path, ["--dataset", "kitti", "--protocol", "umeyama", "--gt-root", str(tmp_path / "gt"),
                        "--cases", "04", "--pred", str(tmp_path / "vipe"), "--pred-format", "vipe"],
             "not available with --pred-format vipe")


def test_allow_partial_skips_missing_names(tmp_path: Path) -> None:
    case = "stairs__seq-01"
    gt = gt_7scenes_poses(tmp_path / "gt", case)
    write_vipe_output(tmp_path / "vipe", "other", estimate(gt, seed=18))
    argv = ["--dataset", "7scenes", "--gt-root", str(tmp_path / "gt"), "--cases", case, "--pred",
            str(tmp_path / "vipe"), "--pred-format", "vipe"]
    _refused(tmp_path, argv, "use --allow-partial")
    _refused(tmp_path, [*argv, "--allow-partial"], "no 7scenes predictions found")


# --------------------------------------------------------------------------- ViPE's own io


def _import_vipe_io():
    """``vipe.utils.io`` on a machine without the compiled ``vipe_ext`` (stubbed: no kernel runs here)."""

    pytest.importorskip("torch")
    if importlib.util.find_spec("vipe") is None:
        pytest.skip("vipe not importable")
    if "vipe_ext" not in sys.modules and importlib.util.find_spec("vipe_ext") is None:
        # As ViPE's own CPU tests do: every kernel resolves to a stub that raises if it is called.
        def submodule(name: str) -> types.ModuleType:
            if name.startswith("__"):
                raise AttributeError(name)
            module = types.ModuleType(f"vipe_ext.{name}")

            def kernel(attribute: str):
                if attribute.startswith("__"):
                    raise AttributeError(attribute)

                def unavailable(*args, **kwargs):
                    raise RuntimeError(f"vipe_ext.{name}.{attribute} is a test stub")

                return unavailable

            module.__dict__["__getattr__"] = kernel
            return module

        stub = types.ModuleType("vipe_ext")
        stub.__dict__["__getattr__"] = submodule
        sys.modules["vipe_ext"] = stub
    try:
        from vipe.streams.base import FrameAttribute
        from vipe.utils import io
        from vipe.utils.cameras import CameraType
    except Exception as error:  # noqa: BLE001 - optional cross-check
        pytest.skip(f"vipe.utils.io not importable: {error}")
    return io, FrameAttribute, CameraType


@needs_exr
def test_reader_matches_vipe_io(tmp_path: Path) -> None:
    """Artifacts written by ViPE's own ``save_*_artifacts`` convert to the plain prediction's arrays, and the
    torch-free depth reader decodes exactly what ``read_depth_artifacts`` returns."""

    io, FrameAttribute, CameraType = _import_vipe_io()
    import torch

    rng = np.random.default_rng(3)
    n, hw = 9, (31, 47)
    c2w = estimate(trajectory(n, 10), seed=19)
    k4 = np.asarray([40.5, 41.25, 23.0, 15.0], dtype=np.float32)
    depth = half(rng.uniform(0.0, 80.0, size=(n, *hw)))
    depth[:, :2] = 0.0

    class Pose:
        def __init__(self, matrix: np.ndarray) -> None:
            self._matrix = torch.from_numpy(matrix)

        def matrix(self) -> torch.Tensor:
            return self._matrix

    class Stream:
        attributes = {
            FrameAttribute.POSE: [Pose(pose) for pose in c2w],
            FrameAttribute.INTRINSICS: [torch.from_numpy(k4.copy()) for _ in range(n)],
            FrameAttribute.CAMERA_TYPE: [CameraType.PINHOLE] * n,
            FrameAttribute.METRIC_DEPTH: [torch.from_numpy(d.copy()) for d in depth],
        }

        def get_stream_attribute(self, attribute):
            return self.attributes[attribute]

    out = io.ArtifactPath(tmp_path / "vipe", "seq")
    io.save_pose_artifacts(out, Stream())
    io.save_intrinsics_artifacts(out, Stream())
    io.save_depth_artifacts(out, Stream())
    reference = [(index, value.numpy()) for index, value in io.read_depth_artifacts(out.depth_path)]
    ours = list(vipe_output.iter_depth(out.depth_path))
    assert [i for i, _ in ours] == [i for i, _ in reference] == list(range(n))
    for (_, a), (_, b) in zip(ours, reference):
        assert a.dtype == b.dtype == np.float32 and np.array_equal(a, b)
    assert np.array_equal(np.stack([a for _, a in ours]), depth)

    found = vipe_output.locate(tmp_path / "vipe", "local", "seq")
    assert found is not None
    with vipe_output.materialize(found, "seq", with_depth=True) as converted:
        with np.load(converted.prediction) as arrays:
            assert np.array_equal(arrays["c2w"], c2w) and arrays["c2w"].dtype == np.float64
            assert np.array_equal(arrays["frame_ids"], np.arange(n, dtype=np.int32))
            assert np.array_equal(arrays["intrinsics"], np.repeat(k4[None], n, 0))
            assert arrays["input_hw"].tolist() == list(hw)
        cache = converted.depth_cache / "seq"
        cached = np.fromfile(cache / "depth.f32", dtype=np.float32).reshape(n, *hw)
        assert np.array_equal(cached, depth)
        assert np.array_equal(np.fromfile(cache / "mask.u8", dtype=np.uint8).reshape(n, *hw), depth != 0)
