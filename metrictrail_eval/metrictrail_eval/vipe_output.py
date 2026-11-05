"""Read a ViPE output directory as predictions (``--pred-format vipe``).

A ViPE run (``pipeline.output.path``) holds, per input video ``<name>``:

* ``pose/<name>.npz``: ``data`` [N, 4, 4] OpenCV camera-to-world and ``inds``
  [N] frame indices (``vipe.utils.io.save_pose_artifacts``; ``pipeline=metrictrail``
  stores ``data`` as float64 and adds ``keyframe_ids``, ``runtime``,
  ``source_frame_ids``, which are not read here);
* ``pose_lc/<name>.npz``: the optional loop-closure output in the same format
  (``--stage LC``; ``pose/`` stays the local trajectory);
* ``intrinsics/<name>.npz``: ``data`` [N, 4] ``fx, fy, cx, cy`` and ``inds``,
  plus ``intrinsics/<name>_camera.txt`` (``idx: CAMERA_TYPE`` per frame);
* ``depth/<name>.zip``: one ``{frame:05d}.exr`` per frame with one HALF channel
  ``Z``, metres, 0 where there is no depth (``save_depth_artifacts`` /
  ``read_depth_artifacts``; ``pipeline=metrictrail`` saves the MoGe-2 metric
  depth of its final tracking pass there);
* ``vipe/<name>_metrictrail.json``: the ``pipeline=metrictrail`` run record.

:func:`materialize` turns one sequence into exactly what the dataset scorers read
from a plain prediction directory, in a scratch directory that is deleted
afterwards, so every protocol runs unchanged:

* ``<case>.npz``: ``frame_ids`` = ``inds`` (int32), ``c2w`` = ``data``
  (float64) and, when known, ``input_hw`` (the run record's tracked raster, or
  the depth raster); for ``--metric points`` also ``intrinsics`` = the
  intrinsics ``data`` (as stored, float32 in ``pipeline=metrictrail``);
* ``--metric points``: a frontend depth cache ``<case>/{depth.f32, mask.u8,
  contract.json, SUCCESS}`` holding the decoded depth (float32 of the stored
  HALF values) and ``mask = depth != 0``, read through the scorers' own cache
  rules (7-Scenes ``mask & finite & 0.2 <= d <= 500``, Oxford ``mask & finite &
  d > 1e-4``; both reject 0, so the mask only restates ViPE's "0 = no depth").

ViPE's ``source_frame_ids`` are the stream's raw frame indices, not dataset frame
ids, and are not mapped to the ``source_frame_ids`` of a plain prediction file.

When the run record exists it is checked: ``status`` success, ``tracking.num_frames``
= the pose count, no frame stride in ``input_raster`` (benchmark GT is matched by
prepared frame ordinal), and for points an input raster that was not cropped
(the point protocols map the prediction raster onto the dataset camera by pure
scaling, so ``policy=unchanged`` and the plain ``policy=resize`` used for
7-Scenes are accepted, ``cover_crop`` is not).

The depth reader mirrors ``vipe.utils.io.read_depth_artifacts`` without torch
(it needs the ``OpenEXR`` and ``Imath`` modules; trajectory scoring does not); an
EXR that cannot be decoded is an error here, not an all-NaN map.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

FORMAT = "vipe"
#: Pose artifact directory per ``--stage``.
POSE_DIRS = {"local": "pose", "LC": "pose_lc"}
PINHOLE = "PINHOLE"
DEPTH_ENTRY = re.compile(r"^(\d+)\.exr$")
#: Scratch sub-directory holding the converted depth caches (``<root>/<case>/depth.f32``).
DEPTH_CACHE_DIR = "depth_cache"


class VipeOutputError(ValueError):
    """A ViPE output that cannot be scored (missing or inconsistent artifact)."""


@dataclass(frozen=True)
class VipeArtifacts:
    """The artifact paths of one ViPE output name (files may not exist)."""

    root: Path
    name: str
    stage: str

    @property
    def pose(self) -> Path:
        return self.root / POSE_DIRS[self.stage] / f"{self.name}.npz"

    @property
    def intrinsics(self) -> Path:
        return self.root / "intrinsics" / f"{self.name}.npz"

    @property
    def camera_types(self) -> Path:
        return self.root / "intrinsics" / f"{self.name}_camera.txt"

    @property
    def depth(self) -> Path:
        return self.root / "depth" / f"{self.name}.zip"

    @property
    def run_record(self) -> Path:
        return self.root / "vipe" / f"{self.name}_metrictrail.json"


def is_vipe_output(root: Path) -> bool:
    return (Path(root) / POSE_DIRS["local"]).is_dir()


def locate(root: Path, stage: str, name: str) -> VipeArtifacts | None:
    """The artifacts of ``name`` when its pose for ``stage`` exists, else None."""

    if stage not in POSE_DIRS:
        raise ValueError(f"stage must be one of {tuple(POSE_DIRS)}, got {stage!r}")
    if not name or Path(name).name != name:
        raise VipeOutputError(f"invalid ViPE artifact name {name!r}")
    found = VipeArtifacts(Path(root), name, stage)
    return found if found.pose.is_file() else None


# --------------------------------------------------------------------------- readers


def _check_inds(inds: np.ndarray, count: int, source: Path) -> np.ndarray:
    if inds.ndim != 1 or len(inds) != count or not np.issubdtype(inds.dtype, np.integer):
        raise VipeOutputError(
            f"{source}: inds must be integer [N] with N = {count}, got {inds.dtype} {inds.shape}"
        )
    inds = inds.astype(np.int64)
    if count == 0 or inds[0] < 0 or np.any(np.diff(inds) <= 0):
        raise VipeOutputError(f"{source}: inds must be non-empty, non-negative and strictly increasing")
    return inds


def read_pose(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """``pose/<name>.npz`` -> (c2w float64 [N, 4, 4], inds int64 [N], stored dtype)."""

    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive.files or "inds" not in archive.files:
            raise VipeOutputError(f"{path} is not a ViPE pose artifact (needs data and inds)")
        data = np.asarray(archive["data"])
        inds = np.asarray(archive["inds"])
    if data.ndim != 3 or data.shape[1:] != (4, 4):
        raise VipeOutputError(f"{path}: data must be [N, 4, 4] camera-to-world, got {data.shape}")
    inds = _check_inds(inds, len(data), path)
    return np.asarray(data, dtype=np.float64), inds, data.dtype.str


def read_intrinsics(found: VipeArtifacts, inds: np.ndarray) -> np.ndarray:
    """``intrinsics/<name>.npz`` ``data`` [N, 4] (as stored) for the pose frames; pinhole only."""

    path = found.intrinsics
    if not path.is_file():
        raise VipeOutputError(f"{found.name}: no {path} (needed for point metrics)")
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive.files or "inds" not in archive.files:
            raise VipeOutputError(f"{path} is not a ViPE intrinsics artifact (needs data and inds)")
        data = np.asarray(archive["data"])
        k_inds = np.asarray(archive["inds"])
    if data.ndim != 2 or data.shape[1] != 4:
        raise VipeOutputError(f"{path}: data must be pinhole [N, 4] fx, fy, cx, cy, got {data.shape}")
    if not np.array_equal(_check_inds(k_inds, len(data), path), inds):
        raise VipeOutputError(f"{path}: inds differ from the pose frames")
    if found.camera_types.is_file():
        types: dict[int, str] = {}
        for line in found.camera_types.read_text(encoding="utf-8").splitlines():
            if line.strip():
                index, _, kind = line.partition(":")
                types[int(index)] = kind.strip()
        if sorted(types) != inds.tolist() or any(kind != PINHOLE for kind in types.values()):
            raise VipeOutputError(
                f"{found.camera_types}: point metrics need a {PINHOLE} camera on every pose frame"
            )
    return data


def _openexr() -> tuple[Any, Any]:
    try:
        import Imath
        import OpenEXR
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit("--pred-format vipe --metric points needs the OpenEXR and Imath modules") from error
    return OpenEXR, Imath


def depth_entries(archive: zipfile.ZipFile, source: Path) -> list[tuple[int, str]]:
    """``(frame index, member name)`` of every ``{frame:05d}.exr`` member, in frame order."""

    entries = []
    for member in archive.namelist():
        match = DEPTH_ENTRY.match(member)
        if match is None:
            raise VipeOutputError(f"{source}: unexpected member {member!r} (ViPE writes <frame>.exr)")
        entries.append((int(match.group(1)), member))
    entries.sort()
    indices = [index for index, _ in entries]
    if len(set(indices)) != len(indices):
        raise VipeOutputError(f"{source}: duplicate frame indices")
    return entries


def decode_depth(OpenEXR: Any, Imath: Any, handle: Any, source: str) -> np.ndarray:
    """One EXR member as ``read_depth_artifacts`` decodes it: channel ``Z`` of the data window, float32."""

    exr = OpenEXR.InputFile(handle)
    header = exr.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    channel = header["channels"].get("Z")
    if channel is None:
        raise VipeOutputError(f"{source}: no Z channel")
    if channel.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype: Any = np.float16
    elif channel.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    else:
        raise VipeOutputError(f"{source}: Z channel has pixel type {channel.type}")
    values = np.frombuffer(exr.channel("Z"), dtype=dtype).reshape(height, width)
    return values.astype(np.float32)


def iter_depth(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """``depth/<name>.zip`` -> ``(frame index, float32 [H, W] metres)`` in frame order."""

    OpenEXR, Imath = _openexr()
    with zipfile.ZipFile(path, "r") as archive:
        for index, member in depth_entries(archive, path):
            with archive.open(member) as handle:
                try:
                    depth = decode_depth(OpenEXR, Imath, handle, f"{path}:{member}")
                except VipeOutputError:
                    raise
                except Exception as error:  # noqa: BLE001 - corrupt member: refuse, never NaN-fill
                    raise VipeOutputError(f"{path}:{member} could not be decoded: {error}") from error
            yield index, depth


def read_run_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VipeOutputError(f"{path}: not a run record")
    return value


def check_run_record(
    record: dict[str, Any] | None, count: int, *, points: bool, source: Path
) -> dict[str, Any]:
    """Refuse runs whose frames or raster do not match a prepared benchmark sequence."""

    if record is None:
        return {"tracking_status": None, "input_raster": None, "input_hw": None}
    status = record.get("status", "success")
    if status != "success":
        raise VipeOutputError(f"{source}: run status is {status!r}")
    tracking = record.get("tracking") or {}
    if "num_frames" in tracking and int(tracking["num_frames"]) != count:
        raise VipeOutputError(
            f"{source}: tracked {tracking['num_frames']} frames, the pose artifact has {count}"
        )
    raster = record.get("input_raster") or {}
    stride = int(raster.get("frame_stride") or 1)
    if stride != 1:
        raise VipeOutputError(
            f"{source}: the input raster kept one frame in {stride}; benchmark GT is matched by prepared frame "
            "ordinal, so run benchmark frames with pipeline.init.input_raster.policy=unchanged"
        )
    source_hw, prepared_hw = raster.get("source_hw"), raster.get("prepared_hw")
    # A plain resize (policy "resize", e.g. 7-Scenes 480x640 -> 434x574) is pure per-axis scaling, which is
    # exactly how the point protocols map the prediction raster onto the dataset camera; a crop is not.
    resized_only = raster.get("policy") == "resize"
    if (
        points
        and not resized_only
        and source_hw is not None
        and prepared_hw is not None
        and list(source_hw) != list(prepared_hw)
    ):
        raise VipeOutputError(
            f"{source}: the input raster was cropped from {list(source_hw)} to {list(prepared_hw)}; "
            "the point protocols map the prediction raster onto the dataset camera by pure scaling, so run "
            "benchmark frames with pipeline.init.input_raster.policy=unchanged or resize"
        )
    input_hw = tracking.get("input_hw")
    raster_summary = {
        key: raster[key] for key in ("policy", "frame_stride", "source_hw", "prepared_hw") if key in raster
    }
    return {
        "tracking_status": record.get("tracking_status"),
        "input_raster": raster_summary or None,
        "input_hw": [int(value) for value in input_hw] if input_hw is not None else None,
    }


# --------------------------------------------------------------------------- conversion


@dataclass
class Converted:
    """One sequence converted for the dataset scorers (valid inside :func:`materialize`)."""

    directory: Path
    prediction: Path
    depth_cache: Path | None
    provenance: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def restore(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Replace scratch paths in a scorer result by the ViPE artifact they were made from."""

        prefixes = sorted(self.sources.items(), key=lambda item: -len(item[0]))
        restored = {}
        for key, value in values.items():
            if isinstance(value, str):
                value = next((source for prefix, source in prefixes if value.startswith(prefix)), value)
            restored[key] = value
        return restored


def write_depth_cache(
    found: VipeArtifacts, inds: np.ndarray, directory: Path
) -> tuple[list[int], dict[str, Any]]:
    """Decode ``depth/<name>.zip`` into a frontend depth cache at ``directory`` (rows in pose order)."""

    if not found.depth.is_file():
        raise VipeOutputError(
            f"{found.name}: no {found.depth} (point metrics need the saved metric depth; "
            "pipeline.output.save_metric_depth=off writes none)"
        )
    with zipfile.ZipFile(found.depth, "r") as archive:
        indices = [index for index, _ in depth_entries(archive, found.depth)]
    if indices != inds.tolist():
        raise VipeOutputError(
            f"{found.depth}: depth frames {indices[:3]}...({len(indices)}) differ from the pose frames "
            f"{inds[:3].tolist()}...({len(inds)})"
        )
    directory.mkdir(parents=True)
    count = len(inds)
    depth_map = mask_map = None
    hw: tuple[int, int] | None = None
    zero_pixels = 0
    try:
        for row, (index, depth) in enumerate(iter_depth(found.depth)):
            if index != int(inds[row]):
                raise VipeOutputError(f"{found.depth}: frame {index} out of order")
            if hw is None:
                hw = (int(depth.shape[0]), int(depth.shape[1]))
                shape = (count, *hw)
                depth_map = np.memmap(directory / "depth.f32", dtype=np.float32, mode="w+", shape=shape)
                mask_map = np.memmap(directory / "mask.u8", dtype=np.uint8, mode="w+", shape=shape)
            elif tuple(depth.shape) != hw:
                raise VipeOutputError(f"{found.depth}: frame {index} is {depth.shape}, frame {inds[0]} is {hw}")
            assert depth_map is not None and mask_map is not None
            depth_map[row] = depth
            mask_map[row] = depth != 0
            zero_pixels += int(np.count_nonzero(depth == 0))
        assert depth_map is not None and mask_map is not None and hw is not None
        depth_map.flush()
        mask_map.flush()
    finally:
        del depth_map, mask_map
    (directory / "contract.json").write_text(
        json.dumps({"count": count, "input_hw": list(hw), "source": str(found.depth)}) + "\n", encoding="utf-8"
    )
    (directory / "SUCCESS").write_text(
        json.dumps({"status": "success", "completed_frames": count}) + "\n", encoding="utf-8"
    )
    return list(hw), {"depth_raster_hw": list(hw), "depth_zero_pixels": zero_pixels}


def prediction_arrays(
    found: VipeArtifacts, *, with_depth: bool
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """The prediction-file arrays of one ViPE output (no depth), the pose inds and the record summary."""

    c2w, inds, pose_dtype = read_pose(found.pose)
    summary = check_run_record(
        read_run_record(found.run_record), len(inds), points=with_depth, source=found.run_record
    )
    arrays: dict[str, np.ndarray] = {"frame_ids": inds.astype(np.int32), "c2w": c2w}
    if with_depth:
        arrays["intrinsics"] = read_intrinsics(found, inds)
    if summary.get("input_hw") is not None:
        arrays["input_hw"] = np.asarray(summary["input_hw"], dtype=np.int32)
    provenance = {
        "name": found.name,
        "pose": str(found.pose),
        "pose_dtype": pose_dtype,
        "frames": int(len(inds)),
        "intrinsics": str(found.intrinsics) if with_depth else None,
        "depth": str(found.depth) if with_depth else None,
        "run_record": str(found.run_record) if found.run_record.is_file() else None,
        **summary,
    }
    return arrays, inds, provenance


@contextlib.contextmanager
def materialize(
    found: VipeArtifacts, case: str, *, with_depth: bool, work_dir: Path | None = None
) -> Iterator[Converted]:
    """Write ``<scratch>/<case>.npz`` (points: also ``<scratch>/depth_cache/<case>/``); removed on exit."""

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="metrictrail_eval_vipe_", dir=work_dir)).absolute()
    try:
        arrays, inds, provenance = prediction_arrays(found, with_depth=with_depth)
        depth_cache = None
        sources = {str(directory / f"{case}.npz"): str(found.pose)}
        if with_depth:
            depth_cache = directory / DEPTH_CACHE_DIR
            depth_hw, depth_info = write_depth_cache(found, inds, depth_cache / case)
            if "input_hw" in arrays and arrays["input_hw"].tolist() != depth_hw:
                raise VipeOutputError(
                    f"{found.name}: the run record's raster {arrays['input_hw'].tolist()} differs from the depth "
                    f"raster {depth_hw}"
                )
            arrays["input_hw"] = np.asarray(depth_hw, dtype=np.int32)
            provenance.update(depth_info, input_hw=depth_hw)
            sources[str(depth_cache)] = str(found.depth)
        prediction = directory / f"{case}.npz"
        np.savez(prediction, **arrays)
        for prefix, source in list(sources.items()):
            sources.setdefault(str(Path(prefix).resolve()), source)
        yield Converted(directory, prediction, depth_cache, provenance, sources)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
