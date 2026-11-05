# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINO-DPVO checkpoint: payload checks, weight resolution and network loading.

The DINO-DPVO checkpoint holds only DPVO's
``patchify`` and ``update`` modules; the frozen DINOv3 ViT-S/16 backbone is
omitted and loaded from its own weights. Its ``model_config`` is SHA-256 bound
and names the DINOv3 weights used at training time (a machine-specific path), so
the loader verifies, in this order: the file digest, the ``model_config``
digest, the structure (matching + context DINO fusion, no intrinsics
calibration, only the backbone omitted); only then is the configured DINOv3
weights file substituted, and it must match the ``weights_sha256`` recorded in
the verified config.

Weight files are resolved the ViPE way: explicit config value, else an
environment variable, else a file under ``$TORCH_HOME/hub``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .dino_fusion import dino_from_config, make_dino_patchifier_class, sha256_file

DINO_DPVO_SHA256 = "d0aa57e165b159fc84bdb9d94918ad9b508c153acfb4c46f5058ab80421fb5f5"
DINO_DPVO_FILENAME = "dino_dpvo_multidataset15k.pth"
DINO_DPVO_ENV = "VIPE_METRICTRAIL_DPVO_CHECKPOINT"

DINOV3_VITS16_SHA256_PREFIX = "08c60483"
DINOV3_VITS16_FILENAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINOV3_VITS16_ENV = "VIPE_DINOV3_VITS16_CHECKPOINT"


def torch_hub_dir() -> Path:
    return Path(torch.hub.get_dir())


def resolve_dino_dpvo_checkpoint(configured: str | Path | None) -> Path:
    """Config value, else ``$VIPE_METRICTRAIL_DPVO_CHECKPOINT``, else ``$TORCH_HOME/hub/metrictrail/<file>``."""

    if configured:
        return Path(configured).expanduser()
    if os.environ.get(DINO_DPVO_ENV):
        return Path(os.environ[DINO_DPVO_ENV]).expanduser()
    return torch_hub_dir() / "metrictrail" / DINO_DPVO_FILENAME


def resolve_dinov3_weights(configured: str | Path | None) -> Path:
    """Config value, else ``$VIPE_DINOV3_VITS16_CHECKPOINT``, else ``$TORCH_HOME/hub/checkpoints/<file>``."""

    if configured:
        return Path(configured).expanduser()
    if os.environ.get(DINOV3_VITS16_ENV):
        return Path(os.environ[DINOV3_VITS16_ENV]).expanduser()
    return torch_hub_dir() / "checkpoints" / DINOV3_VITS16_FILENAME


def checkpoint_payload(path: Path, *, dinov3_weights: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Verify the DINO-DPVO checkpoint and return it with the effective DINO config.

    Order matters: file digest, then the ``model_config`` digest, then the
    structural checks, and only then the substitution of the DINOv3 weights.
    The DINOv3 weights actually used must match ``weights_sha256`` from the
    verified config.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DINO-DPVO checkpoint not found: {resolved} (set slam.weights or ${DINO_DPVO_ENV})")
    if sha256_file(resolved) != DINO_DPVO_SHA256:
        raise ValueError(f"DINO-DPVO checkpoint digest mismatch: {resolved}")
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(resolved, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError("DINO-DPVO checkpoint must be a mapping")
    required = {
        "format_version",
        "model",
        "model_config",
        "model_config_sha256",
        "omitted_external_state_prefixes",
        "phase",
    }
    if required - set(value):
        raise KeyError(f"DINO-DPVO checkpoint lacks {sorted(required - set(value))}")
    encoded = json.dumps(dict(value["model_config"]), sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != value["model_config_sha256"]:
        raise ValueError("DINO-DPVO model_config digest mismatch")
    dino = dict(value["model_config"].get("dino", {}))
    calibration = value["model_config"].get("intrinsics_calibration", {})
    if (
        int(value["format_version"]) != 1
        or value["phase"] != "joint"
        or dino.get("enabled") is not True
        or dino.get("fuse_matching") is not True
        or dino.get("fuse_context") is not True
        or bool(calibration.get("enabled", False))
        or tuple(value["omitted_external_state_prefixes"]) != ("patchify.dino.backbone.",)
    ):
        raise ValueError("checkpoint is not the matching+context DINO-DPVO model")
    # The verified config names training-time locations; substitute the configured
    # weights (ViPE's in-tree DINOv3 replaces the hub checkout). The digest below
    # still binds the weights themselves.
    dino.pop("repository", None)
    dino["weights"] = str(Path(dinov3_weights).expanduser().resolve())
    weights = Path(dino["weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"DINOv3 weights are missing: {weights} (set slam.dinov3_weights or ${DINOV3_VITS16_ENV})"
        )
    if sha256_file(weights) != dino["weights_sha256"]:
        raise ValueError("external DINOv3-S backbone digest mismatch")
    return value, dino


def load_dino_dpvo_network(
    network_class: type[torch.nn.Module], path: Path, *, dinov3_weights: Path
) -> torch.nn.Module:
    """Construct DPVO's VONet with the DINO-fused patchifier and load the weights."""

    value, dino = checkpoint_payload(path, dinov3_weights=dinov3_weights)
    patchifier_class = make_dino_patchifier_class()
    network: Any = network_class()
    network.patchify = patchifier_class(
        dino_from_config(dino),
        patch_size=int(network.P),
        adapter_dim=int(dino.get("adapter_dim", 32)),
        fuse_matching=True,
        fuse_context=True,
    )
    state = OrderedDict((str(key).removeprefix("module."), tensor) for key, tensor in value["model"].items())
    missing, unexpected = network.load_state_dict(state, strict=False)
    invalid_missing = [name for name in missing if not name.startswith("patchify.dino.backbone.")]
    if invalid_missing or unexpected:
        raise RuntimeError(f"DINO-DPVO state mismatch: missing={invalid_missing}, unexpected={unexpected}")
    network._metrictrail_dino_enabled = True
    network._metrictrail_dino_matching = True
    network._metrictrail_dino_context = True
    return network


def checkpoint_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {"file": resolved.name, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
