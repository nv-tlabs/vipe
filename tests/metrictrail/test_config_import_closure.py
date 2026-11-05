# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViPE's config import closure stays numpy- and torch-free with MetricTrail registered.

ViPE's Docs workflow (``.github/workflows/pages.yml``) installs only the ``docs``
dependency group (hydra-core, mkdocs, omegaconf, pydantic) and then runs
``scripts/generate_config_docs.py --check``, which imports ``vipe.config.pipeline``
and therefore ``vipe.metrictrail.config`` and the SharedK schema. Each check runs
in a fresh interpreter in which importing numpy or torch raises.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SCRIPT = REPO_ROOT / "scripts" / "generate_config_docs.py"

BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = ("numpy", "torch")


    class _Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r} (blocked by the test)", name=name)
            return None


    sys.meta_path.insert(0, _Blocked())


    def assert_closure_clean():
        leaked = sorted(name for name in sys.modules if name.partition(".")[0] in BLOCKED)
        assert not leaked, leaked
    """
)


def _run(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(body)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_blocker_blocks() -> None:
    result = _run("import numpy\n")
    assert result.returncode != 0 and "blocked by the test" in result.stderr


def test_vipe_config_and_metrictrail_schema_import_without_numpy_or_torch() -> None:
    result = _run(
        """
        import vipe.config.pipeline
        import vipe.config
        import vipe.metrictrail
        import vipe.metrictrail.config
        import vipe.metrictrail.calibration
        import vipe.metrictrail.calibration.config
        import vipe.metrictrail.calibration.policy
        from vipe.metrictrail.calibration import MODES, SCHEMA
        from vipe.config import parse_typed_config
        from vipe.metrictrail.config import MetricTrailPipelineConfig

        config = parse_typed_config("default", ["pipeline=metrictrail", "streams.base_path=clip.mp4"])
        assert isinstance(config.pipeline, MetricTrailPipelineConfig)
        assert config.pipeline.init.calibration.terminal_min_accepted_keyframes == 16
        assert SCHEMA == "sharedk_calibration_v1" and "robust128" in MODES
        assert_closure_clean()
        print("closure-ok")
        """
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "closure-ok" in result.stdout


@pytest.mark.skipif(not DOCS_SCRIPT.is_file(), reason="needs a ViPE source checkout")
def test_generate_config_docs_check_runs_without_numpy_or_torch() -> None:
    result = _run(
        f"""
        import runpy

        sys.argv = [{str(DOCS_SCRIPT)!r}, "--check"]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
            code = 0
        except SystemExit as exit:
            code = exit.code if isinstance(exit.code, int) else (0 if exit.code is None else 1)
        assert_closure_clean()
        sys.exit(code)
        """
    )
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-3000:])


def test_protocol_reexports_the_policy_objects() -> None:
    from vipe.metrictrail.calibration import policy, protocol

    for name in (
        "SCHEMA",
        "MAX_SAFE_ACCEPTED_KEYFRAMES",
        "FINAL_STEPS",
        "MILESTONE_STEPS",
        "OFFICIAL_MILESTONES",
        "TERMINAL_MIN_FLOOR",
        "ModeSpec",
        "MODES",
        "PIPELINE_BUDGETS",
        "permits_terminal_exhaustion_endpoint",
        "check_terminal_policy",
    ):
        assert getattr(protocol, name) is getattr(policy, name), name
