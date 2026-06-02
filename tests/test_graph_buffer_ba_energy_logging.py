# mypy: ignore-errors

from __future__ import annotations

import ast
import logging
from pathlib import Path
import unittest


BUFFER_PATH = Path(__file__).resolve().parents[1] / "vipe" / "slam" / "components" / "buffer.py"
HELPER_NAMES = {
    "_should_compute_ba_energy",
    "_ba_energy_value",
    "_log_ba_energy",
}


def _load_ba_energy_helpers():
    tree = ast.parse(BUFFER_PATH.read_text(), filename=str(BUFFER_PATH))
    helper_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
    ]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "logging": logging,
        "logger": logging.getLogger("vipe.slam.components.buffer"),
    }
    exec(compile(module, str(BUFFER_PATH), "exec"), namespace)
    return namespace


class _ItemMustNotBeCalled:
    def __getitem__(self, _):
        return self

    def item(self):
        raise AssertionError("BA energy scalar was materialized")


class _CountingEnergyValue:
    def __init__(self, value: float) -> None:
        self.value = value
        self.item_calls = 0

    def item(self) -> float:
        self.item_calls += 1
        return self.value


class _CountingEnergy:
    def __init__(self) -> None:
        self.values = [_CountingEnergyValue(1.25), _CountingEnergyValue(0.5)]

    def __getitem__(self, idx):
        return self.values[idx]


class GraphBufferBAEnergyLoggingTest(unittest.TestCase):
    def test_ba_energy_log_helper_skips_item_when_info_disabled(self):
        helpers = _load_ba_energy_helpers()
        logger = helpers["logger"]
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            should_log = helpers["_should_compute_ba_energy"](True)
            if should_log:
                helpers["_log_ba_energy"](1, _ItemMustNotBeCalled())
        finally:
            logger.setLevel(previous_level)

        self.assertFalse(should_log)

    def test_ba_energy_log_helper_materializes_item_when_info_enabled(self):
        helpers = _load_ba_energy_helpers()
        energy = _CountingEnergy()

        with self.assertLogs("vipe.slam.components.buffer", level="INFO") as logs:
            should_log = helpers["_should_compute_ba_energy"](True)
            if should_log:
                helpers["_log_ba_energy"](3, energy)

        self.assertTrue(should_log)
        self.assertEqual([value.item_calls for value in energy.values], [1, 1])
        self.assertIn("BA iters = 3, energy: 1.25 -> 0.5", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
