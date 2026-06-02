import unittest
from pathlib import Path

try:
    import torch

    from vipe.slam.components.factor_graph import FactorGraph
except ImportError:  # pragma: no cover - lets local validation run without runtime deps
    torch = None
    FactorGraph = None


class FactorGraphMotionCompileProbeStaticTest(unittest.TestCase):
    def test_reduce_overhead_probe_is_wired_to_motion_feature_sites(self):
        source = (Path(__file__).resolve().parents[1] / "vipe/slam/components/factor_graph.py").read_text()

        self.assertIn("MOTION_FEATURE_COMPILE_MARKER", source)
        self.assertIn('mode="reduce-overhead"', source)
        self.assertIn('fullgraph=True', source)
        self.assertIn('f"{MOTION_FEATURE_COMPILE_MARKER}:compiled"', source)
        self.assertEqual(source.count("self._prepare_motion_features(coords1)"), 1)


@unittest.skipUnless(torch is not None and FactorGraph is not None, "torch and VIPE runtime deps are required")
class FactorGraphMotionCompileProbeRuntimeTest(unittest.TestCase):
    def test_cpu_motion_feature_path_falls_back_to_observable_eager_marker(self):
        coords0 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
        coords1 = torch.tensor(
            [
                [[[0.25, -100.0], [2.0, 0.5]], [[-0.5, 1.5], [70.0, -70.0]]],
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            ]
        )
        target = torch.tensor(
            [
                [
                    [[[0.75, 0.5], [1.0, 1.5]], [[0.25, 2.5], [5.0, -10.0]]],
                    [[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]],
                ]
            ]
        )

        graph = object.__new__(FactorGraph)
        graph.coords0 = coords0
        graph.target = target

        coords1_batched, motn = graph._prepare_motion_features(coords1)

        reference_coords1 = coords1[None]
        reference = torch.cat([reference_coords1 - coords0, target - reference_coords1], dim=-1)
        reference = reference.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        torch.testing.assert_close(coords1_batched, reference_coords1, atol=0.0, rtol=0.0)
        torch.testing.assert_close(motn, reference, atol=0.0, rtol=0.0)

        stats = graph.motion_feature_compile_stats
        self.assertEqual(stats["compiled"], 0)
        self.assertEqual(stats["eager"], 1)
        self.assertEqual(stats["unsupported"], 1)
        self.assertIn("non_cuda_device", stats["last_marker"])


if __name__ == "__main__":
    unittest.main()
