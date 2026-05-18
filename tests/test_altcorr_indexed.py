import unittest

try:
    import torch

    from vipe.ext import droid_net_ext
    from vipe.slam.networks.droid_net import AltCorrBlock
except ImportError:  # pragma: no cover - lets discovery work without the runtime env
    torch = None
    droid_net_ext = None
    AltCorrBlock = None


def _has_cuda_droid_ext() -> bool:
    return torch is not None and torch.cuda.is_available() and droid_net_ext is not None


@unittest.skipUnless(_has_cuda_droid_ext(), "CUDA droid extension is required")
class AltCorrIndexedTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2028)

    def test_indexed_altcorr_matches_corr_layer_reference(self):
        device = torch.device("cuda")
        batch, n_frames, channels, height, width = 1, 7, 128, 48, 64
        n_edges = 5

        fmaps = torch.randn(batch, n_frames, channels, height, width, device=device, dtype=torch.float16)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).float()[None].repeat(batch, n_edges, 1, 1, 1)
        coords = coords + 0.4 * torch.randn_like(coords)

        ii = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
        jj = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.long)
        corr = AltCorrBlock(fmaps)

        coords_with_samples = coords.unsqueeze(dim=-2)
        reference = corr.corr_fn_reference(coords_with_samples, ii, jj)
        indexed = corr.corr_fn_indexed(coords_with_samples, ii, jj)

        torch.testing.assert_close(indexed, reference, atol=2e-4, rtol=1e-4)

        default = corr(coords, ii, jj)
        torch.testing.assert_close(default, reference.squeeze(dim=-1), atol=2e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
