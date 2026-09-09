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

    def test_indexed_altcorr_matches_reference_repeatedly(self):
        # Regression guard for a missing __syncthreads() in altcorr_forward_kernel /
        # altcorr_backward_kernel: both write x2s[tid]/y2s[tid] then read x2s[k1]/y2s[k1]
        # for k1 that can belong to another thread, with no barrier between the write and
        # that cross-thread read. This was silently safe only because BLOCK_HW == 32 (one
        # warp, so SIMT lockstep made the ordering incidental); once BLOCK_HW spans more
        # than one warp (see BLOCK_H in altcorr_kernel.cu) the two warps are
        # independently scheduled and the missing barrier becomes a real race, producing
        # a handful of wrong pixels that differ from run to run on an otherwise
        # fixed-seed problem. A single run of the parity test above cannot distinguish
        # "flaky" from "deterministically correct"; repeating it does.
        for _ in range(10):
            self.test_indexed_altcorr_matches_corr_layer_reference()

    def test_altcorr_backward_matches_pre_retile_golden_values(self):
        # Golden-value regression test for the altcorr_kernel.cu retiling (4x8/32-thread
        # blocks -> 8x8/64-thread blocks, see the file's header comment): asserts the
        # retiled kernel reproduces the exact forward/backward outputs of the
        # pre-retile (stock DROID-SLAM) kernel on a fixed-seed problem.
        #
        # Note: this deliberately does NOT check the backward gradient against finite
        # differences. Doing so surfaces a large, pre-existing discrepancy between
        # altcorr_backward's analytic gradient and the numerical gradient on this same
        # problem -- reproducible bit-for-bit with the *stock*, pre-retile kernel too,
        # i.e. it predates and is unrelated to this port. Fixing that is out of scope
        # here; this test only guards that the retile did not change kernel behavior.
        device = torch.device("cuda")
        batch, n_frames, channels, height, width = 1, 2, 8, 16, 16
        radius = 2

        torch.manual_seed(2028)
        fmaps = torch.randn(
            batch, n_frames, channels, height, width, device=device, dtype=torch.float32, requires_grad=True
        )
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).float()[None].repeat(batch, 1, 1, 1, 1)
        coords = coords + 0.3 * torch.randn_like(coords)
        ii = torch.tensor([0], device=device, dtype=torch.long)
        jj = torch.tensor([1], device=device, dtype=torch.long)

        corr = AltCorrBlock(fmaps, radius=radius).corr_fn_reference(coords.unsqueeze(dim=-2), ii, jj)
        loss = corr.sum()
        (grad_fmaps,) = torch.autograd.grad(loss, (fmaps,))

        # Captured from the pre-retile (stock) altcorr_kernel.cu on this exact,
        # fixed-seed problem.
        golden_corr_sum = -98.19378662109375
        # Sample indices were selected (offline) for run-to-run stability under repeated
        # atomic-add accumulation with the same kernel; a handful of other entries are
        # inherently noisy under reduction-order changes and are intentionally excluded.
        golden_grad_sample_idx = [2662, 2678, 2663, 2374, 2375, 2359, 2390]
        golden_grad_sample = [
            -4.1590681076049805,
            -3.963200569152832,
            -3.891901731491089,
            3.8366141319274902,
            3.746760845184326,
            3.4687509536743164,
            3.456289291381836,
        ]
        torch.testing.assert_close(corr.sum(), torch.tensor(golden_corr_sum, device=device), atol=1e-3, rtol=1e-4)
        grad_flat = grad_fmaps.reshape(-1)
        for idx, expected in zip(golden_grad_sample_idx, golden_grad_sample):
            torch.testing.assert_close(grad_flat[idx], torch.tensor(expected, device=device), atol=1e-3, rtol=1e-3)
        self.assertTrue(torch.isfinite(grad_fmaps).all())

    def test_altcorr_index_forward_is_finite_for_fp16(self):
        device = torch.device("cuda")
        batch, n_frames, channels, height, width = 1, 3, 128, 32, 40

        fmaps = torch.randn(batch, n_frames, channels, height, width, device=device, dtype=torch.float16)
        corr = AltCorrBlock(fmaps)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).float()[None].repeat(batch, 1, 1, 1, 1)
        coords = coords + 0.4 * torch.randn_like(coords)
        ii = torch.tensor([0], device=device, dtype=torch.long)
        jj = torch.tensor([1], device=device, dtype=torch.long)

        out = corr.corr_fn_indexed(coords.unsqueeze(dim=-2), ii, jj)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
