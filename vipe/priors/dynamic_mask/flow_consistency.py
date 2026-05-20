"""Forward-backward optical-flow consistency masks."""

import torch
import torch.nn.functional as F


def _warp_with_flow(field: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `field` by `flow`.

    field: (B, C, H, W). flow: (B, 2, H, W) in pixel units (dx, dy).
    For each output pixel (y, x) returns field at (y + dy, x + dx),
    using bilinear sampling.
    """
    B, _, H, W = flow.shape
    device = flow.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow.dtype),
        torch.arange(W, device=device, dtype=flow.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
    sample = base_grid + flow
    # Convert pixel coords to normalized [-1, 1] grid_sample format.
    sample_x = 2.0 * sample[:, 0] / (W - 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / (H - 1) - 1.0
    grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def forward_backward_consistency(
    fwd_flow: torch.Tensor,
    bwd_flow: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute consistency masks for both directions of a flow pair.

    A pixel x in frame A is considered consistent w.r.t. its forward flow
    f_AB iff   |f_AB(x) + f_BA(x + f_AB(x))|^2 < alpha * (|f_AB(x)|^2 +
    |f_BA(x + f_AB(x))|^2) + beta.

    Args:
        fwd_flow: (B, 2, H, W) flow from A -> B.
        bwd_flow: (B, 2, H, W) flow from B -> A.

    Returns:
        fwd_mask: (B, H, W) bool, True where A's forward flow is consistent.
        bwd_mask: (B, H, W) bool, True where B's backward flow is consistent.
    """
    bwd_warped_to_a = _warp_with_flow(bwd_flow, fwd_flow)  # B->A flow sampled at x + f_AB(x)
    cycle_a = fwd_flow + bwd_warped_to_a
    cycle_a_sq = (cycle_a**2).sum(dim=1)
    fwd_mag_sq = (fwd_flow**2).sum(dim=1)
    bwd_mag_sq_at_a = (bwd_warped_to_a**2).sum(dim=1)
    fwd_mask = cycle_a_sq < alpha * (fwd_mag_sq + bwd_mag_sq_at_a) + beta

    fwd_warped_to_b = _warp_with_flow(fwd_flow, bwd_flow)
    cycle_b = bwd_flow + fwd_warped_to_b
    cycle_b_sq = (cycle_b**2).sum(dim=1)
    bwd_mag_sq = (bwd_flow**2).sum(dim=1)
    fwd_mag_sq_at_b = (fwd_warped_to_b**2).sum(dim=1)
    bwd_mask = cycle_b_sq < alpha * (bwd_mag_sq + fwd_mag_sq_at_b) + beta

    return fwd_mask, bwd_mask
