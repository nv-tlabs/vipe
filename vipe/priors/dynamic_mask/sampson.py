"""Sampson reprojection-error scoring of optical flow vs. fitted fundamental matrix."""

import cv2
import numpy as np


def fit_fundamental_and_sampson(
    pts1: np.ndarray,
    pts2: np.ndarray,
    valid_mask: np.ndarray,
    H: int,
    W: int,
    max_points: int = 5000,
    use_ransac: bool = True,
    ransac_threshold: float = 0.01,
    ransac_confidence: float = 0.99,
    ransac_max_iters: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fit a fundamental matrix on (pts1, pts2) restricted to `valid_mask`,
    then return a per-pixel Sampson error map of shape (H, W).

    Coordinates of `pts1` and `pts2` are assumed already normalized to
    [-1, 1] x [-1, 1] (which makes RANSAC thresholds resolution-independent).
    Pixels outside the consistency mask are zeroed in the output.
    """
    flat_mask = valid_mask.reshape(-1).astype(bool)
    if flat_mask.sum() < 8:
        return np.zeros((H, W), dtype=np.float32)

    p1 = pts1[flat_mask]
    p2 = pts2[flat_mask]

    if len(p1) > max_points:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(p1), size=max_points, replace=False)
        p1 = p1[idx]
        p2 = p2[idx]

    if use_ransac:
        F, _ = cv2.findFundamentalMat(
            p1.astype(np.float32),
            p2.astype(np.float32),
            cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iters,
        )
    else:
        F, _ = cv2.findFundamentalMat(p1.astype(np.float32), p2.astype(np.float32), cv2.FM_LMEDS)

    if F is None:
        return np.zeros((H, W), dtype=np.float32)

    F = F.astype(np.float32)
    ones = np.ones((pts1.shape[0], 1), dtype=np.float32)
    x1h = np.concatenate([pts1.astype(np.float32), ones], axis=1)
    x2h = np.concatenate([pts2.astype(np.float32), ones], axis=1)
    Fx1 = x1h @ F.T
    Fx2 = x2h @ F
    num = (np.sum(x2h * Fx1, axis=1)) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + 1e-8
    err = num / den
    err = err.reshape(H, W) * valid_mask.astype(np.float32)
    return err.astype(np.float32)


def sampson_error_for_frame(
    fwd_flow: np.ndarray | None,
    fwd_mask: np.ndarray | None,
    bwd_flow: np.ndarray | None,
    bwd_mask: np.ndarray | None,
    H: int,
    W: int,
    **kwargs,
) -> np.ndarray:
    """Compute the per-pixel motion-likelihood error for a single frame,
    by taking the max of forward and backward Sampson errors (sqrt'd to
    behave more linearly), as in dyn_mask.py.

    Either direction may be None (e.g., the very first / last frame).
    """
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    xn = 2 * (xx + 0.5) / W - 1
    yn = 2 * (yy + 0.5) / H - 1
    pts1 = np.stack([xn.ravel(), yn.ravel()], axis=-1).astype(np.float32)

    err_list: list[np.ndarray] = []

    if fwd_flow is not None and fwd_mask is not None:
        flow_n = np.stack(
            [
                2.0 * fwd_flow[..., 0] / (W - 1),
                2.0 * fwd_flow[..., 1] / (H - 1),
            ],
            axis=-1,
        ).astype(np.float32)
        pts2 = (pts1 + flow_n.reshape(-1, 2)).astype(np.float32)
        err_list.append(fit_fundamental_and_sampson(pts1, pts2, fwd_mask, H, W, **kwargs))

    if bwd_flow is not None and bwd_mask is not None:
        flow_n = np.stack(
            [
                2.0 * bwd_flow[..., 0] / (W - 1),
                2.0 * bwd_flow[..., 1] / (H - 1),
            ],
            axis=-1,
        ).astype(np.float32)
        pts2 = (pts1 + flow_n.reshape(-1, 2)).astype(np.float32)
        err_list.append(fit_fundamental_and_sampson(pts1, pts2, bwd_mask, H, W, **kwargs))

    if not err_list:
        return np.zeros((H, W), dtype=np.float32)
    err = np.maximum.reduce(err_list)
    return np.sqrt(err.clip(min=0)).astype(np.float32)
