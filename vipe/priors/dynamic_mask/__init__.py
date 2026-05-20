"""Automatic dynamic-object mask estimation via optical-flow motion scores.

Algorithm overview (after Sundaram-style fwd/bwd consistency + Sampson-error scoring):

1. For every consecutive frame pair compute forward and backward optical flow
   with a class-agnostic flow model (default: SEA-RAFT).
2. Build per-pixel forward/backward consistency masks using the standard
   |fwd + bwd(warp)|^2 < alpha * (|fwd|^2 + |bwd(warp)|^2) + beta criterion.
3. For every frame fit a fundamental matrix between consistent correspondences
   (using RANSAC), then evaluate the Sampson reprojection error of every pixel.
   The error indicates how much that pixel's motion deviates from the
   epipolar geometry of the dominant (background) flow.
4. For every tracked instance, average the per-pixel Sampson error over its
   support, weighting by flow confidence. Frames where the error is too small
   (i.e. the object happens to move with the camera) are filtered out so static
   chunks don't dilute the score.
5. Select instances whose final motion score exceeds a fraction of the maximum
   as dynamic. Their union is the final dynamic mask.
"""

from .pipeline import DynamicMaskPipeline


__all__ = ["DynamicMaskPipeline"]
