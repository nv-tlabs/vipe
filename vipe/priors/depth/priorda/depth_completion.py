# This file includes code originally from the PriorDA repository:
# https://github.com/SpatialVision/Prior-Depth-Anything
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

import re
import time
import warnings
from typing import Dict, Tuple

import torch

from .dav2 import build_backbone
from .utils import depth2disparity


class DepthCompletion(torch.nn.Module):
    @staticmethod
    def build(**kwargs):
        return DepthCompletion(**kwargs)

    def __init__(self, args, fmde_path, device=None):
        super().__init__()

        self.args = args
        self.K = args.K

        self.set_device(device)
        self.depth_model = self.init_depth_model(fmde_path)

    def set_device(self, device=None):
        if device is not None:
            self.device = device
            return

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def unify_format(
        self,
        images,
        sparse_depths,
        sparse_masks,
        cover_masks,
        prior_depths,
        geometric_depths,
    ):
        # Tune the shape of the tensors.
        if images.max() <= 1:
            images = images * 255
        if images.dtype != torch.uint8:
            images = images.to(torch.uint8)
        if len(sparse_depths.shape) == 4:
            sparse_depths = sparse_depths.squeeze(dim=1)
        if len(sparse_masks.shape) == 4:
            sparse_masks = sparse_masks.squeeze(dim=1)

        if prior_depths is not None and len(prior_depths.shape) == 4:
            prior_depths = prior_depths.squeeze(dim=1)
        if geometric_depths is not None and len(geometric_depths.shape) == 4:
            geometric_depths = geometric_depths.squeeze(dim=1)
        if cover_masks is not None and len(cover_masks.shape) == 4:
            cover_masks = cover_masks.squeeze(dim=1)

        # Move the tensors to the target device.
        images = images.to(self.device)
        sparse_depths, sparse_masks = sparse_depths.to(self.device), sparse_masks.to(self.device)

        return (
            images,
            sparse_depths,
            sparse_masks,
            cover_masks,
            prior_depths,
            geometric_depths,
        )

    @torch.no_grad()
    def preprocess(
        self,
        images: torch.Tensor,
        sparse_depths: torch.Tensor,
        sparse_masks: torch.Tensor,
        cover_masks=None,
        prior_depths=None,
        geometric_depths=None,
    ):
        """
        1. Unify the format of all the inputs.
        2. Obtain the model-predicted affine-invariant depth map.
        3. Convert the ground-truth depth to disparity.
        """
        (
            int_images,
            sparse_depths,
            sparse_masks,
            cover_masks,
            prior_depths,
            geometric_depths,
        ) = self.unify_format(
            images,
            sparse_depths,
            sparse_masks,
            cover_masks,
            prior_depths,
            geometric_depths,
        )

        # Preprocess pred_disparities.
        if geometric_depths is not None:
            warnings.warn("The geometric depth is provided by the user. ")
            pred_disparities = depth2disparity(geometric_depths)
        else:
            # heit = sparse_depths.shape[-2] // 14 * 14
            heit = 518
            if hasattr(self, "timer"):
                torch.cuda.synchronize()
                t0 = time.time()
            pred_disparities = self.depth_model(int_images, heit, device=self.device)
            if hasattr(self, "timer"):
                torch.cuda.synchronize()
                t1 = time.time()
                self.timer.append(t1 - t0)

            pred_disparities = pred_disparities.squeeze(1)

        # Preprocess sparse_depths and prior depths.
        sparse_disparities = depth2disparity(sparse_depths)
        prior_disparities = depth2disparity(prior_depths)

        return (
            pred_disparities,
            sparse_disparities,
            sparse_masks,
            cover_masks,
            prior_disparities,
        )

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        sparse_depths: torch.Tensor,
        sparse_masks: torch.Tensor,
        cover_masks=None,
        prior_depths=None,
        geometric_depths=None,
        pattern=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Processe input images and sparse depth information to produce completed depth maps.
        We use global alignment and KNN alignment to refine the depth predictions.

        Args:
            images (torch.Tensor): The input images.
            sparse_depths (torch.Tensor): The sparse depth information.
            sparse_masks (torch.Tensor): Indicating which points in the sparse depth are valid.
            cover_masks (torch.Tensor, optional): Indicating areas to be covered by prior depth.
            prior_depths (torch.Tensor, optional): Prior depth information for covering large areas.
            pattern (optional): Pattern for sampling sparse depth points.

        Returns:
            Dict[str, torch.Tensor]: Containing the processed data, including:
                - 'uncertainties': A tensor representing the uncertainty of the depth predictions.
                - 'scaled_preds': A tensor representing the scaled depth predictions.
                - 'global_preds': A tensor representing the globally aligned depth predictions.
        """

        (
            pred_disparities,
            sparse_disparities,
            sparse_masks,
            cover_masks,
            prior_disparities,
        ) = self.preprocess(
            images,
            sparse_depths,
            sparse_masks,
            cover_masks,
            prior_depths,
            geometric_depths,
        )

        output = {}

        # The masks denote the areas to be completed. Exclude the sparse points to accelerate.
        complete_masks = torch.ones_like(sparse_masks).to(torch.bool)
        complete_masks[sparse_masks] = False

        # ================================== Global Alignment.
        global_preds = self.ss_completer(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
        )
        output["global_preds"] = global_preds

        # ================================== KNN Alignments.
        if self.args.double_global:
            scaled_preds = global_preds.clone()
            scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
        else:
            # Scale the pred_disparities with KNN alignment.
            scaled_preds = self.kss_completer(
                sparse_disparities=sparse_disparities,
                pred_disparities=pred_disparities,
                sparse_masks=sparse_masks,
                K=self.K,
                complete_masks=complete_masks,
            )

        """ Notes: The sparse points have been covered in the ss-/kss-completer. """
        if cover_masks.sum() > 0:
            # To cover the large areas that have been known.
            scaled_preds[cover_masks] = prior_disparities[cover_masks]
        elif pattern is not None:
            assert not re.fullmatch(r"^cubic_\d+$", pattern)
            assert not re.fullmatch(r"^distance_\d+_\d+$", pattern)
        output["scaled_preds"] = scaled_preds

        # ================================== Process the Uncertainty map.
        cal_mask = global_preds > 0.0
        masked_scaled, scaled_global = scaled_preds[cal_mask], global_preds[cal_mask]
        uctn = torch.abs(masked_scaled - scaled_global) / scaled_global
        uncertainties = torch.zeros_like(scaled_preds, dtype=torch.float32)
        uncertainties[cal_mask] = uctn

        # If needed, normalize the Uncertainty.
        if self.args.normalize_confidence:
            uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
        output["uncertainties"] = uncertainties

        return output

    def init_depth_model(self, fmde_path):
        """We only implement @depth-anything-v2 here, you can replace it with other depth estimation models."""

        depth_model = build_backbone(depth_size=self.args.frozen_model_size, model_path=fmde_path).to(self.device)
        depth_model.freeze_network({"encoder", "decoder"})
        depth_model = depth_model.eval()

        return depth_model

    def calc_scale_shift(self, k_sparse_targets, k_pred_targets, currk_dists=None, knn=False):
        k_pred_targets += torch.rand(*k_pred_targets.shape, device=self.device) * 1e-5
        X = torch.stack([k_pred_targets, torch.ones_like(k_pred_targets, device=self.device)], dim=2)

        # To perform weights to the knn points.
        if knn > 0:
            k_sparse_targets, X = self.perform_weighted(k_sparse_targets, X, currk_dists)
        elif k_pred_targets.shape[0] > 1:
            k_sparse_targets = k_sparse_targets.unsqueeze(-1)

        solution = torch.linalg.lstsq(X, k_sparse_targets)
        scale, shift = solution[0][:, 0].squeeze(), solution[0][:, 1].squeeze()

        return scale, shift

    def perform_weighted(
        self, sparse_ori: torch.Tensor, pred_ori: torch.Tensor, dists: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform weighted operations on input tensors using distance-based weights. A diagonal
        matrix is created from the normalized weights and used to weight the inputs.

        Notes:
            - Weights are calculated as the inverse of the distances.
            - Weights are normalized to ensure they sum to 1.

        Args:
            sparse_ori (torch.Tensor): Sparse original map.
            pred_ori (torch.Tensor): Predicted map.
            dists (torch.Tensor): Distances used for weight calculation.

        Returns:
            Tuple: Containing two tensors:
                - sparse_weighted: The weighted version of the sparse original map.
                - pred_weighted: The weighted version of the predicted map.
        """

        weights = 1 / dists
        wsum = weights.sum(dim=1, keepdim=True)
        weights = weights / wsum
        W = torch.diag_embed(weights)

        pred_weighted = W @ pred_ori
        sparse_weighted = W @ sparse_ori.unsqueeze(-1)
        return sparse_weighted, pred_weighted

    def knn_aligns(
        self, sparse_disparities, pred_disparities, sparse_masks, complete_masks, K
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform K-Nearest Neighbors (KNN) alignment on sparse and predicted disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            complete_masks (torch.Tensor): Indicating which points in the map to be completed.
            K (int): The number of nearest neighbors to find for each map point.

        Returns:
            Tuple: Containing three tensors:
                - dists: The Euclidean distances from each sparse point to its K nearest neighbors.
                - k_sparse_targets: Disparities of the K nearest neighbors from the sparse data.
                - k_pred_targets: Disparities of the K nearest neighbors from the predicted data.
        """

        # Coordinates are processed to ensure compatibility with the KNN function.
        batch_sparse = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float()  # [N, 3] (b, x, y)
        batch_complete = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float()  # [M, 3] (b, x, y)

        _batch_x, _batch_y = (
            batch_sparse[:, 0].contiguous(),
            batch_complete[:, 0].contiguous(),
        )
        x, y = batch_sparse[:, -2:].contiguous(), batch_complete[:, -2:].contiguous()

        # Use `vipe_ext` to find the K nearest neighbors.
        import vipe_ext as _C

        _, inds = _C.utils_ext.nearest_neighbours(y, x, K)
        knn_indices = inds.view(-1, K)  # [M, K]

        # Use `torch_cluster.knn` to find K nearest neighbors.
        # knn_map = torch_cluster.knn(x=x, y=y, k=K, batch_x=batch_x, batch_y=batch_y) # [2, M * K]
        # knn_indices = knn_map[1, :].view(-1, K)

        k_sparse_targets = sparse_disparities[sparse_masks][knn_indices]
        k_pred_targets = pred_disparities[sparse_masks][knn_indices]

        knn_coords = x[knn_indices]
        expanded_complete_points = y.unsqueeze(dim=1).repeat(1, K, 1)
        dists = torch.norm(expanded_complete_points - knn_coords, dim=2)

        return dists, k_sparse_targets, k_pred_targets

    def kss_completer(self, sparse_disparities, pred_disparities, complete_masks, sparse_masks, K=5) -> torch.Tensor:
        """
        Perform K-Nearest Neighbors (KNN) interpolation to complete sparse disparities.Use a batch-oriented
        implementation of KNN interpolation to complete the sparse disparities. We leverages "torch_cluster.knn"
        for acceleration and GPU memory efficiency.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map.
            pred_disparities (torch.Tensor): Dredicted disparities for sparse map points.
            complete_masks (torch.Tensor): Indicating which points in the complete map are valid.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
            K (int): The number of nearest neighbors to use for interpolation. Defaults to 5.

        Returns:
            The completed disparities, interpolated from the nearest neighbors.
        """

        # Use `knn_aligns` to find the K nearest neighbors and calculate distances.
        bottomk_dists, k_sparse_targets, k_pred_targets = self.knn_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
            K=K,
            complete_masks=complete_masks,
        )

        scaled_preds = torch.zeros_like(sparse_disparities, device=self.device, dtype=torch.float32)
        scale, shift = self.calc_scale_shift(
            k_sparse_targets=k_sparse_targets,
            k_pred_targets=k_pred_targets,
            currk_dists=bottomk_dists,
            knn=True,
        )

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds[complete_masks] = pred_disparities[complete_masks] * scale + shift
        # The completed disparities are computed by combining the scaled predictions and the original sparse disparities.
        scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
        return scaled_preds

    def global_aligns(self, sparse_disparities, pred_disparities, sparse_masks) -> Tuple[torch.Tensor, ...]:
        """
        Perform global alignment on sparse and predicted disparities. Extract the valid disparities from
        both sparse and predicted map based on the sparse masks.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            Tuple[torch.Tensor]: Containing two tensors:
                - k_sparse_targets: The valid disparities from the sparse map.
                - k_pred_targets: The valid disparities from the predicted map.
        """

        # The valid disparities are extracted and unsqueezed to maintain consistent dimensions.
        k_sparse_targets = sparse_disparities[sparse_masks].unsqueeze(dim=0)
        k_pred_targets = pred_disparities[sparse_masks].unsqueeze(dim=0)

        return k_sparse_targets, k_pred_targets

    def ss_completer(self, sparse_disparities, pred_disparities, sparse_masks) -> torch.Tensor:
        """
        Complete sparse disparities using a simple scaling and shifting approach. Perform a global
        alignment of the sparse and predicted disparities, then applies a scaling and shifting
        transformation to complete the sparse disparities.

        Args:
            sparse_disparities (torch.Tensor): Disparities for sparse map points.
            pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
            sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.

        Returns:
            The completed disparities, computed by scaling and shifting the predicted disparities.
        """

        # Use `global_aligns` to extract valid disparities.
        k_sparse_targets, k_pred_targets = self.global_aligns(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks,
        )

        scale, shift = self.calc_scale_shift(k_sparse_targets=k_sparse_targets, k_pred_targets=k_pred_targets)

        # Apply scaling and shifting to the predicted disparities based on the nearest neighbors.
        scaled_preds = pred_disparities * scale + shift
        return scaled_preds
