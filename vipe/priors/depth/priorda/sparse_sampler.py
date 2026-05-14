# This file includes code originally from the PriorDA repository:
# https://github.com/SpatialVision/Prior-Depth-Anything
# Licensed under the Apache-2.0 License. See THIRD_PARTY_LICENSES.md for details.

import re
import warnings
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import torch_cluster
except ImportError:
    torch_cluster = None


class SparseSampler:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.min_depth = 0.0001  # We always filter out depth <= 0.

    def __call__(self, image, prior, geometric=None, pattern=None, K=5, prior_cover=False) -> Dict[str, torch.Tensor]:
        """
        1. Handles the loading and preprocessing of image and prior depth data.
        2. Samples sparse depth points based on the provided pattern or prior depth information.

        Args:
            image:
                The path of the image (readable for Image.open()) or a tensor/array representing the image.
                Shape should be [H, W, 3] with values in the range [0, 255].
            prior:
                The path of the prior depth (e.g., '*.png') or a tensor/array representing the prior depth.
                Shape should be [H, W] with type float32.
            geometric (optional):
                The path of the geometric depth (e.g., '*.png') or a tensor/array representing the geometric depth.
                Shape should be [H, W] with type float32.
            pattern (optional):
                Pattern for sampling sparse depth points. If None, prior depth is used.
            K (int):
                The minimum number of known points required. Defaults to 5.
            prior_cover (bool, optional):
                Determine if the prior depth should be used to cover sparse points. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Containing the processed data, including:
                - 'rgb': The loaded RGB image.
                - 'prior_depth': The loaded prior depth.
                - 'sparse_depth': The sampled sparse depth.
                - 'sparse_mask': Indicating valid sparse depth points.
                - 'cover_mask': Indicating covered points based on prior depth.
        """

        assert pattern is None or isinstance(pattern, str)
        data = {}

        # Load RGB image.
        if isinstance(image, str):
            if image.endswith(".npy"):
                np_image = np.load(image)
                ts_image = torch.from_numpy(np_image).permute(2, 0, 1).to(torch.uint8)
            else:
                pil_image = Image.open(image)
                np_image = np.asarray(pil_image)
                ts_image = torch.from_numpy(np_image.copy()).permute(2, 0, 1).to(torch.uint8)
        elif isinstance(image, torch.Tensor):
            np_image = image.cpu().numpy()
            ts_image = image.cpu().permute(2, 0, 1).to(torch.uint8)
        elif isinstance(image, np.ndarray):
            np_image = image.copy()
            ts_image = torch.from_numpy(image).permute(2, 0, 1).to(torch.uint8)
        data["rgb"] = ts_image.unsqueeze(0)

        # Load prior depth.
        if isinstance(prior, str):
            if prior.endswith(".npy"):
                np_prior = np.load(prior)
                ts_prior = torch.from_numpy(np_prior)
            else:
                # The format should be compatible with Image.open
                pil_prior = Image.open(prior)
                np_prior = np.asarray(pil_prior).astype(np.float32)
                ts_prior = torch.from_numpy(np_prior.copy())
        elif isinstance(prior, np.ndarray):
            ts_prior = torch.from_numpy(np_prior)
        elif isinstance(prior, torch.Tensor):
            ts_prior = prior.cpu()
        data["prior_depth"] = ts_prior.unsqueeze(0).unsqueeze(0)

        # Load geometric depth.
        if geometric is not None:
            if isinstance(geometric, str):
                if geometric.endswith(".npy"):
                    np_geometric = np.load(geometric)
                    ts_geometric = torch.from_numpy(np_geometric)
                else:
                    # The format should be compatible with Image.open
                    pil_geometric = Image.open(geometric)
                    np_geometric = np.asarray(pil_geometric).astype(np.float32)
                    ts_geometric = torch.from_numpy(np_geometric.copy())
            elif isinstance(geometric, np.ndarray):
                ts_geometric = torch.from_numpy(np_geometric)
            elif isinstance(geometric, torch.Tensor):
                ts_geometric = geometric.cpu()
            data["geometric_depth"] = ts_geometric.unsqueeze(0).unsqueeze(0)

        # Sample the points manually if `pattern` is provided, otherwise use prior.
        if pattern or ts_prior.shape[-2:] != ts_image.shape[-2:]:
            sparse_depth, sparse_mask, cover_mask = self.get_sparse_depth(
                image=np_image, prior=ts_prior, pattern=pattern
            )

            # We do not implement hybrid-pattern here.
            if ts_prior.shape[-2:] != ts_image.shape[-2:]:
                assert pattern is None, "When testing with low-res prior, please set `pattern` to None"

            """ Force to keep the prior in the condition. """
            if prior_cover:
                assert ts_prior.shape[-2:] == ts_image.shape[-2:]
                cover_mask = ts_prior > self.min_depth
        else:
            """
            If `pattern` is None, the value of `prior_cover` does not
            matter and all prior will cover in kss_completer.
            """
            sparse_depth = ts_prior.clone()
            sparse_mask = sparse_depth > self.min_depth
            cover_mask = torch.zeros_like(sparse_mask)

        data["sparse_depth"] = sparse_depth.unsqueeze(0).unsqueeze(0)
        data["sparse_mask"] = sparse_mask.unsqueeze(0).unsqueeze(0)
        data["cover_mask"] = cover_mask.unsqueeze(0).unsqueeze(0)

        # Check samples and move points to the target device.
        if sparse_mask.sum() < K:
            raise ValueError("There are not enough known points.")
        data = {k: v.to(self.device) for k, v in data.items() if v is not None}
        return data

    def get_sparse_depth(self, image, prior, pattern=None):
        height, width, c = image.shape[-3:]
        low_height, low_width = prior.shape[-2:]

        if height != low_height or width != low_width:
            pattern = "downscale_"
            # print("============================ Testing with known low depth. ============================")
        # else:
        #     print(f"============================ Testing with {pattern}. ============================")

        if pattern.isdigit():
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            num_sample = int(pattern)

            idx_nnz = torch.nonzero(prior.view(-1) > self.min_depth, as_tuple=False)
            num_idx = len(idx_nnz)
            if num_idx < num_sample:
                warnings.warn(f"Aiming to sample {num_sample} points, but only {num_idx} valid points in the map.")

            idx_sample = torch.randperm(num_idx)[:num_sample]
            idx_nnz = idx_nnz[idx_sample[:]]

            sparse_mask = torch.zeros((height * width), dtype=torch.bool)
            sparse_mask[idx_nnz] = True
            sparse_mask = sparse_mask.view((height, width))

            sparse_depth = prior * sparse_mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)

        elif re.fullmatch(r"^downscale_\d*$", pattern):
            prior = prior.unsqueeze(0)

            if pattern != "downscale_":
                prior_mask = prior > self.min_depth

                factor = pattern.split("_")[-1]
                factor = int(factor)

                # Fill in the blank areas in the image.
                filled_depth = self.interpolate_depths(prior, prior_mask, ~prior_mask)

                # Downscale the prior depth map.
                low_height, low_width = height // factor, width // factor
                prior = F.interpolate(
                    filled_depth.unsqueeze(0),
                    size=(low_height, low_width),
                    mode="bilinear",
                    align_corners=True,
                )
                prior = prior.squeeze()

            # Insert the low-res prior depth map into the higher one.
            s_height, s_width = height / low_height, width / low_width
            idx_height, idx_width = (
                (s_height * torch.arange(low_height)).long(),
                (s_width * torch.arange(low_width)).long(),
            )

            down_mask = torch.zeros((height, width), dtype=torch.bool)
            down_mask[..., idx_height[:, None], idx_width] = True

            sparse_depth = torch.zeros((height, width), dtype=torch.float32)
            sparse_depth[down_mask] = prior.flatten()

            sparse_mask = sparse_depth > self.min_depth
            # Filter the sparse mask with valid mask if sampled manually.
            if pattern != "downscale_":
                sparse_mask &= prior_mask.squeeze(0)
            sparse_depth = sparse_depth * sparse_mask.type_as(sparse_depth)
            cover_mask = torch.zeros_like(sparse_mask)

        elif re.fullmatch(r"^cubic_\d+$", pattern):
            clen = pattern.split("_")[-1]
            clen = int(clen)

            # Sample a cube in the image based on top-lerf coords and clen
            cubic_mask = torch.ones_like(prior, dtype=torch.bool)
            height_upper, width_upper = height - clen, width - clen
            h = np.random.randint(0, height_upper)
            w = np.random.randint(0, width_upper)
            cubic_mask[h : h + clen, w : w + clen] = False
            cover_mask = torch.logical_and(cubic_mask, prior > self.min_depth)

            vacant_depth = prior * cover_mask.type_as(prior)
            sparse_depth, sparse_mask, _ = self.get_sparse_depth(image, vacant_depth, pattern="2000")

        elif re.fullmatch(r"^distance_\d+_\d+$", pattern):
            # The lower bound and high-bound of the interval that
            # we want to keep the depth known
            low_dist, high_dist = pattern.split("_")[-2:]
            low_dist, high_dist = int(low_dist), int(high_dist)

            # Only keep depth within the range --- (low_dist, high_dist)
            cover_mask = torch.logical_and(
                (prior > self.min_depth),
                torch.logical_and(prior > low_dist, prior < high_dist),
            )

            range_depth = prior * cover_mask.type_as(prior)
            sparse_depth, sparse_mask, _ = self.get_sparse_depth(image, range_depth, pattern="2000")

        elif pattern == "sift" or pattern == "orb":
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            assert image is not None

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if pattern == "sift":
                detector = cv2.SIFT.create()
            elif pattern == "orb":
                detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            else:
                raise NotImplementedError

            keypoints = detector.detect(gray)
            mask = torch.zeros([height, width])

            if len(keypoints) < 20:
                return self.get_sparse_depth(image=image, prior=prior, pattern="2000")

            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                mask[x, y] = 1.0

            train_sfm_max_dropout_rate = 0.0
            if train_sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, train_sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(mask)
                mask_keep = torch.bernoulli(mask_keep)

                mask = mask * mask_keep

            sparse_mask = (mask * (prior > self.min_depth).type_as(prior)).to(torch.bool)
            sparse_depth = prior * mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)

        elif re.fullmatch(r"^LiDAR_\d+$", pattern):
            # Adapted from OMNI-DC, available at https://github.com/princeton-vl/OMNI-DC
            w_c = 0.5 * width
            h_c = 0.5 * height
            focal = height

            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            dep_np = prior.numpy()

            # sample the lidar patterns
            pitch_max = 0.5
            pitch_min = -0.5
            num_lines = int(pattern.split("_")[1])
            num_horizontal_points = 200

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi / 2.1, np.pi / 2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid)  # assume the distace is unit
            z = np.sqrt(1.0 - x**2 - y**2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1)  # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8)  # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[
                :,
                ((points_2D[0] >= 0) & (points_2D[0] < width) & (points_2D[1] >= 0) & (points_2D[1] < height)),
            ]

            mask = np.zeros([height, width])
            mask[points_2D_valid[1], points_2D_valid[0]] = 1.0
            # only keep the orginal valid regions
            mask = mask * (dep_np > self.min_depth).astype(float)

            sparse_mask = torch.from_numpy(mask).to(torch.bool)
            sparse_depth = prior * sparse_mask.type_as(prior)
            cover_mask = torch.zeros_like(sparse_mask)

        else:
            raise NotImplementedError(
                (
                    "'pattern' should be in format of ['^LiDAR_\d+$', 'sift', 'orb', '^cubic_\d+$', '^distance_\d+_\d+$',"
                    "'^downscale_\d*$', '(int)'], but the provided 'pattern' is -- '{}'".format(pattern)
                )
            )

        return sparse_depth, sparse_mask, cover_mask

    def interpolate_depths(self, sparse_depths, sparse_masks, complete_masks):
        if torch_cluster is None:
            raise ImportError("torch_cluster is required for SparseSampler.interpolate_depths")

        known_points = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float()  # [N, 3] (b, x, y)
        complete_depths = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float()  # [M, 3] (b, x, y)

        batch_x, batch_y = (
            known_points[:, 0].contiguous(),
            complete_depths[:, 0].contiguous(),
        )
        x, y = known_points[:, -2:].contiguous(), complete_depths[:, -2:].contiguous()

        knn_map = torch_cluster.knn(x=x, y=y, k=5, batch_x=batch_x, batch_y=batch_y)  # [2, M * K]
        knn_indices = knn_map[1, :].view(-1, 5)
        knn_depths = sparse_depths[sparse_masks][knn_indices]

        filled_depths = torch.zeros_like(sparse_depths)
        filled_depths[sparse_masks] = sparse_depths[sparse_masks]
        filled_depths[complete_masks] = knn_depths.mean(dim=-1)

        return filled_depths
