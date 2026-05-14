"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from ...layers import (
    MLP,
    AttentionBlock,
    AttentionLayer,
    PositionEmbeddingSine,
    ResUpsampleBil,
)
from ...utils.coordinate import coords_grid
from ...utils.geometric import flat_interpolate
from ...utils.positional_embedding import generate_fourier_features


def orthonormal_init(num_tokens, dims):

    pe = torch.randn(num_tokens, dims)

    # Apply Gram-Schmidt process to make the matrix orthonormal
    for i in range(num_tokens):
        for j in range(i):
            # Subtract the projection of current row onto previous row
            pe[i] -= torch.dot(pe[i], pe[j]) * pe[j]

        # Normalize the current row
        pe[i] = F.normalize(pe[i], p=2, dim=0)

    return pe


class ListAdapter(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        for input_dim in input_dims:
            self.input_adapters.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, xs: torch.Tensor) -> list[torch.Tensor]:
        outs = [self.input_adapters[i](x) for i, x in enumerate(xs)]
        return outs


class CameraHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_params = 4

        self.aggregate1 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            use_bias=False,
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            use_bias=False,
        )
        self.latents_pos = nn.Parameter(torch.randn(1, self.num_params, hidden_dim), requires_grad=True)
        self.project = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=hidden_dim)
        self.out_pinhole = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=1)

    def fill_intrinsics(self, x):
        fx, fy, cx, cy = x.unbind(dim=-1)
        fx = torch.exp(fx)
        fy = torch.exp(fy)
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        diagonal = (self.shapes[0] ** 2 + self.shapes[1] ** 2) ** 0.5
        correction_tensor = torch.tensor(
            [0.7 * diagonal, 0.7 * diagonal, self.shapes[1], self.shapes[0]],
            device=x.device,
            dtype=x.dtype,
        )
        intrinsics = torch.stack([fx, fy, cx, cy], dim=1)
        intrinsics = correction_tensor.unsqueeze(0) * intrinsics
        return intrinsics

    def forward(self, features, cls_tokens, pos_embed) -> torch.Tensor:
        features = features.unbind(dim=-1)
        tokens = self.project(cls_tokens)

        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        tokens = self.aggregate1(tokens, pos_embed=latents_pos)
        tokens = self.aggregate2(tokens, pos_embed=latents_pos)

        x = self.out_pinhole(tokens.clone()).squeeze(-1)
        camera_intrinsics = self.fill_intrinsics(x)
        return camera_intrinsics

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes


class DepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths: int | list[int] = 4,
        camera_dim: int = 256,
        dropout: float = 0.0,
        kernel_size: int = 7,
        layer_scale: float = 1.0,
        out_dim: int = 1,
        use_norm=False,
        num_prompt_blocks=1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.camera_dim = camera_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.ups = nn.ModuleList([])
        self.depth_mlp = nn.ModuleList([])
        self.process_features = nn.ModuleList([])
        self.project_features = nn.ModuleList([])
        self.prompt_camera = nn.ModuleList([])
        mult = 2
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        for _ in range(4):
            self.prompt_camera.append(
                AttentionLayer(
                    num_blocks=num_prompt_blocks,
                    dim=hidden_dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale=-1.0,
                    context_dim=hidden_dim,
                    use_bias=False,
                )
            )

        for i, depth in enumerate(depths):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=max(1, 2 * i),
                    stride=max(1, 2 * i),
                    padding=0,
                )
            )

            self.ups.append(
                ResUpsampleBil(
                    current_dim,
                    output_dim=output_dim,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    kernel_size=kernel_size,
                    num_layers=depth,
                    use_norm=use_norm,
                )
            )

            depth_mlp = nn.Identity()
            if i == len(depths) - 1:
                depth_mlp = nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))

            self.depth_mlp.append(depth_mlp)

        self.confidence_mlp = nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))

        self.to_depth_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_confidence_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.to_confidence_hr = nn.Sequential(
            nn.Conv2d(output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_rays(self, rays):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=self.shapes, antialias=True)
        rays_embedding = rays_embedding / torch.norm(rays_embedding, dim=-1, keepdim=True).clip(min=1e-4)
        x, y, z = rays_embedding[..., 0], rays_embedding[..., 1], rays_embedding[..., 2]
        polar = torch.acos(z)
        x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
        azimuth = torch.atan2(y, x_clipped)
        rays_embedding = torch.stack([polar, azimuth], dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.hidden_dim,
            max_freq=max(self.shapes) // 2,
            use_log=True,
            cat_orig=False,
        )
        return rays_embedding

    def condition(self, feat, rays_embeddings):
        conditioned_features = [
            prompter(rearrange(feature, "b h w c -> b (h w) c"), rays_embeddings)
            for prompter, feature in zip(self.prompt_camera, feat)
        ]
        return conditioned_features

    def process(self, features_list, rays_embeddings):
        conditioned_features = self.condition(features_list, rays_embeddings)
        init_latents = self.to_latents(conditioned_features[0])
        init_latents = rearrange(init_latents, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]).contiguous()
        conditioned_features = [
            rearrange(x, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]).contiguous()
            for x in conditioned_features
        ]
        latents = init_latents

        out_features = []
        for i, up in enumerate(self.ups):
            latents = latents + self.process_features[i](conditioned_features[i + 1])
            latents = up(latents)
            out_features.append(latents)

        return out_features, init_latents

    def depth_proj(self, out_features):
        h_out, w_out = out_features[-1].shape[-2:]
        # aggregate output and project to depth
        for i, (layer, features) in enumerate(zip(self.depth_mlp, out_features)):
            out_depth_features = layer(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out_depth_features = F.interpolate(
                out_depth_features,
                size=(h_out, w_out),
                mode="bilinear",
                align_corners=True,
            )
            if i == len(self.depth_mlp) - 1:
                logdepth = out_depth_features

        logdepth = self.to_depth_lr(logdepth)
        logdepth = F.interpolate(logdepth, size=self.original_shapes, mode="bilinear", align_corners=True)
        logdepth = self.to_depth_hr(logdepth)
        return logdepth

    def confidence_proj(self, out_features):
        highres_features = out_features[-1].permute(0, 2, 3, 1)
        confidence = self.confidence_mlp(highres_features).permute(0, 3, 1, 2)
        confidence = self.to_confidence_lr(confidence)
        confidence = F.interpolate(confidence, size=self.original_shapes, mode="bilinear", align_corners=True)
        confidence = self.to_confidence_hr(confidence)
        return confidence

    def decode(self, out_features):
        logdepth = self.depth_proj(out_features)
        confidence = self.confidence_proj(out_features)
        return logdepth, confidence

    def forward(
        self,
        features: list[torch.Tensor],
        rays_hr: torch.Tensor,
        pos_embed,
        level_embed,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features[0].shape[0]

        rays_embeddings = self.embed_rays(rays_hr)
        features, proj_latents_16 = self.process(features, rays_embeddings)
        logdepth, logconf = self.decode(features)

        return logdepth, logconf, proj_latents_16


class Decoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
        self.test_gt_camera = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def run_camera(self, cls_tokens, features, pos_embed, original_shapes, rays_gt):
        H, W = original_shapes

        # camera layer
        intrinsics = self.camera_layer(features=features, cls_tokens=cls_tokens, pos_embed=pos_embed)
        B, N = intrinsics.shape
        device = intrinsics.device
        dtype = intrinsics.dtype

        id_coords = coords_grid(B, H, W, device=features.device, homogeneous=True)
        intrinsics_matrix_inverse = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)
        intrinsics_matrix_inverse[:, 0, 0] = 1.0 / intrinsics[:, 0]
        intrinsics_matrix_inverse[:, 1, 1] = 1.0 / intrinsics[:, 1]
        intrinsics_matrix_inverse[:, 0, 2] = -intrinsics[:, 2] / intrinsics[:, 0]
        intrinsics_matrix_inverse[:, 1, 2] = -intrinsics[:, 3] / intrinsics[:, 1]

        intrinsics_matrix = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)
        intrinsics_matrix[:, 0, 0] = intrinsics[:, 0]
        intrinsics_matrix[:, 1, 1] = intrinsics[:, 1]
        intrinsics_matrix[:, 0, 2] = intrinsics[:, 2]
        intrinsics_matrix[:, 1, 2] = intrinsics[:, 3]

        rays_pred = intrinsics_matrix_inverse @ id_coords.reshape(B, 3, -1)
        rays_pred = rays_pred.reshape(B, 3, H, W)
        rays_pred = rays_pred / torch.norm(rays_pred, dim=1, keepdim=True).clamp(min=1e-5)

        ### LEGACY CODE FOR TRAINING
        # if self.training and rays_gt is not None:
        #     prob = -1.0  # 0.8 * (1 - tanh(self.steps / 100000)) + 0.2
        #     where_use_gt_rays = torch.rand(B, 1, 1, device=device, dtype=dtype) < prob
        #     where_use_gt_rays = where_use_gt_rays.int()
        #     rays = rays_gt * where_use_gt_rays + rays_pred * (1 - where_use_gt_rays)

        rays = rays_pred if rays_gt is None else rays_gt
        rays = rearrange(rays, "b c h w -> b (h w) c")

        return intrinsics_matrix, rays

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        image_metas: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        B, C, H, W = inputs["image"].shape
        device = inputs["image"].device
        inputs["features"][0].dtype

        # get features in b n d format
        common_shape = inputs["features"][0].shape[1:3]

        # input shapes repeat shapes for each level, times the amount of the layers:
        features = self.input_adapter(inputs["features"])

        # positional embeddings, spatial and level
        level_embed = self.level_embeds.repeat(B, common_shape[0] * common_shape[1], 1, 1)
        level_embed = rearrange(level_embed, "b n l d -> b (n l) d")
        dummy_tensor = torch.zeros(B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False)
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(1, self.num_resolutions, 1)

        # get cls tokens projections
        camera_tokens = inputs["tokens"]
        camera_tokens = self.camera_token_adapter(camera_tokens)
        self.camera_layer.set_shapes((H, W))

        intrinsics, rays = self.run_camera(
            torch.cat(camera_tokens, dim=1),
            features=torch.stack(features, dim=-1).detach(),
            pos_embed=(pos_embed + level_embed).detach(),
            original_shapes=(H, W),
            rays_gt=inputs.get("rays", None),
        )

        # run bulk of the model
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))
        logdepth, logconfidence, depth_features = self.depth_layer(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )

        return {
            "radius": torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0),
            "depth_features": depth_features,
            "confidence": torch.exp(logconfidence.clip(min=-8.0, max=8.0)),
            "intrinsics": intrinsics,
            "rays": rays,
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def build(self, config):
        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        expansion = config["model"]["expansion"]
        num_heads = config["model"]["num_heads"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        layer_scale = config["model"]["layer_scale"]

        depth = config["model"]["pixel_decoder"]["depths"]
        self.downsample = 4
        depths_encoder = config["model"]["pixel_encoder"]["depths"]

        self.num_resolutions = len(depths_encoder)
        self.test_fixed_camera = False

        out_dim = config["model"]["pixel_decoder"]["out_dim"]
        kernel_size = config["model"]["pixel_decoder"].get("kernel_size", 7)

        self.slices_encoder = list(zip([d - 1 for d in depths_encoder], depths_encoder))
        input_dims = [input_dims[d - 1] for d in depths_encoder]

        # # adapt from encoder features, just project
        camera_dims = input_dims
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.camera_token_adapter = ListAdapter(camera_dims, hidden_dim)

        # # camera layer
        self.camera_layer = CameraHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )

        self.depth_layer = DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=96,
            num_resolutions=self.num_resolutions,
            layer_scale=layer_scale,
            out_dim=out_dim,
            kernel_size=kernel_size,
            num_prompt_blocks=1,
            use_norm=False,
        )
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            orthonormal_init(len(input_dims), hidden_dim).reshape(1, 1, len(input_dims), hidden_dim),
            requires_grad=False,
        )
