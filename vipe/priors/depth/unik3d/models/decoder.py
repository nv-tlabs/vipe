"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.init import trunc_normal_

from ..utils.coordinate import coords_grid
from ..utils.geometric import flat_interpolate
from ..utils.misc import get_params
from ..utils.positional_embedding import generate_fourier_features
from ..utils.sht import rsh_cart_3
from .layers import MLP, AttentionBlock, AttentionLayer, GradChoker, PositionEmbeddingSine, ResUpsampleBil


def orthonormal_init(num_tokens, dims):
    pe = torch.randn(num_tokens, dims)

    # Apply Gram-Schmidt process to make the matrix orthonormal
    # Awful loop..
    for i in range(num_tokens):
        for j in range(i):
            pe[i] -= torch.dot(pe[i], pe[j]) * pe[j]
        pe[i] = F.normalize(pe[i], p=2, dim=0)

    return pe


class ListAdapter(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        self.checkpoint = True
        for input_dim in input_dims:
            self.input_adapters.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = [self.input_adapters[i](x) for i, x in enumerate(xs)]
        return outs


class AngularModule(nn.Module):
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
        self.pin_params = 3
        self.deg1_params = 3
        self.deg2_params = 5
        self.deg3_params = 7
        self.num_params = self.pin_params + self.deg1_params + self.deg2_params + self.deg3_params

        self.aggregate1 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.latents_pos = nn.Parameter(torch.randn(1, self.num_params, hidden_dim), requires_grad=True)
        self.in_features = nn.Identity()

        self.project_pin = nn.Linear(hidden_dim, self.pin_params * hidden_dim, bias=False)
        self.project_deg1 = nn.Linear(hidden_dim, self.deg1_params * hidden_dim, bias=False)
        self.project_deg2 = nn.Linear(hidden_dim, self.deg2_params * hidden_dim, bias=False)
        self.project_deg3 = nn.Linear(hidden_dim, self.deg3_params * hidden_dim, bias=False)

        self.out_pinhole = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=1)
        self.out_deg1 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg2 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg3 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)

    def fill_intrinsics(self, x):
        hfov, cx, cy = x.unbind(dim=-1)
        hfov = torch.sigmoid(hfov - 1.1)  # 1.1 magic number s.t hfov = pi/2 for x=0
        ratio = self.shapes[0] / self.shapes[1]
        vfov = hfov * ratio
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        correction_tensor = torch.tensor(
            [2 * torch.pi, 2 * torch.pi, self.shapes[1], self.shapes[0]],
            device=x.device,
            dtype=x.dtype,
        )

        intrinsics = torch.stack([hfov, vfov, cx, cy], dim=1)
        intrinsics = correction_tensor.unsqueeze(0) * intrinsics
        return intrinsics

    # @profile_method(verbose=True)
    def forward(self, cls_tokens) -> torch.Tensor:
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)

        pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens = cls_tokens.chunk(4, dim=1)
        pin_tokens = rearrange(self.project_pin(pin_tokens), "b n (h c) -> b (n h) c", h=self.pin_params)
        deg1_tokens = rearrange(self.project_deg1(deg1_tokens), "b n (h c) -> b (n h) c", h=self.deg1_params)
        deg2_tokens = rearrange(self.project_deg2(deg2_tokens), "b n (h c) -> b (n h) c", h=self.deg2_params)
        deg3_tokens = rearrange(self.project_deg3(deg3_tokens), "b n (h c) -> b (n h) c", h=self.deg3_params)
        tokens = torch.cat([pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens], dim=1)

        tokens = self.aggregate1(tokens, pos_embed=latents_pos)
        tokens = self.aggregate2(tokens, pos_embed=latents_pos)

        tokens_pinhole, tokens_deg1, tokens_deg2, tokens_deg3 = torch.split(
            tokens,
            [self.pin_params, self.deg1_params, self.deg2_params, self.deg3_params],
            dim=1,
        )
        x = self.out_pinhole(tokens_pinhole).squeeze(-1)
        d1 = self.out_deg1(tokens_deg1)
        d2 = self.out_deg2(tokens_deg2)
        d3 = self.out_deg3(tokens_deg3)

        camera_intrinsics = self.fill_intrinsics(x)
        return camera_intrinsics, torch.cat([d1, d2, d3], dim=1)

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes


class RadialModule(nn.Module):
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
        num_prompt_blocks: int = 1,
        use_norm: bool = False,
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
        self.out = nn.ModuleList([])
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
            depth_mlp = (
                nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))
                if i == len(depths) - 1
                else nn.Identity()
            )
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

    # @profile_method(verbose=True)
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

    # @profile_method(verbose=True)
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

    # @profile_method(verbose=True)
    def depth_proj(self, out_features):
        depths = []
        h_out, w_out = out_features[-1].shape[-2:]
        # aggregate output and project to depth
        for i, (layer, features) in enumerate(zip(self.depth_mlp, out_features)):
            out_depth_features = layer(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(self.depth_mlp) - 1:
                continue
            depths.append(out_depth_features)
        out_depth_features = F.interpolate(out_depth_features, size=(h_out, w_out), mode="bilinear", align_corners=True)
        logdepth = self.to_depth_lr(out_depth_features)
        logdepth = F.interpolate(logdepth, size=self.original_shapes, mode="bilinear", align_corners=True)
        logdepth = self.to_depth_hr(logdepth)
        return logdepth

    # @profile_method(verbose=True)
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
    ) -> torch.Tensor:
        rays_embeddings = self.embed_rays(rays_hr)
        features, lowres_features = self.process(features, rays_embeddings)
        logdepth, logconf = self.decode(features)
        return logdepth, logconf, lowres_features


class Decoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)

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

    def run_camera(self, cls_tokens, original_shapes, rays_gt):
        H, W = original_shapes

        # camera layer
        intrinsics, sh_coeffs = self.angular_module(cls_tokens=cls_tokens)
        B, N = intrinsics.shape
        device = intrinsics.device
        dtype = intrinsics.dtype

        id_coords = coords_grid(B, H, W, device=sh_coeffs.device)

        # This is fov based
        longitude = (id_coords[:, 0] - intrinsics[:, 2].view(-1, 1, 1)) / W * intrinsics[:, 0].view(-1, 1, 1)
        latitude = (id_coords[:, 1] - intrinsics[:, 3].view(-1, 1, 1)) / H * intrinsics[:, 1].view(-1, 1, 1)
        x = torch.cos(latitude) * torch.sin(longitude)
        z = torch.cos(latitude) * torch.cos(longitude)
        y = -torch.sin(latitude)
        unit_sphere = torch.stack([x, y, z], dim=-1)
        unit_sphere = unit_sphere / torch.norm(unit_sphere, dim=-1, keepdim=True).clip(min=1e-5)

        harmonics = rsh_cart_3(unit_sphere)[..., 1:]  # remove constant-value harmonic
        rays_pred = torch.einsum("bhwc,bcd->bhwd", harmonics, sh_coeffs)
        rays_pred = rays_pred / torch.norm(rays_pred, dim=-1, keepdim=True).clip(min=1e-5)
        rays_pred = rays_pred.permute(0, 3, 1, 2)

        ### LEGACY CODE for training
        # if self.training:
        #     prob = 1 - tanh(self.steps / self.num_steps)
        #     where_use_gt_rays = torch.rand(B, 1, 1, device=device, dtype=dtype) < prob
        #     where_use_gt_rays = where_use_gt_rays.int()
        #     rays = rays_gt * where_use_gt_rays + rays_pred * (1 - where_use_gt_rays)

        # should clean also nans
        if self.training:
            rays = rays_pred
        else:
            rays = rays_gt if rays_gt is not None else rays_pred
        rays = rearrange(rays, "b c h w -> b (h w) c")

        return intrinsics, rays

    def forward(self, inputs, image_metas) -> torch.Tensor:
        B, C, H, W = inputs["image"].shape
        device = inputs["image"].device

        rays_gt = inputs.get("rays", None)

        # get features in b n d format
        common_shape = inputs["features"][0].shape[1:3]

        # input shapes repeat shapes for each level, times the amount of the layers:
        features = self.input_adapter(inputs["features"])

        # positional embeddings, spatial and level
        level_embed = self.level_embeds.repeat(B, common_shape[0] * common_shape[1], 1, 1)
        level_embed = rearrange(level_embed, "b n l d -> b (n l) d")
        dummy_tensor = torch.zeros(B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False)
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(1, 4, 1)

        # get cls tokens projections
        camera_tokens = inputs["tokens"]
        camera_tokens = [self.choker(x.contiguous()) for x in camera_tokens]
        camera_tokens = self.camera_token_adapter(camera_tokens)
        self.angular_module.set_shapes((H, W))

        intrinsics, rays = self.run_camera(
            torch.cat(camera_tokens, dim=1),
            original_shapes=(H, W),
            rays_gt=rays_gt,
        )

        # run bulk of the model
        self.radial_module.set_shapes(common_shape)
        self.radial_module.set_original_shapes((H, W))
        logradius, logconfidence, lowres_features = self.radial_module(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )
        radius = torch.exp(logradius.clip(min=-8.0, max=8.0) + 2.0)
        confidence = torch.exp(logconfidence.clip(min=-8.0, max=10.0))

        outputs = {
            "distance": radius,
            "lowres_features": lowres_features,
            "confidence": confidence,
            "K": intrinsics,
            "rays": rays,
        }

        return outputs

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def get_params(self, lr, wd):
        angles_p, _ = get_params(self.angular_module, lr, wd)
        radius_p, _ = get_params(self.radial_module, lr, wd)
        tokens_p, _ = get_params(self.camera_token_adapter, lr, wd)
        input_p, _ = get_params(self.input_adapter, lr, wd)
        return [*tokens_p, *angles_p, *input_p, *radius_p]

    def build(self, config):
        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        expansion = config["model"]["expansion"]
        num_heads = config["model"]["num_heads"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        layer_scale = config["model"]["layer_scale"]
        depth = config["model"]["pixel_decoder"]["depths"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        out_dim = config["model"]["pixel_decoder"]["out_dim"]
        kernel_size = config["model"]["pixel_decoder"]["kernel_size"]
        self.slices_encoder = list(zip([d - 1 for d in depths_encoder], depths_encoder))
        input_dims = [input_dims[d - 1] for d in depths_encoder]
        self.steps = 0
        self.num_steps = config["model"].get("num_steps", 100000)

        camera_dims = input_dims
        self.choker = GradChoker(config["model"]["pixel_decoder"]["detach"])
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.camera_token_adapter = ListAdapter(camera_dims, hidden_dim)
        self.angular_module = AngularModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.radial_module = RadialModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=96,
            layer_scale=layer_scale,
            out_dim=out_dim,
            kernel_size=kernel_size,
            num_prompt_blocks=config["model"]["pixel_decoder"]["num_prompt_blocks"],
            use_norm=False,
        )
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            orthonormal_init(len(input_dims), hidden_dim).reshape(1, 1, len(input_dims), hidden_dim),
            requires_grad=False,
        )
