# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This file is adapted from the DINOv3 inference model code distributed under
# the DINOv3 License Agreement. See THIRD_PARTY_LICENSES.md for details.

import math
from functools import partial
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        out_features = self.out_features
        assert out_features % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


def _rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rope_rotate_half(x) * sin)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(0.0)

    def apply_rope(self, q: Tensor, k: Tensor, rope: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)

        prefix = q.shape[-2] - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        k_prefix = k[:, :, :prefix, :]
        q = torch.cat((q_prefix, _rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
        k = torch.cat((k_prefix, _rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, tokens, _ = x.shape
        channels = self.qkv.in_features

        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        del act_layer, drop
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden = int(hidden_features * 2 / 3)
        swiglu_hidden = hidden + (-hidden % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float | Tensor = 1e-5, inplace: bool = False, device=None) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_hw = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        patch_hw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        height, width = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, height, width, self.embed_dim)
        return x

    def reset_parameters(self) -> None:
        scale = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(scale), math.sqrt(scale))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(scale), math.sqrt(scale))


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        head_dim = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer("periods", torch.empty(head_dim // 4, device=device, dtype=dtype), persistent=True)
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_hw = torch.empty(2, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords *= jitter_hw[None, :]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            coords *= torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = (base**exponents) / base * self.max_period
        self.periods.data = periods


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


def _named_apply(fn: Callable[[nn.Module, str], None], module: nn.Module, name: str = "") -> nn.Module:
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        _named_apply(fn, child_module, child_name)
    if name:
        fn(module, name)
    return module


def _init_weights_vit(module: nn.Module, name: str = "") -> None:
    del name
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            out_features = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[out_features // 3 : 2 * out_features // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device=None,
    ) -> None:
        del drop_path_rate
        super().__init__()
        norm_layer_cls = _NORM_LAYER_DICT[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=_DTYPE_DICT[pos_embed_rope_dtype],
            device=device,
        )

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=_FFN_LAYER_DICT[ffn_layer],
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self) -> None:
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        _named_apply(_init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Tensor | None = None) -> tuple[Tensor, tuple[int, int]]:
        x = self.patch_embed(x)
        batch, height, width, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(batch, -1, -1),
                storage_tokens.expand(batch, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (height, width)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int | Sequence[int] = 1) -> list[Tensor]:
        x, (height, width) = self.prepare_tokens_with_masks(x)
        blocks_to_take = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n

        output = []
        rope_sincos = self.rope_embed(H=height, W=width)
        for idx, block in enumerate(self.blocks):
            x = block(x, rope_sincos)
            if idx in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    assert self.cls_norm is not None
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            batch, _, height, width = x.shape
            outputs = [
                out.reshape(batch, height // self.patch_size, width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(zip(outputs, class_tokens, extra_tokens))


_FFN_LAYER_DICT = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

_NORM_LAYER_DICT = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

_DTYPE_DICT = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


_DINO_CONFIGS = {
    "vits": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
    "vitb": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
    "vitl": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
}


def make_dinov3_backbone(model_name: str) -> DinoVisionTransformer:
    if model_name not in _DINO_CONFIGS:
        raise ValueError(f"Unsupported DAP DINOv3 backbone '{model_name}'. Expected one of {list(_DINO_CONFIGS)}.")
    model = DinoVisionTransformer(
        img_size=224,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        **_DINO_CONFIGS[model_name],
    )
    model.init_weights()
    return model


class DINOv3Adapter(nn.Module):
    MODEL_MAP = {
        "vits": "vits",
        "vitb": "vitb",
        "vitl": "vitl",
    }

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Unknown model_name={model_name}, must be one of {list(self.MODEL_MAP)}")

        self.model = make_dinov3_backbone(self.MODEL_MAP[model_name])
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            raise AttributeError("DINOv3 model missing embed_dim")
        self.patch_size = getattr(self.model, "patch_size", None)
        if self.patch_size is None:
            raise AttributeError("DINOv3 model missing patch_size")
        self.blocks = getattr(self.model, "blocks", None)
        if self.blocks is None:
            raise AttributeError("DINOv3 model missing blocks")

        self.n_blocks = getattr(self.model, "n_blocks", len(self.blocks))
        self.depth = self.n_blocks
        self.norm = nn.LayerNorm(self.embed_dim)

    def get_intermediate_layers(self, x: Tensor, n=1, return_class_token: bool = False, norm: bool = True):
        outputs = self.model.get_intermediate_layers(x, n=n, reshape=False, return_class_token=True, norm=norm)

        patch_maps = []
        cls_tokens = []
        height, width = x.shape[-2], x.shape[-1]
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        for out_all, out_cls in outputs:
            if norm:
                out_all = self.norm(out_all)

            out_patches = out_all[:, 1:, :]
            batch, tokens, channels = out_patches.shape
            sqrt_tokens = int(tokens**0.5)
            if sqrt_tokens * sqrt_tokens == tokens:
                grid = out_patches.transpose(1, 2).reshape(batch, channels, sqrt_tokens, sqrt_tokens)
            else:
                grid = out_patches.transpose(1, 2).reshape(batch, channels, tokens, 1)
                grid = F.interpolate(grid, size=(grid_h * grid_w, 1), mode="bilinear").squeeze(-1)
                grid = grid.reshape(batch, channels, grid_h, grid_w)

            if grid.shape[-2:] != (grid_h, grid_w):
                grid = F.interpolate(grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False)

            patch_maps.append(grid.contiguous())
            cls_tokens.append(out_cls)

        if return_class_token:
            return tuple(zip(patch_maps, cls_tokens))
        return tuple(patch_maps)
