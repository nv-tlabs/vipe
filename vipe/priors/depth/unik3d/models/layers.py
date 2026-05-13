from functools import partial
from math import pi
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.autograd import Function

from ..utils.misc import default


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        dilation: int = 1,
        layer_scale: float = 1.0,
        use_norm: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.activation = nn.LeakyReLU()
        self.skip_add = nn.quantized.FloatFunctional()
        self.gamma = nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1)) if layer_scale > 0.0 else 1.0
        self.norm1 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.skip_add.add(self.gamma * out, x)


class ResUpsampleBil(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim: int = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        use_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim // 2
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    # @profile_method(verbose=True)
    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x


class ResUpsample(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                )
            )
        self.up = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2, padding=0)

    # @profile_method(verbose=True)
    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x


class ResUpsampleSH(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                )
            )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_dim // 4,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def generate_fourier_features(x, max_freq=64, num_bands=16):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(-max_freq / 2, max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x.flatten(-2)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
        "invalid dimensions for broadcastable concatentation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 30,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): contains the position of the current toke

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        rope_cache = self.cache[input_pos]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.unsqueeze(2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        return x_out.flatten(3)


class ChockerFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.alpha
        return grad_input, None


class GradChoker(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        alpha = torch.tensor(self.alpha, requires_grad=False, device=x.device)
        return ChockerFunction.apply(x, alpha)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x


class SimpleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim

        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads)
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        detach_query: bool = False,
        residual_ls: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = dim if context_dim is None else context_dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=False)
        self.ls1_1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 and not residual_ls else nn.Identity()
        self.ls1_2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 and residual_ls else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.detach_query = detach_query

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.detach_query:
            x = x.detach()
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q.permute(0, 2, 1, 3), input_pos=rope_pos).permute(0, 2, 1, 3)
            k = rope(k.permute(0, 2, 1, 3), input_pos=rope_pos).permute(0, 2, 1, 3)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads)
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = self.ls1_1(
            self.attn(
                x,
                rope=rope,
                rope_pos=rope_pos,
                attn_bias=attn_bias,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
            )
        ) + self.ls1_2(x)
        x = self.ls2(self.mlp(x)) + x
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        detach_query: bool = False,
        residual_ls: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    cosine=cosine,
                    gated=gated,
                    layer_scale=layer_scale,
                    context_dim=context_dim,
                    detach_query=detach_query,
                    residual_ls=residual_ls,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                attn_bias=attn_bias,
                rope=rope,
                rope_pos=rope_pos,
            )
        return x


class AttentionDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        single_head_ca: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.single_head_ca = single_head_ca
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv_ca = nn.Linear(context_dim, dim * 2, bias=False)
        self.q_ca = nn.Linear(dim, dim, bias=False)
        self.kv_sa = nn.Linear(dim, dim * 2, bias=False)
        self.q_sa = nn.Linear(dim, dim, bias=False)
        self.norm_x_sa = nn.LayerNorm(dim)
        self.norm_x_ca = nn.LayerNorm(dim)
        self.norm_ctx_ca = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out_ca = nn.Linear(dim, dim, bias=False)
        self.out_sa = nn.Linear(dim, dim, bias=False)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls3 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def cross_attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_heads = 1 if self.single_head_ca else self.num_heads
        x = self.norm_x_ca(x)
        context = self.norm_ctx_ca(context)
        k, v = rearrange(self.kv_ca(context), "b n (kv h d) -> b h n d kv", h=num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q_ca(x), "b n (h d) -> b h n d", h=num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=num_heads)
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b h n d", h=num_heads)
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_ca(x)
        return x

    def self_attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm_x_sa(x)
        k, v = rearrange(self.kv_sa(x), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q_sa(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        elif pos_embed is not None:
            pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
            q = q + pos_embed

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_sa(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
        rope_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.cross_attn(
                    x,
                    rope=rope,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.self_attn(x, rope=rope, attn_bias=attn_bias, pos_embed=pos_embed)) + x
        x = self.ls3(self.mlp(x)) + x
        return x
