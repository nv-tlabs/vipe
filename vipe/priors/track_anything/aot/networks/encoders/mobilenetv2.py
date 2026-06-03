# This file includes code originally from the AOT Benchmark repository:
# https://github.com/yoxu515/aot-benchmark
# Licensed under the BSD-3-Clause License. See THIRD_PARTY_LICENSES.md for details.

from typing import Callable, List, Optional

from torch import Tensor, nn

from ...utils.learning import freeze_params


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int = -1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        if padding == -1:
            padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )
        self.out_channels = out_planes


ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        dilation: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    dilation=dilation,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        output_stride=8,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        freeze_at=0,
    ) -> None:
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        last_channel = 1280
        input_channel = 32
        current_stride = 1
        rate = 1

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty or a 4-element list, "
                f"got {inverted_residual_setting}"
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        current_stride *= 2

        for t, c, n, s in inverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, stride, dilation, t, norm_layer))
                else:
                    features.append(block(input_channel, output_channel, 1, rate, t, norm_layer))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)

        self._initialize_weights()

        feature_4x = self.features[0:4]
        feature_8x = self.features[4:7]
        feature_16x = self.features[7:14]
        feature_32x = self.features[14:]

        self.stages = [feature_4x, feature_8x, feature_16x, feature_32x]
        self.freeze(freeze_at)

    def forward(self, x):
        xs = []
        for stage in self.stages:
            x = stage(x)
            xs.append(x)
        return xs

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            for module in self.stages[0][0]:
                freeze_params(module)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)
