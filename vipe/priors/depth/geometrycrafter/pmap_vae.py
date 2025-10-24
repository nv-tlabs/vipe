from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, Encoder
from diffusers.utils import is_torch_version
from diffusers.models.unets.unet_3d_blocks import UpBlockTemporalDecoder, MidBlockTemporalDecoder
from diffusers.models.resnet import SpatioTemporalResBlock

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PMapTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Tuple[int] = (1, 1, 1),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, 
            block_out_channels[-1], 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_block = UpBlockTemporalDecoder(
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.out_blocks = nn.ModuleList([])
        self.time_conv_outs = nn.ModuleList([])
        for out_channel in out_channels:
            self.out_blocks.append(
                nn.ModuleList([
                    nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        block_out_channels[0], 
                        block_out_channels[0] // 2, 
                        kernel_size=3, 
                        padding=1
                    ),
                    SpatioTemporalResBlock(
                        in_channels=block_out_channels[0] // 2,
                        out_channels=block_out_channels[0] // 2,
                        temb_channels=None,
                        eps=1e-6,
                        temporal_eps=1e-5,
                        merge_factor=0.0,
                        merge_strategy="learned",
                        switch_spatial_to_temporal_mix=True
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        block_out_channels[0] // 2, 
                        out_channel, 
                        kernel_size=1,
                    )
                ])
            )

            conv_out_kernel_size = (3, 1, 1)
            padding = [int(k // 2) for k in conv_out_kernel_size]
            self.time_conv_outs.append(nn.Conv3d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=conv_out_kernel_size,
                padding=padding,
            ))

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        image_only_indicator: torch.Tensor,
        num_frames: int = 1,
    ):
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                    )
        else:
            # middle
            sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, image_only_indicator=image_only_indicator)

        # post-process

        output = []
        
        for out_block, time_conv_out in zip(self.out_blocks, self.time_conv_outs):
            x = sample
            for layer in out_block:
                if isinstance(layer, SpatioTemporalResBlock):
                    x = layer(x, None, image_only_indicator)
                else:
                    x = layer(x)
            
            
            batch_frames, channels, height, width = x.shape
            batch_size = batch_frames // num_frames
            x = x[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            x = time_conv_out(x)
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
            output.append(x)

        return output

class PMapAutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        enc_down_block_types: Tuple[str] = (
            "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"
        ),
        enc_block_out_channels: Tuple[int] = (128, 256, 512, 512),
        enc_layers_per_block: int = 2,
        dec_block_out_channels: Tuple[int] = (128, 256, 512, 512),
        dec_layers_per_block: int = 2,
        out_channels: Tuple[int] = (1, 1, 1),
        mid_block_add_attention: bool = True,
        offset_scale_factor: float = 0.1,
        **kwargs  
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=enc_down_block_types,
            block_out_channels=enc_block_out_channels,
            layers_per_block=enc_layers_per_block,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention
        )
        zero_module(self.encoder.conv_out)

        self.offset_scale_factor = offset_scale_factor

        self.decoder = PMapTemporalDecoder(
            in_channels=latent_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            out_channels=out_channels
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, PMapTemporalDecoder)):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    @apply_forward_hook
    def encode(
        self, 
        x: torch.Tensor, 
        latent_dist: DiagonalGaussianDistribution
    ) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        offset = h * self.offset_scale_factor
        param = latent_dist.parameters.to(h.dtype)
        mean, logvar = torch.chunk(param, 2, dim=1)
        posterior = DiagonalGaussianDistribution(torch.cat([mean + offset, logvar], dim=1))
        return posterior

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)
        return decoded