from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class LightweightLabelVAEOutput(BaseOutput):
    sample: torch.Tensor
    posterior: torch.Tensor


class LightweightLabelVAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 7, # for bits encoding
        intermediate_channels: int = 512,
        out_channels: int = 128,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.2,
        num_mid_blocks: int = 0,
        num_latents: int = 2,
        num_upscalers: int = 3,
        upscale_channels: int = 512,
        act_fn: str = "none",
    ):
        super().__init__()

        # pass init params
        self.enable_mid_block = num_mid_blocks > 0
        self.num_mid_blocks = num_mid_blocks
        self.downsample_factor = 2 ** (len(block_out_channels) - 1)
        self.interpolation_factor = self.downsample_factor // (2 ** num_upscalers)

        # define encoder
        self.encoder = self.define_encoder(
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            intermediate_channels=intermediate_channels,
            norm_num_groups=norm_num_groups,
            latent_channels=latent_channels,
            num_latents=num_latents,
        )

        # define decoder
        self.decoder = self.define_decoder(
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            norm_num_groups=norm_num_groups,
            latent_channels=latent_channels,
            num_upscalers=num_upscalers,
            upscale_channels=upscale_channels,
        )

        self.scaling_factor = scaling_factor
        self.num_latents = num_latents
        self.act_fn = act_fn
        assert self.num_latents in [1, 2, 32]

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError("Gradient checkpointing is not supported for this model.")

    def define_encoder(
        self,
        in_channels: int,
        block_out_channels: Tuple[int],
        intermediate_channels: int,
        norm_num_groups: int,
        latent_channels: int,
        num_latents: int,
    ):
        conv_in = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
        ]

        # down
        down_blocks = []
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            down_blocks.extend([
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
                nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
            ])
        encoder_down_blocks = [
            *down_blocks,
            nn.Conv2d(block_out_channels[-1], intermediate_channels, kernel_size=3, padding=1),
        ]

        # mid
        encoder_mid_blocks = []
        if self.enable_mid_block:
            encoder_mid_blocks.append(UNetMidBlock2D(
                in_channels=intermediate_channels,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                resnet_groups=norm_num_groups,
                temb_channels=None, # noqa
                add_attention=False,
            ))
        else:
            encoder_mid_blocks = [nn.Identity()]

        # out
        encoder_out_blocks = [
            nn.GroupNorm(num_channels=intermediate_channels, num_groups=norm_num_groups, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(intermediate_channels, latent_channels * num_latents, kernel_size=3, padding=1),
        ]

        return nn.Sequential(
            *conv_in,
            *encoder_down_blocks,
            *encoder_mid_blocks,
            *encoder_out_blocks,
        )

    def define_decoder(
        self,
        out_channels: int,
        intermediate_channels: int,
        norm_num_groups: int,
        latent_channels: int,
        num_upscalers: int,
        upscale_channels: int,
    ):
        conv_in = nn.Conv2d(latent_channels, intermediate_channels, kernel_size=3, padding=1)

        # mid
        if self.enable_mid_block:
            decoder_mid_blocks = UNetMidBlock2D(
                in_channels=intermediate_channels,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                resnet_groups=norm_num_groups,
                temb_channels=None, # noqa
                add_attention=False,
            )
        else:
            decoder_mid_blocks = nn.Identity()

        dim = upscale_channels
        upscaler = []
        for i in range(num_upscalers):
            in_channels = intermediate_channels if i == 0 else dim
            upscaler.extend([
                nn.ConvTranspose2d(in_channels, dim, kernel_size=2, stride=2),
                LayerNorm2d(dim),
                nn.SiLU(),
            ])
        upscaler.extend([
            nn.GroupNorm(norm_num_groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1),
        ])

        return nn.Sequential(
            conv_in,
            decoder_mid_blocks,
            *upscaler,
        )

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, interpolate=True, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        dec = self.decoder(z)
        if interpolate:
            dec = torch.nn.functional.interpolate(dec, scale_factor=self.interpolation_factor, mode="bilinear", align_corners=False)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[LightweightLabelVAEOutput, Tuple[torch.Tensor]]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, interpolate=False).sample

        if not return_dict:
            return (dec,)

        return LightweightLabelVAEOutput(sample=dec, posterior=posterior)


class LayerNorm2d(nn.Module):
    # copied from detectron2
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
