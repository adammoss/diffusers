from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch.nn as nn

from diffusers.configuration_utils import register_to_config
from diffusers.models.vae import Decoder, DecoderOutput, Encoder, VectorQuantizer

from diffusers import VQModel as VQModelBase


class VQModel(VQModelBase):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 0.18215,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, down_block_types=down_block_types,
                         up_block_types=up_block_types, block_out_channels=block_out_channels,
                         layers_per_block=layers_per_block, act_fn=act_fn, latent_channels=latent_channels,
                         sample_size=sample_size, num_vq_embeddings=num_vq_embeddings,
                         norm_num_groups=norm_num_groups, vq_embed_dim=vq_embed_dim, scaling_factor=scaling_factor)

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None,
                                        sane_index_shape=False, legacy=False)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )
