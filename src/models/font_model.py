from __future__ import annotations

import glob
import os

import torch
from torch import nn

from .content_encoder import ContentEncoder
from .flux_decoder import FluxDecoderConfig, FluxImageDecoder
from .style_encoder import StyleEncoder


def _resolve_image_encoder_path(image_encoder_path: str | None) -> str:
    if image_encoder_path:
        return image_encoder_path

    env_path = os.environ.get("FONT_IMAGE_ENCODER_PATH")
    if env_path:
        return env_path

    preferred_path = "/data/zhengziyu/models/siglip-so400m-patch14-384"
    if os.path.isdir(preferred_path):
        return preferred_path

    matches = glob.glob("/data/zhengziyu/**/siglip-so400m-patch14-384", recursive=True)
    if matches:
        return sorted(matches)[0]

    return preferred_path


class FontModel(nn.Module):
    """Image-to-image font model with a FLUX decoder.

    Forward inputs:
        content_image: ``[B, 3, H, W]``
        style_image: ``[B, 3, H, W]``

    Forward output:
        output_image: ``[B, 3, H, W]``
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_encoder_path: str | None = None,
        condition_dim: int = 4096,
        image_embedding_dim: int = 1152,
        condition_tokens: int = 128,
        condition_heads: int = 8,
        condition_query_tokens: int = 32,
        image_size: int = 384,
        freeze_image_encoder: bool = True,
        num_heads: int = 24,
        decoder_blocks: int = 19,
        decoder_single_blocks: int = 38,
        flux_model_path: str | None = None,
    ) -> None:
        super().__init__()
        image_encoder_path = _resolve_image_encoder_path(image_encoder_path)
        self.style_encoder = StyleEncoder(
            image_encoder_path=image_encoder_path,
            cross_attention_dim=condition_dim,
            id_embeddings_dim=image_embedding_dim,
            num_tokens=condition_tokens,
            num_heads=condition_heads,
            num_query_tokens=condition_query_tokens,
            image_size=image_size,
            freeze_image_encoder=freeze_image_encoder,
        )
        self.content_encoder = ContentEncoder(
            image_encoder_path=image_encoder_path,
            cross_attention_dim=condition_dim,
            id_embeddings_dim=image_embedding_dim,
            num_tokens=condition_tokens,
            num_heads=condition_heads,
            num_query_tokens=condition_query_tokens,
            image_size=image_size,
            freeze_image_encoder=freeze_image_encoder,
        )
        self.decoder = FluxImageDecoder(
            FluxDecoderConfig(
                image_channels=image_channels,
                hidden_dim=condition_dim,
                num_layers=decoder_blocks,
                num_single_layers=decoder_single_blocks,
                num_attention_heads=num_heads,
                attention_head_dim=max(condition_dim // num_heads, 1),
                pooled_projection_dim=768,
                pretrained_model_name_or_path=flux_model_path,
            )
        )

    def forward(self, content_image: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        # content/style tokens are already projected to the FLUX
        # joint-attention dimension: [B, N, condition_dim].
        content_tokens = self.content_encoder(content_image)
        style_tokens = self.style_encoder(style_image)

        # The content image is used as the initial latent canvas in this stage;
        # FLUX receives content and style tokens through encoder_hidden_states.
        return self.decoder(content_image, content_tokens, style_tokens)
