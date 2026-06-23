from __future__ import annotations

import torch
from torch import nn

from .content_encoder import ContentEncoder
from .decoder import ImageDecoder
from .projection import TokenProjection
from .style_encoder import StyleEncoder


class FontModel(nn.Module):
    """Minimal image-to-image font model.

    Forward inputs:
        content_image: ``[B, 3, H, W]``
        style_image: ``[B, 3, H, W]``

    Forward output:
        output_image: ``[B, 3, H, W]``
    """

    def __init__(
        self,
        image_channels: int = 3,
        encoder_dim: int = 128,
        hidden_dim: int = 128,
        style_tokens: int = 8,
        num_heads: int = 4,
        decoder_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.style_encoder = StyleEncoder(
            image_channels=image_channels,
            feature_dim=encoder_dim,
            token_dim=encoder_dim,
            num_tokens=style_tokens,
        )
        self.content_encoder = ContentEncoder(image_channels=image_channels, token_dim=encoder_dim)
        self.style_projection = TokenProjection(encoder_dim, hidden_dim)
        self.content_projection = TokenProjection(encoder_dim, hidden_dim)
        self.decoder = ImageDecoder(
            latent_channels=image_channels,
            hidden_dim=hidden_dim,
            image_channels=image_channels,
            num_heads=num_heads,
            num_blocks=decoder_blocks,
        )

    def forward(self, content_image: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        content_tokens = self.content_projection(self.content_encoder(content_image))
        style_tokens = self.style_projection(self.style_encoder(style_image))

        # The content image is used as the initial latent canvas in this stage.
        return self.decoder(content_image, content_tokens, style_tokens)
