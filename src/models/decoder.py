from __future__ import annotations

import torch
from torch import nn

from .attention import CrossAttentionBlock


class ImageDecoder(nn.Module):
    """Decode latent image features with content and style conditions.

    Args:
        latent: ``[B, C, H, W]``
        content_tokens: ``[B, N_content, D]``
        style_tokens: ``[B, N_style, D]``

    Returns:
        image tensor with shape ``[B, 3, H, W]``.
    """

    def __init__(
        self,
        latent_channels: int = 3,
        hidden_dim: int = 128,
        image_channels: int = 3,
        num_heads: int = 4,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "content": CrossAttentionBlock(hidden_dim, num_heads),
                        "style": CrossAttentionBlock(hidden_dim, num_heads),
                    }
                )
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> torch.Tensor:
        features = self.input_proj(latent)
        batch_size, channels, height, width = features.shape

        # [B, D, H, W] -> [B, H*W, D], matching attention token layout.
        image_tokens = features.flatten(2).transpose(1, 2)
        for block in self.blocks:
            image_tokens = block["content"](image_tokens, content_tokens)
            image_tokens = block["style"](image_tokens, style_tokens)

        features = image_tokens.transpose(1, 2).view(batch_size, channels, height, width)
        return self.output(features)
