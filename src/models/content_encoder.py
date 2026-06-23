from __future__ import annotations

import torch
from torch import nn


class ContentEncoder(nn.Module):
    """Encode a source glyph image into spatial structure tokens.

    Args:
        content_image: tensor with shape ``[B, 3, H, W]``.

    Returns:
        content tokens with shape ``[B, N, token_dim]`` where
        ``N = ceil(H / 4) * ceil(W / 4)`` for the default two stride-2 blocks.
    """

    def __init__(self, image_channels: int = 3, token_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, token_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, token_dim),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, content_image: torch.Tensor) -> torch.Tensor:
        feature_map = self.encoder(content_image)
        # [B, D, h, w] -> [B, h*w, D], a token for each low-res spatial cell.
        tokens = feature_map.flatten(2).transpose(1, 2)
        return self.norm(tokens)
