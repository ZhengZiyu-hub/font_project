from __future__ import annotations

import torch
from torch import nn

from .projection import PooledFeatureProjection


class StyleEncoder(nn.Module):
    """Encode a style reference image into condition tokens.

    Args:
        style_image: tensor with shape ``[B, 3, H, W]``.

    Returns:
        style tokens with shape ``[B, num_tokens, token_dim]``.

    The current version uses a compact Conv encoder so the project can run
    locally without downloading a large vision backbone. Later, ``self.backbone``
    can be replaced by a pretrained visual encoder while keeping the same
    token output contract.
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_dim: int = 128,
        token_dim: int = 128,
        num_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_tokens = PooledFeatureProjection(feature_dim, token_dim, num_tokens)

    def forward(self, style_image: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(style_image).flatten(1)
        return self.to_tokens(pooled)
