from __future__ import annotations

import torch
from torch import nn


class ContentEncoder(nn.Module):
    """Encode a rendered/content glyph image into FLUX conditioning tokens.

    Inputs:
        content_image: ``[B, 3, H, W]`` in ``[-1, 1]`` or ``[0, 1]``.

    Outputs:
        glyph_tokens: ``[B, num_tokens, hidden_dim]`` for the glyph/content
        branch used by baseline and retrieval modes.
    """

    def __init__(self, hidden_dim: int = 4096, num_tokens: int = 32) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((max(1, num_tokens // 4), 4))
        self.norm = nn.LayerNorm(hidden_dim)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B, 3, H, W], got {tuple(image.shape)}")
        image = image.float()
        if image.detach().amin() < 0:
            image = (image + 1.0) * 0.5
        return image.clamp(0.0, 1.0) * 2.0 - 1.0

    def forward(self, content_image: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(self._normalize_image(content_image))
        tokens = self.pool(feature).flatten(2).transpose(1, 2)
        if tokens.shape[1] > self.num_tokens:
            tokens = tokens[:, : self.num_tokens]
        return self.norm(tokens)
