from __future__ import annotations

import torch
from torch import nn


class TokenProjection(nn.Module):
    """Project token features into the decoder hidden dimension.

    Input shape: ``[B, N, input_dim]``
    Output shape: ``[B, N, output_dim]``
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(tokens))


class PooledFeatureProjection(nn.Module):
    """Expand one pooled image feature into multiple condition tokens.

    This follows the useful idea of mapping one global visual embedding into a
    small token set, while keeping the implementation lightweight.

    Input shape: ``[B, input_dim]``
    Output shape: ``[B, num_tokens, output_dim]``
    """

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, num_tokens * output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        tokens = self.net(feature).view(feature.shape[0], self.num_tokens, self.output_dim)
        return self.norm(tokens)
