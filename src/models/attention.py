from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """Cross-attention block used to inject condition tokens into image tokens.

    Query comes from decoder image tokens. Key/value come from content or style
    tokens. This mirrors the condition-as-extra-KV idea in a compact form.

    Inputs:
        hidden_tokens: ``[B, N_img, D]``
        condition_tokens: ``[B, N_cond, D]``

    Output:
        fused tokens with shape ``[B, N_img, D]``.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, hidden_tokens: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(hidden_tokens)
        key_value = self.condition_norm(condition_tokens)
        attended, _ = self.attn(query=query, key=key_value, value=key_value, need_weights=False)
        hidden_tokens = hidden_tokens + attended
        return hidden_tokens + self.ff(self.ff_norm(hidden_tokens))
