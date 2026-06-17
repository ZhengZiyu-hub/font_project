from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QFormerStyleEncoderConfig:
    input_dim: int = 3584
    hidden_dim: int = 768
    num_queries: int = 16
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 1024
    style_dim: int = 768
    dropout: float = 0.1


class QFormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_visual_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended

        attended, _ = self.cross_attn(
            self.cross_query_norm(queries),
            self.cross_visual_norm(visual_tokens),
            self.cross_visual_norm(visual_tokens),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class QFormerStyleEncoder(nn.Module):
    """Convert offline Qwen visual tokens into compact font-style features."""

    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 768,
        num_queries: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        style_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.config = QFormerStyleEncoderConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            style_dim=style_dim,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.learnable_queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.layers = nn.ModuleList(
            QFormerLayer(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, style_dim),
        )
        nn.init.normal_(self.learnable_queries, std=0.02)

    def get_config(self) -> dict:
        return asdict(self.config)

    def forward(self, qwen_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if qwen_tokens.ndim != 3:
            raise ValueError(
                f"Expected qwen_tokens shape [B, T, D], got {tuple(qwen_tokens.shape)}"
            )
        if qwen_tokens.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {qwen_tokens.shape[-1]}"
            )
        visual_tokens = self.input_norm(self.input_proj(qwen_tokens))
        queries = self.learnable_queries.unsqueeze(0).expand(qwen_tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual_tokens)
        style_tokens = self.mlp_head(self.output_norm(queries))
        style_pooled = F.normalize(style_tokens.mean(dim=1), p=2, dim=-1)
        return style_tokens, style_pooled
