from __future__ import annotations

import torch
from torch import nn


class QFormerProjModel(nn.Module):
    """Project one pooled vision embedding into query-conditioned tokens.

    Input:
        id_embeds: ``[B, id_embeddings_dim]`` pooled image embedding.

    Output:
        tokens: ``[B, num_tokens, cross_attention_dim]`` for decoder
        cross-attention. The learnable query tokens attend to a projected
        key/value sequence, so the image feature is expanded into a richer
        condition token set.
    """

    def __init__(
        self,
        cross_attention_dim: int = 4096,
        id_embeddings_dim: int = 1152,
        num_tokens: int = 128,
        num_heads: int = 8,
        num_query_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.query_embeds = nn.Parameter(torch.randn(num_tokens, cross_attention_dim))

        self.id_proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_query_tokens),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = id_embeds.size(0)

        # [B, E] -> [B, num_query_tokens, D], used as K/V memory.
        projected = self.id_proj(id_embeds)
        kv = projected.view(batch_size, -1, self.cross_attention_dim)

        # [num_tokens, D] -> [B, num_tokens, D], used as learnable queries.
        queries = self.query_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, _ = self.cross_attn(query=queries, key=kv, value=kv)
        attn_output = self.cross_attn_norm(attn_output + queries)

        return self.norm(attn_output)


class MLPProjModel(nn.Module):
    """Expand one pooled vision embedding with an MLP.

    Input:
        id_embeds: ``[B, id_embeddings_dim]``

    Output:
        tokens: ``[B, num_tokens, cross_attention_dim]``
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


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
