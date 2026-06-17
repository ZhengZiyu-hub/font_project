from __future__ import annotations

import torch
import torch.nn as nn


class FluxStyleAdapterPlaceholder(nn.Module):
    """Future interface for projecting style tokens into FLUX attention K/V."""

    def __init__(self, style_dim: int = 768, flux_attention_dim: int | None = None) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.flux_attention_dim = flux_attention_dim

    def forward(self, style_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if style_tokens.ndim != 3 or style_tokens.shape[-1] != self.style_dim:
            raise ValueError(
                f"Expected style_tokens [B, N, {self.style_dim}], got {tuple(style_tokens.shape)}"
            )
        raise NotImplementedError(
            "FLUX integration is intentionally deferred. A future implementation will "
            "project style_tokens into k_style and v_style for attention injection."
        )
