from __future__ import annotations

import torch
from torch import nn

from .style_encoder import StyleEncoder


class ContentEncoder(nn.Module):
    """Encode the content image with the same image-condition stack.

    Input:
        content_image: ``[B, 3, H, W]``.

    Output:
        content_tokens: ``[B, num_tokens, cross_attention_dim]``.

    The reference model does not define a separate lightweight content encoder;
    the editable glyph image is normally passed through the image pipeline
    while style enters as image-condition tokens. For this project structure we
    keep a named ``ContentEncoder`` module, but its internals intentionally use
    the same SigLIP + MLP projection + QFormer projection path as
    ``StyleEncoder`` so there is no simplified Conv branch.
    """

    def __init__(
        self,
        image_encoder_path: str,
        cross_attention_dim: int = 4096,
        id_embeddings_dim: int = 1152,
        num_tokens: int = 128,
        num_heads: int = 8,
        num_query_tokens: int = 32,
        image_size: int = 384,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = StyleEncoder(
            image_encoder_path=image_encoder_path,
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            image_size=image_size,
            freeze_image_encoder=freeze_image_encoder,
        )

    def forward(self, content_image: torch.Tensor) -> torch.Tensor:
        return self.encoder(content_image)
