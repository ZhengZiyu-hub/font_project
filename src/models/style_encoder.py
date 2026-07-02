from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .projection import MLPProjModel, QFormerProjModel


class StyleEncoder(nn.Module):
    """Encode a reference image with a SigLIP vision tower and projection heads.

    Input:
        style_image: ``[B, 3, H, W]`` tensor. Values may be in ``[-1, 1]`` or
        ``[0, 1]``; they are resized to the vision tower image size and
        normalized with mean/std ``0.5``.

    Output:
        style_tokens: ``[B, num_tokens, cross_attention_dim]``. These tokens are
        injected into the FLUX decoder as image-condition tokens.

    The structure mirrors the reference image-conditioning path: pooled SigLIP
    image embedding -> MLP projection + QFormer-style projection -> summed
    condition tokens.
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
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:  # pragma: no cover - depends on environment setup.
            raise ImportError("StyleEncoder requires transformers to load SigLIPVisionModel.") from exc

        self.image_size = image_size
        self.freeze_image_encoder = freeze_image_encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
        self.image_proj_mlp = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
        )
        self.image_proj_qformer = QFormerProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
        )

        if freeze_image_encoder:
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)

    def _preprocess_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Resize and normalize tensor images for SigLIP.

        Shape:
            ``[B, 3, H, W]`` -> ``[B, 3, image_size, image_size]``.
        """

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B, 3, H, W], got {tuple(image.shape)}")

        image = image.float()
        if image.detach().amin() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0.0, 1.0)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )

        # SigLIP processor uses mean=[0.5]*3 and std=[0.5]*3, so [0, 1]
        # becomes [-1, 1].
        return (image - 0.5) / 0.5

    def encode_image_embedding(self, style_image: torch.Tensor) -> torch.Tensor:
        """Return pooled SigLIP image embedding.

        Input shape:
            ``style_image`` is ``[B, 3, H, W]``.

        Output shape:
            pooled image embedding is ``[B, id_embeddings_dim]``. This is used
            both by the style projection heads and by retrieval experiments.
        """

        pixel_values = self._preprocess_tensor(style_image)
        encoder_dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = pixel_values.to(device=style_image.device, dtype=encoder_dtype)
        self.image_encoder.to(style_image.device)

        if self.freeze_image_encoder:
            with torch.no_grad():
                return self.image_encoder(pixel_values).pooler_output
        return self.image_encoder(pixel_values).pooler_output

    def forward(self, style_image: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_image_embedding(style_image)
        pooled = pooled.to(dtype=self.image_proj_mlp.proj[0].weight.dtype)
        # Both heads return [B, num_tokens, cross_attention_dim]; summing keeps
        # the token count unchanged while combining global and query-conditioned
        # image features.
        return self.image_proj_mlp(pooled) + self.image_proj_qformer(pooled)
