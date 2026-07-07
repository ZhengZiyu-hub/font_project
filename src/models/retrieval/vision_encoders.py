from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _BaseImageEmbeddingEncoder(nn.Module):
    """Shared image preprocessing for retrieval encoders.

    Input:
        image: ``[B, 3, H, W]`` in ``[-1, 1]`` or ``[0, 1]``.

    Output:
        normalized image embeddings ``[B, D]`` for cosine retrieval.
    """

    def __init__(self, image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super().__init__()
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def _preprocess(self, image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        image = image.float()
        if image.detach().amin() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0.0, 1.0)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)
        return ((image - mean) / std).to(dtype=dtype)


class ClipImageEmbeddingEncoder(_BaseImageEmbeddingEncoder):
    """CLIP image encoder for coarse font retrieval.

    The module uses ``transformers.CLIPVisionModel`` and returns the pooled image
    embedding normalized to unit length.
    """

    def __init__(self, model_path: str, image_size: int = 224, freeze: bool = True) -> None:
        super().__init__(
            image_size=image_size,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:  # pragma: no cover - depends on environment setup.
            raise ImportError("ClipImageEmbeddingEncoder requires transformers.") from exc

        self.model = CLIPVisionModel.from_pretrained(model_path)
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        pixel_values = self._preprocess(image, dtype=dtype)
        self.model.to(image.device)
        if self.freeze:
            with torch.no_grad():
                pooled = self.model(pixel_values).pooler_output
        else:
            pooled = self.model(pixel_values).pooler_output
        return F.normalize(pooled.float(), dim=-1)


class DinoImageEmbeddingEncoder(_BaseImageEmbeddingEncoder):
    """DINOv2 image encoder for reranking font retrieval candidates."""

    def __init__(self, model_path: str, image_size: int = 224, freeze: bool = True) -> None:
        super().__init__(
            image_size=image_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on environment setup.
            raise ImportError("DinoImageEmbeddingEncoder requires transformers.") from exc

        self.model = AutoModel.from_pretrained(model_path)
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        pixel_values = self._preprocess(image, dtype=dtype)
        self.model.to(image.device)
        if self.freeze:
            with torch.no_grad():
                hidden_states = self.model(pixel_values).last_hidden_state
        else:
            hidden_states = self.model(pixel_values).last_hidden_state
        patch_tokens = hidden_states[:, 1:]
        return F.normalize(patch_tokens.float().mean(dim=1), dim=-1)
