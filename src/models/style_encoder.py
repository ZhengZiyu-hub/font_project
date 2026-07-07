from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .projection import MLPProjModel, QFormerProjModel


class QFormerFusion(nn.Module):
    """Fuse multi-scale visual tokens with learnable query tokens.

    Inputs:
        condition_tokens: ``[B, N_condition, D]`` from global/local/texture
            branches.

    Output:
        fused style tokens: ``[B, num_tokens, D]``.
    """

    def __init__(self, hidden_dim: int, num_tokens: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = condition_tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=self.condition_norm(condition_tokens),
            value=self.condition_norm(condition_tokens),
            need_weights=False,
        )
        tokens = self.cross_attn_norm(queries + attended)
        return tokens + self.ff(self.ff_norm(tokens))


class TextureEncoder(nn.Module):
    """Small CNN branch for local stroke/texture cues.

    Input:
        image: ``[B, 3, H, W]`` in normalized image space.

    Output:
        texture tokens: ``[B, N_texture, D]`` from a low-resolution feature map.
    """

    def __init__(self, hidden_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((max(1, num_tokens // 4), 4))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(image)
        feature = self.pool(feature)
        tokens = feature.flatten(2).transpose(1, 2)
        if tokens.shape[1] > self.num_tokens:
            tokens = tokens[:, : self.num_tokens]
        return self.norm(tokens)


class StyleEncoder(nn.Module):
    """Multi-scale style encoder.

    Input:
        style_image: ``[B, 3, H, W]`` tensor in ``[-1, 1]`` or ``[0, 1]``.

    Internal branches:
        global_style_tokens: SigLIP pooled image embedding -> ``[B, 1, D]``.
        local_style_tokens: DINOv2 patch tokens -> ``[B, N_local, D]``.
        texture_tokens: CNN texture tokens -> ``[B, N_texture, D]``.

    Output:
        style_tokens: ``[B, num_tokens, cross_attention_dim]``. The external
        API and output shape are unchanged from the previous StyleEncoder.
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
        pretrained_path: str | None = None,
        freeze: bool = True,
        dino_model_path: str | None = None,
        dino_image_size: int = 224,
        dino_local_tokens: int = 256,
        texture_tokens: int = 64,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, SiglipVisionModel
        except ImportError as exc:  # pragma: no cover - depends on environment setup.
            raise ImportError("StyleEncoder requires transformers for SigLIP and DINOv2.") from exc

        self.image_size = image_size
        self.dino_image_size = dino_image_size
        self.freeze_image_encoder = freeze_image_encoder
        self.dino_local_tokens = dino_local_tokens
        self.pretrained_path = pretrained_path
        self.freeze = freeze

        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
        dino_model_path = dino_model_path or os.environ.get("FONT_DINO_MODEL_PATH")
        if dino_model_path is None:
            local_dino_path = "/data/zhengziyu/models/dinov2-base"
            dino_model_path = local_dino_path if os.path.isdir(local_dino_path) else "facebook/dinov2-base"
        self.local_image_encoder = AutoModel.from_pretrained(dino_model_path)

        self.global_projection = nn.Sequential(
            nn.Linear(id_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )
        local_hidden_dim = int(self.local_image_encoder.config.hidden_size)
        self.local_projection = nn.Sequential(
            nn.Linear(local_hidden_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )
        self.texture_encoder = TextureEncoder(cross_attention_dim, texture_tokens)
        self.fusion = QFormerFusion(cross_attention_dim, num_tokens, num_heads)
        self.pretrained_mlp_projection: MLPProjModel | None = None
        self.pretrained_qformer_projection: QFormerProjModel | None = None

        if pretrained_path:
            self._load_pretrained_projection(
                pretrained_path=pretrained_path,
                cross_attention_dim=cross_attention_dim,
                id_embeddings_dim=id_embeddings_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                num_query_tokens=num_query_tokens,
            )

        if freeze_image_encoder:
            self.image_encoder.eval()
            self.local_image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)
            for param in self.local_image_encoder.parameters():
                param.requires_grad_(False)
        if freeze:
            self.set_trainable(False)
        else:
            self.set_trainable(True)

    def set_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze all style encoder parameters."""

        for param in self.parameters():
            param.requires_grad_(trainable)
        self.train(trainable)
        if not trainable:
            self.eval()

    def _extract_state_dict(self, payload: object) -> dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise TypeError("Style encoder checkpoint must be a dictionary.")
        for key in ("state_dict", "model", "module"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload

    def _load_pretrained_projection(
        self,
        pretrained_path: str,
        cross_attention_dim: int,
        id_embeddings_dim: int,
        num_tokens: int,
        num_heads: int,
        num_query_tokens: int,
    ) -> None:
        """Load pretrained SigLIP projection weights.

        Supported checkpoint format:
            ``{"image_proj_mlp": ..., "image_proj_qformer": ...}``

        These projections map pooled SigLIP embeddings ``[B, 1152]`` to FLUX
        condition tokens ``[B, num_tokens, cross_attention_dim]``.
        """

        checkpoint = self._extract_state_dict(torch.load(pretrained_path, map_location="cpu"))
        if "image_proj_mlp" not in checkpoint or "image_proj_qformer" not in checkpoint:
            missing = {"image_proj_mlp", "image_proj_qformer"} - set(checkpoint.keys())
            raise KeyError(f"Style checkpoint is missing projection weights: {sorted(missing)}")

        mlp_projection = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
        )
        qformer_projection = QFormerProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=id_embeddings_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
        )
        mlp_projection.load_state_dict(checkpoint["image_proj_mlp"], strict=True)
        qformer_projection.load_state_dict(checkpoint["image_proj_qformer"], strict=True)
        self.pretrained_mlp_projection = mlp_projection
        self.pretrained_qformer_projection = qformer_projection

    def uses_pretrained_projection(self) -> bool:
        return self.pretrained_mlp_projection is not None and self.pretrained_qformer_projection is not None

    def _to_unit_range(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape [B, 3, H, W], got {tuple(image.shape)}")
        image = image.float()
        if image.detach().amin() < 0:
            image = (image + 1.0) * 0.5
        return image.clamp(0.0, 1.0)

    def _preprocess_siglip(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            self._to_unit_range(image),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        return (image - 0.5) / 0.5

    def _preprocess_dino(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            self._to_unit_range(image),
            size=(self.dino_image_size, self.dino_image_size),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def encode_image_embedding(self, style_image: torch.Tensor) -> torch.Tensor:
        """Return pooled SigLIP image embedding for retrieval compatibility.

        Output shape:
            ``[B, id_embeddings_dim]``.
        """

        pixel_values = self._preprocess_siglip(style_image)
        encoder_dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = pixel_values.to(device=style_image.device, dtype=encoder_dtype)
        self.image_encoder.to(style_image.device)

        if self.freeze_image_encoder:
            with torch.no_grad():
                return self.image_encoder(pixel_values).pooler_output
        return self.image_encoder(pixel_values).pooler_output

    def encode_global_style(self, style_image: torch.Tensor) -> torch.Tensor:
        """SigLIP global branch: ``[B, 3, H, W] -> [B, 1, D]``."""

        pooled = self.encode_image_embedding(style_image)
        pooled = pooled.to(dtype=self.global_projection[0].weight.dtype)
        return self.global_projection(pooled).unsqueeze(1)

    def encode_local_style(self, style_image: torch.Tensor) -> torch.Tensor:
        """DINOv2 local branch: ``[B, 3, H, W] -> [B, N_local, D]``."""

        pixel_values = self._preprocess_dino(style_image)
        encoder_dtype = next(self.local_image_encoder.parameters()).dtype
        pixel_values = pixel_values.to(device=style_image.device, dtype=encoder_dtype)
        self.local_image_encoder.to(style_image.device)

        if self.freeze_image_encoder:
            with torch.no_grad():
                hidden_states = self.local_image_encoder(pixel_values).last_hidden_state
        else:
            hidden_states = self.local_image_encoder(pixel_values).last_hidden_state

        patch_tokens = hidden_states[:, 1:]
        if patch_tokens.shape[1] > self.dino_local_tokens:
            patch_tokens = patch_tokens[:, : self.dino_local_tokens]
        patch_tokens = patch_tokens.to(dtype=self.local_projection[0].weight.dtype)
        return self.local_projection(patch_tokens)

    def encode_dino_embedding(self, style_image: torch.Tensor) -> torch.Tensor:
        """Return pooled DINOv2 embedding for retrieval reranking.

        Output shape:
            ``[B, D_dino]``. This is intentionally kept in the native DINOv2
            hidden dimension so font databases can store rerank prototypes that
            are independent from the FLUX condition dimension.
        """

        pixel_values = self._preprocess_dino(style_image)
        encoder_dtype = next(self.local_image_encoder.parameters()).dtype
        pixel_values = pixel_values.to(device=style_image.device, dtype=encoder_dtype)
        self.local_image_encoder.to(style_image.device)

        if self.freeze_image_encoder:
            with torch.no_grad():
                hidden_states = self.local_image_encoder(pixel_values).last_hidden_state
        else:
            hidden_states = self.local_image_encoder(pixel_values).last_hidden_state

        patch_tokens = hidden_states[:, 1:]
        return F.normalize(patch_tokens.float().mean(dim=1), dim=-1)

    def encode_texture(self, style_image: torch.Tensor) -> torch.Tensor:
        """CNN texture branch: ``[B, 3, H, W] -> [B, N_texture, D]``."""

        texture_image = self._preprocess_siglip(style_image).to(dtype=self.texture_encoder.encoder[0].weight.dtype)
        return self.texture_encoder(texture_image)

    def forward(self, style_image: torch.Tensor) -> torch.Tensor:
        if self.uses_pretrained_projection():
            pooled = self.encode_image_embedding(style_image)
            pooled = pooled.to(dtype=self.pretrained_mlp_projection.proj[0].weight.dtype)
            mlp_tokens = self.pretrained_mlp_projection(pooled)
            qformer_tokens = self.pretrained_qformer_projection(pooled)
            return mlp_tokens + qformer_tokens

        global_style_tokens = self.encode_global_style(style_image)
        local_style_tokens = self.encode_local_style(style_image)
        texture_tokens = self.encode_texture(style_image)

        condition_tokens = torch.cat([global_style_tokens, local_style_tokens, texture_tokens], dim=1)
        return self.fusion(condition_tokens)
