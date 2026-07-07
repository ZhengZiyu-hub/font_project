from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .font_renderer import render_text_with_font


@dataclass
class RetrievedContentPrior:
    """Rendered retrieval content prior for branch B.

    Shapes:
        rendered_images: ``[B, K, 3, H, W]`` top-k rendered glyph images.
        fused_image: ``[B, 3, H, W]`` weighted image blend.
        latent: optional VAE latent. Shape depends on the VAE backend, usually
            ``[B, C_latent, H_latent, W_latent]``.
        weights: ``[B, K]`` blending weights derived from retrieval scores.
    """

    rendered_images: torch.Tensor
    fused_image: torch.Tensor
    latent: torch.Tensor | None
    weights: torch.Tensor
    font_ids: list[list[str]]


def _as_batch_text(text_prompt: str | Sequence[str]) -> list[str]:
    if isinstance(text_prompt, str):
        return [text_prompt]
    prompts = [str(text) for text in text_prompt]
    if not prompts:
        raise ValueError("text_prompt must not be empty.")
    return prompts


def _as_batch_fonts(font_ids: Sequence[str] | Sequence[Sequence[str]], batch_size: int) -> list[list[str]]:
    if not font_ids:
        raise ValueError("font_ids must not be empty.")
    first = font_ids[0]
    if isinstance(first, str):
        fonts = [str(font_id) for font_id in font_ids]  # type: ignore[arg-type]
        return [fonts for _ in range(batch_size)]
    rows = [[str(font_id) for font_id in row] for row in font_ids]  # type: ignore[union-attr]
    if len(rows) != batch_size:
        raise ValueError(f"font_ids batch {len(rows)} does not match text batch {batch_size}.")
    if any(len(row) == 0 for row in rows):
        raise ValueError("Each font_ids row must contain at least one font id.")
    return rows


def _as_score_tensor(
    scores: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None,
    fonts: list[list[str]],
    device: torch.device,
) -> torch.Tensor:
    max_k = max(len(row) for row in fonts)
    if scores is None:
        score_tensor = torch.zeros(len(fonts), max_k, device=device)
        for row_idx, row in enumerate(fonts):
            score_tensor[row_idx, : len(row)] = 1.0
        return score_tensor

    if isinstance(scores, torch.Tensor):
        score_tensor = scores.float().to(device)
        if score_tensor.ndim == 1:
            score_tensor = score_tensor.unsqueeze(0).expand(len(fonts), -1)
    else:
        if not scores:
            raise ValueError("scores must not be empty when provided.")
        first = scores[0]  # type: ignore[index]
        if isinstance(first, (int, float)):
            score_tensor = torch.tensor(scores, dtype=torch.float32, device=device).unsqueeze(0).expand(len(fonts), -1)
        else:
            score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)

    if score_tensor.ndim != 2:
        raise ValueError(f"scores must have shape [K] or [B, K], got {tuple(score_tensor.shape)}.")
    if score_tensor.shape[0] != len(fonts):
        raise ValueError(f"scores batch {score_tensor.shape[0]} does not match font_ids batch {len(fonts)}.")
    if score_tensor.shape[1] < max_k:
        pad = torch.full((score_tensor.shape[0], max_k - score_tensor.shape[1]), float("-inf"), device=device)
        score_tensor = torch.cat([score_tensor, pad], dim=1)
    return score_tensor[:, :max_k]


class RetrievedContentPriorGenerator(nn.Module):
    """Render and fuse top-k retrieved fonts into a content prior.

    Inputs:
        text_prompt: ``str`` or ``list[str]``.
        font_ids: top-k font ids, either ``[K]`` for a shared top-k list or
            ``[B, K]`` for per-sample retrieval.
        scores: optional retrieval scores. If provided, softmax(scores) becomes
            the blending weight. If omitted, fonts are blended uniformly.

    Outputs:
        ``RetrievedContentPrior`` containing top-k pixel images, fused pixel
        image and optional VAE latent.
    """

    def __init__(
        self,
        font_root: str | Path | None = None,
        image_size: int = 1024,
        font_size: int | None = None,
        vae: nn.Module | None = None,
        cache_enabled: bool = True,
        blend_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if blend_temperature <= 0:
            raise ValueError("blend_temperature must be positive.")
        self.font_root = font_root
        self.image_size = image_size
        self.font_size = font_size
        self.vae = vae
        self.cache_enabled = cache_enabled
        self.blend_temperature = blend_temperature
        self._render_cache: dict[tuple[str, str, int, int | None, str | None], torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._render_cache.clear()

    def _cache_key(self, text: str, font_id: str) -> tuple[str, str, int, int | None, str | None]:
        font_root = str(self.font_root) if self.font_root is not None else None
        return (text, font_id, self.image_size, self.font_size, font_root)

    def _render_cached(self, text: str, font_id: str) -> torch.Tensor:
        key = self._cache_key(text, font_id)
        if self.cache_enabled and key in self._render_cache:
            return self._render_cache[key].clone()

        image = render_text_with_font(
            text=text,
            font_id=font_id,
            font_root=self.font_root,
            image_size=self.image_size,
            font_size=self.font_size,
        )
        if self.cache_enabled:
            self._render_cache[key] = image.detach().cpu()
        return image

    def _render_batch(self, prompts: list[str], fonts: list[list[str]], device: torch.device) -> torch.Tensor:
        max_k = max(len(row) for row in fonts)
        rendered_rows: list[torch.Tensor] = []
        for text, row in zip(prompts, fonts):
            images = [self._render_cached(text, font_id) for font_id in row]
            while len(images) < max_k:
                images.append(images[-1].clone())
            rendered_rows.append(torch.stack(images, dim=0))
        return torch.stack(rendered_rows, dim=0).to(device=device)

    def _blend(self, rendered_images: torch.Tensor, scores: torch.Tensor, fonts: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_k = rendered_images.shape[1]
        valid_mask = torch.zeros(rendered_images.shape[0], max_k, dtype=torch.bool, device=rendered_images.device)
        for row_idx, row in enumerate(fonts):
            valid_mask[row_idx, : len(row)] = True

        masked_scores = scores.to(rendered_images.device).masked_fill(~valid_mask, float("-inf"))
        weights = F.softmax(masked_scores / self.blend_temperature, dim=-1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        fused_image = (rendered_images * weights[:, :, None, None, None]).sum(dim=1)
        return fused_image.clamp(-1.0, 1.0), weights

    def _encode_latent(self, fused_image: torch.Tensor) -> torch.Tensor | None:
        if self.vae is None:
            return None

        vae = self.vae.to(fused_image.device)
        vae_dtype = next(vae.parameters()).dtype
        image = fused_image.to(dtype=vae_dtype)
        with torch.no_grad():
            encoded = vae.encode(image)
            if hasattr(encoded, "latent_dist"):
                latent = encoded.latent_dist.sample()
            elif isinstance(encoded, tuple):
                latent = encoded[0]
            else:
                latent = encoded

        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
        shift_factor = getattr(getattr(vae, "config", None), "shift_factor", None)
        if scaling_factor is not None:
            if shift_factor is not None:
                latent = (latent - shift_factor) * scaling_factor
            else:
                latent = latent * scaling_factor
        return latent

    def forward(
        self,
        text_prompt: str | Sequence[str],
        font_ids: Sequence[str] | Sequence[Sequence[str]],
        scores: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
        device: torch.device | str | None = None,
    ) -> RetrievedContentPrior:
        prompts = _as_batch_text(text_prompt)
        fonts = _as_batch_fonts(font_ids, batch_size=len(prompts))
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        rendered_images = self._render_batch(prompts, fonts, target_device)
        score_tensor = _as_score_tensor(scores, fonts, target_device)
        fused_image, weights = self._blend(rendered_images, score_tensor, fonts)
        latent = self._encode_latent(fused_image)
        return RetrievedContentPrior(
            rendered_images=rendered_images,
            fused_image=fused_image,
            latent=latent,
            weights=weights,
            font_ids=fonts,
        )


def retrieved_content_prior_generator(
    text_prompt: str | Sequence[str],
    font_ids: Sequence[str] | Sequence[Sequence[str]],
    scores: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    font_root: str | Path | None = None,
    image_size: int = 1024,
    font_size: int | None = None,
    vae: nn.Module | None = None,
    cache_enabled: bool = True,
    blend_temperature: float = 1.0,
    device: torch.device | str | None = None,
) -> RetrievedContentPrior:
    """Functional wrapper for one-shot retrieved content prior generation."""

    generator = RetrievedContentPriorGenerator(
        font_root=font_root,
        image_size=image_size,
        font_size=font_size,
        vae=vae,
        cache_enabled=cache_enabled,
        blend_temperature=blend_temperature,
    )
    return generator(text_prompt=text_prompt, font_ids=font_ids, scores=scores, device=device)
