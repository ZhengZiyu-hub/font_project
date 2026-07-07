from __future__ import annotations

import glob
import os

import torch
from torch import nn

from .content_encoder import ContentEncoder
from .experiment_modes import normalize_experiment_mode, print_experiment_mode, resolve_experiment_mode
from .flux_decoder import FluxDecoderConfig, FluxImageDecoder
from .glyph_encoder import GlyphEncoder
from .retrieval import FontRetriever, RetrievedContentPriorGenerator
from .style_encoder import StyleEncoder


def _resolve_image_encoder_path(image_encoder_path: str | None) -> str:
    if image_encoder_path:
        return image_encoder_path

    env_path = os.environ.get("FONT_IMAGE_ENCODER_PATH")
    if env_path:
        return env_path

    preferred_path = "/data/zhengziyu/models/siglip-so400m-patch14-384"
    if os.path.isdir(preferred_path):
        return preferred_path

    matches = glob.glob("/data/zhengziyu/**/siglip-so400m-patch14-384", recursive=True)
    if matches:
        return sorted(matches)[0]

    return preferred_path


def _resolve_flux_model_path(flux_model_path: str | None) -> str:
    if flux_model_path:
        return flux_model_path

    env_path = os.environ.get("FONT_FLUX_MODEL_PATH")
    if env_path:
        return env_path

    preferred_path = "/data/zhengziyu/models/FLUX.1-dev"
    if os.path.isdir(preferred_path):
        return preferred_path

    matches = glob.glob("/data/zhengziyu/**/FLUX.1-dev", recursive=True)
    if matches:
        return sorted(matches)[0]

    return preferred_path


def _resolve_dino_model_path(dino_model_path: str | None) -> str:
    if dino_model_path:
        return dino_model_path

    env_path = os.environ.get("FONT_DINO_MODEL_PATH")
    if env_path:
        return env_path

    preferred_path = "/data/zhengziyu/models/dinov2-base"
    if os.path.isdir(preferred_path):
        return preferred_path

    matches = glob.glob("/data/zhengziyu/**/dinov2-base", recursive=True)
    if matches:
        return sorted(matches)[0]

    return "facebook/dinov2-base"


class FontModel(nn.Module):
    """Unified condition-token interface for FLUX font experiments.

    Main forward signature:

    ``forward(latents, text_tokens, style_tokens, glyph_tokens, timestep, ids)``

    ``glyph_tokens`` is required even when empty. Baseline mode should pass
    ``None`` to ``forward_baseline`` or an explicit empty tensor ``[B, 0, D]`` to
    ``forward``.
    """

    def __init__(
        self,
        mode: str = "baseline",
        image_encoder_path: str | None = None,
        dino_model_path: str | None = None,
        condition_dim: int = 4096,
        image_embedding_dim: int = 1152,
        condition_tokens: int = 128,
        condition_heads: int = 8,
        condition_query_tokens: int = 32,
        text_max_length: int = 512,
        image_size: int = 384,
        output_size: int = 1024,
        freeze_image_encoder: bool = True,
        style_encoder_pretrained_path: str | None = None,
        style_encoder_freeze: bool = True,
        flux_model_path: str | None = None,
        flux_dtype: str = "bfloat16",
        use_routed_conditioning: bool = True,
        routed_initial_gates: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_glyph_prior: bool = False,
        use_retrieval_prior: bool = False,
        font_db_path: str | None = None,
        retrieval_top_k: int = 1,
        retrieval_coarse_k: int = 32,
        content_prior_image_size: int = 1024,
        content_prior_font_root: str | None = None,
        content_prior_blend_temperature: float = 1.0,
        content_prior_cache: bool = True,
        glyph_tokens: int = 32,
        glyph_max_length: int = 64,
        glyph_heads: int = 8,
        glyph_layers: int = 2,
        radical_map_path: str | None = None,
    ) -> None:
        super().__init__()
        if mode == "baseline" and use_retrieval_prior:
            mode = "B"
        elif mode == "baseline" and use_glyph_prior:
            mode = "baseline"
        mode_config = resolve_experiment_mode(mode)
        use_glyph_prior = mode_config.use_glyph_prior
        use_retrieval_prior = mode_config.use_retrieval_prior

        self.mode = mode_config.mode
        self.condition_dim = condition_dim
        self.output_size = output_size
        self.use_glyph_prior = use_glyph_prior
        self.use_retrieval_prior = use_retrieval_prior
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_coarse_k = retrieval_coarse_k
        self.content_prior_image_size = content_prior_image_size

        image_encoder_path = _resolve_image_encoder_path(image_encoder_path)
        dino_model_path = _resolve_dino_model_path(dino_model_path)
        flux_model_path = _resolve_flux_model_path(flux_model_path)

        self.style_encoder = StyleEncoder(
            image_encoder_path=image_encoder_path,
            cross_attention_dim=condition_dim,
            id_embeddings_dim=image_embedding_dim,
            num_tokens=condition_tokens,
            num_heads=condition_heads,
            num_query_tokens=condition_query_tokens,
            image_size=image_size,
            freeze_image_encoder=freeze_image_encoder,
            pretrained_path=style_encoder_pretrained_path,
            freeze=style_encoder_freeze,
            dino_model_path=dino_model_path,
        )
        self.font_retriever = None
        self.glyph_encoder = None
        self.content_encoder = None
        if use_glyph_prior:
            self.glyph_encoder = GlyphEncoder(
                hidden_dim=condition_dim,
                num_tokens=glyph_tokens,
                max_length=glyph_max_length,
                num_heads=glyph_heads,
                num_layers=glyph_layers,
                radical_map_path=radical_map_path,
            )
        if mode_config.use_content_encoder:
            self.content_encoder = ContentEncoder(hidden_dim=condition_dim, num_tokens=glyph_tokens)
        if use_retrieval_prior:
            self.font_retriever = FontRetriever(
                font_db_path=font_db_path,
                clip_embedding_fn=self.style_encoder.encode_image_embedding,
                dino_embedding_fn=self.style_encoder.encode_dino_embedding,
                cache_enabled=True,
                coarse_k=retrieval_coarse_k,
            )

        self.decoder = FluxImageDecoder(
            FluxDecoderConfig(
                pretrained_model_name_or_path=flux_model_path,
                hidden_dim=condition_dim,
                height=output_size,
                width=output_size,
                max_sequence_length=text_max_length,
                torch_dtype=flux_dtype,
                use_routed_conditioning=use_routed_conditioning,
                routed_initial_gates=routed_initial_gates,
            )
        )
        self.content_prior_generator = None
        if use_retrieval_prior:
            self.content_prior_generator = RetrievedContentPriorGenerator(
                font_root=content_prior_font_root,
                image_size=content_prior_image_size,
                vae=self.decoder.vae,
                cache_enabled=content_prior_cache,
                blend_temperature=content_prior_blend_temperature,
            )
        print_experiment_mode(self.mode)

    def empty_glyph_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return required glyph placeholder ``[B, 0, D]``."""

        return self.decoder.empty_glyph_tokens(batch_size, self.condition_dim, device=device, dtype=dtype)

    def encode_text(self, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use FLUX text encoders and return text tokens plus ids."""

        return self.decoder.encode_text(prompt)

    def encode_style(self, style_image: torch.Tensor) -> torch.Tensor:
        """Encode style image to style tokens ``[B, N_style, D]``."""

        return self.style_encoder(style_image)

    def encode_glyph(self, text_prompt: str | list[str]) -> torch.Tensor:
        """Encode prompt text into glyph prior tokens ``[B, N_glyph, D]``.

        Branch A uses this path instead of any external content image. The
        returned tokens can be concatenated with text/style tokens inside FLUX
        cross-attention.
        """

        if self.glyph_encoder is None:
            prompts = [text_prompt] if isinstance(text_prompt, str) else text_prompt
            return self.empty_glyph_tokens(len(prompts))
        return self.glyph_encoder(text_prompt)

    def encode_content_glyph(self, content_image: torch.Tensor) -> torch.Tensor:
        """Encode content/retrieval image into glyph tokens for baseline and B."""

        if self.content_encoder is None:
            raise RuntimeError("Content encoder is only enabled for baseline and mode B.")
        return self.content_encoder(content_image)

    def prepare_latents(
        self,
        batch_size: int,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create packed FLUX latents and ``img_ids``."""

        return self.decoder.prepare_latents(batch_size, generator=generator, latents=latents)

    def retrieve_fonts(self, style_image: torch.Tensor, k: int | None = None) -> list[list[str]]:
        """Return retrieval prior ids. A later glyph encoder can turn ids into tokens."""

        if self.font_retriever is None:
            raise RuntimeError("retrieve_fonts requires use_retrieval_prior=True and a font database.")
        return self.font_retriever(style_image, k=k or self.retrieval_top_k)

    def retrieve_fonts_with_scores(self, style_image: torch.Tensor, k: int | None = None):
        """Return top-k font ids plus retrieval scores for branch B."""

        if self.font_retriever is None:
            raise RuntimeError("retrieve_fonts_with_scores requires use_retrieval_prior=True and a font database.")
        return self.font_retriever.retrieve_from_image(style_image, k=k or self.retrieval_top_k)

    def generate_retrieved_content_prior(
        self,
        text_prompt: str | list[str],
        font_ids: list[str] | list[list[str]],
        scores: torch.Tensor | list[float] | list[list[float]] | None = None,
    ):
        """Render top-k retrieved fonts and return pixel prior plus VAE latent.

        Output contains:
            rendered_images: ``[B, K, 3, H, W]``.
            fused_image: ``[B, 3, H, W]``.
            latent: VAE encoded latent if FLUX VAE is available.
            weights: ``[B, K]`` blend weights.
        """

        if self.content_prior_generator is None:
            raise RuntimeError("generate_retrieved_content_prior requires use_retrieval_prior=True.")
        return self.content_prior_generator(
            text_prompt=text_prompt,
            font_ids=font_ids,
            scores=scores,
            device=self.decoder.device,
        )

    def retrieve_and_generate_content_prior(
        self,
        style_image: torch.Tensor,
        text_prompt: str | list[str],
        k: int | None = None,
    ):
        """Branch B end-to-end helper: retrieve fonts, render, blend and VAE encode."""

        retrieval_results = self.retrieve_fonts_with_scores(style_image, k=k)
        font_ids = [result.font_ids for result in retrieval_results]
        scores = [result.scores for result in retrieval_results]
        prior = self.generate_retrieved_content_prior(text_prompt=text_prompt, font_ids=font_ids, scores=scores)
        return prior, retrieval_results

    def prepare_experiment_glyph_tokens(
        self,
        text_prompt: str | list[str],
        style_image: torch.Tensor | None = None,
        content_image: torch.Tensor | None = None,
        mode: str | None = None,
    ):
        """Prepare glyph condition for the active experiment.

        A returns ``None``. No empty glyph tensor is created.
        B returns tokens from retrieval-rendered prior.
        baseline returns tokens from the provided content image.
        """

        active_mode = normalize_experiment_mode(mode or self.mode)
        if active_mode == "A":
            return None, {"mode": "A", "glyph_source": None}
        if active_mode == "B":
            if style_image is None:
                raise ValueError("mode B requires style_image.")
            prior, retrieval_results = self.retrieve_and_generate_content_prior(style_image, text_prompt)
            glyph_tokens = self.encode_content_glyph(prior.fused_image)
            return glyph_tokens, {
                "mode": "B",
                "glyph_source": "retrieval_render",
                "content_prior": prior,
                "retrieval_results": retrieval_results,
            }
        if content_image is None:
            raise ValueError("baseline mode requires content_image.")
        return self.encode_content_glyph(content_image), {"mode": "baseline", "glyph_source": "content_image"}

    def forward(
        self,
        latents: torch.Tensor,
        text_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        glyph_tokens: torch.Tensor | None,
        timestep: torch.Tensor,
        ids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Unified token-level forward."""

        return self.decoder(latents, text_tokens, style_tokens, glyph_tokens, timestep, ids)

    def forward_baseline(
        self,
        latents: torch.Tensor,
        text_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        ids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Backward-compatible baseline: no glyph prior, but glyph slot exists."""

        glyph_tokens = self.empty_glyph_tokens(
            latents.shape[0],
            device=latents.device,
            dtype=style_tokens.dtype,
        )
        return self.forward(latents, text_tokens, style_tokens, glyph_tokens, timestep, ids)
