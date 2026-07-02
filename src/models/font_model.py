from __future__ import annotations

import glob
import os

import torch
from torch import nn

from .flux_decoder import FluxDecoderConfig, FluxImageDecoder
from .retrieval import FontRetriever, render_text_with_font
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


class FontModel(nn.Module):
    """Font customization model with baseline and experiment modes.

    Experiment A forward inputs:
        style_image: ``[B, 3, H, W]``
        text_prompt: ``str | list[str]`` with batch length ``B``
        timestep: ``[B]``

    Experiment B forward inputs:
        style_image: ``[B, 3, H, W]``
        text_prompt: ``str | list[str]`` with batch length ``B``
        timestep: ``[B]``

    Forward output:
        output_image: ``[B, 3, H, W]``
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_encoder_path: str | None = None,
        condition_dim: int = 4096,
        image_embedding_dim: int = 1152,
        condition_tokens: int = 128,
        condition_heads: int = 8,
        condition_query_tokens: int = 32,
        text_max_length: int = 64,
        image_size: int = 384,
        output_size: int = 64,
        freeze_image_encoder: bool = True,
        num_heads: int = 24,
        decoder_blocks: int = 19,
        decoder_single_blocks: int = 38,
        flux_model_path: str | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        flux_dtype: str = "bfloat16",
        no_content_mode: bool = False,
        retrieval_content_mode: bool = False,
        font_db_path: str | None = None,
        font_root: str | None = None,
        retrieval_top_k: int = 1,
    ) -> None:
        super().__init__()
        if no_content_mode and retrieval_content_mode:
            raise ValueError("no_content_mode and retrieval_content_mode are mutually exclusive.")

        self.image_channels = image_channels
        self.output_size = output_size
        self.no_content_mode = no_content_mode
        self.retrieval_content_mode = retrieval_content_mode
        self.font_root = font_root
        self.retrieval_top_k = retrieval_top_k
        image_encoder_path = _resolve_image_encoder_path(image_encoder_path)
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
        )
        self.font_retriever = None
        if retrieval_content_mode:
            self.font_retriever = FontRetriever(
                font_db_path=font_db_path,
                embedding_fn=self.style_encoder.encode_image_embedding,
                cache_enabled=True,
            )
        self.decoder = FluxImageDecoder(
            FluxDecoderConfig(
                pretrained_model_name_or_path=flux_model_path,
                hidden_dim=condition_dim,
                height=output_size,
                width=output_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=text_max_length,
                torch_dtype=flux_dtype,
            )
        )

    def _normalize_prompts(self, text_prompt: str | list[str], batch_size: int) -> list[str]:
        if isinstance(text_prompt, str):
            return [text_prompt for _ in range(batch_size)]
        if not isinstance(text_prompt, list):
            raise TypeError("text_prompt must be a string or list of strings.")
        if len(text_prompt) != batch_size:
            raise ValueError(f"text batch {len(text_prompt)} does not match image batch {batch_size}")
        return text_prompt

    def forward(
        self,
        first_image: torch.Tensor,
        second_input: torch.Tensor | str | list[str],
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run baseline or an experiment branch.

        Baseline, ``no_content_mode=False``:
            ``forward(content_image, style_image, timestep)``.

        Experiment A, ``no_content_mode=True``:
            ``forward(style_image, text_prompt, timestep)``.

        Experiment B, ``retrieval_content_mode=True``:
            ``forward(style_image, text_prompt, timestep)``.
        """

        if self.no_content_mode:
            return self.forward_no_content(first_image, second_input, timestep=timestep)
        if self.retrieval_content_mode:
            return self.forward_retrieval_content(first_image, second_input, timestep=timestep)
        if not isinstance(second_input, torch.Tensor):
            raise TypeError("Baseline mode expects second_input to be style_image tensor.")
        return self.forward_baseline(first_image, second_input, timestep=timestep)

    def forward_no_content(
        self,
        style_image: torch.Tensor,
        text_prompt: str | list[str],
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Experiment A: text + style image only, no content image branch."""

        if not self.no_content_mode:
            raise RuntimeError("forward_no_content requires no_content_mode=True.")

        prompts = self._normalize_prompts(text_prompt, style_image.shape[0])
        style_tokens = self.style_encoder(style_image)

        # Condition order for experiment A:
        # [FLUX text tokens, style tokens] -> FLUX encoder_hidden_states.
        return self.decoder(prompts, extra_condition_tokens=style_tokens)

    def _render_retrieved_content(
        self,
        prompts: list[str],
        retrieved_font_ids: list[list[str]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Render retrieved-font content images.

        Output shape:
            ``[B, 3, output_size, output_size]``.
        """

        images = []
        for prompt, font_ids in zip(prompts, retrieved_font_ids):
            if not font_ids:
                raise RuntimeError("Font retrieval returned an empty result.")
            images.append(
                render_text_with_font(
                    prompt,
                    font_ids[0],
                    font_root=self.font_root,
                    image_size=self.output_size,
                )
            )
        return torch.stack(images).to(device=device, dtype=dtype)

    def forward_retrieval_content(
        self,
        style_image: torch.Tensor,
        text_prompt: str | list[str],
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Experiment B: retrieve a font prior, render content, then decode.

        Retrieval path:
            style_image -> font retriever -> top-k font ids -> rendered content
            image -> FLUX img2img latent prior.

        Condition order:
            [FLUX text tokens, style tokens].
        """

        if not self.retrieval_content_mode:
            raise RuntimeError("forward_retrieval_content requires retrieval_content_mode=True.")
        if self.font_retriever is None or not self.font_retriever.has_database():
            raise RuntimeError("retrieval_content_mode requires a loaded font_db.pt.")
        prompts = self._normalize_prompts(text_prompt, style_image.shape[0])
        retrieved_font_ids = self.font_retriever(style_image, k=self.retrieval_top_k)
        retrieved_content_image = self._render_retrieved_content(
            prompts,
            retrieved_font_ids,
            device=style_image.device,
            dtype=style_image.dtype,
        )

        style_tokens = self.style_encoder(style_image)
        return self.decoder(prompts, extra_condition_tokens=style_tokens, image=retrieved_content_image)

    def forward_baseline(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Baseline path: content image + style image, no text condition."""

        style_tokens = self.style_encoder(style_image)
        prompts = ["" for _ in range(content_image.shape[0])]
        return self.decoder(prompts, extra_condition_tokens=style_tokens, image=content_image)
