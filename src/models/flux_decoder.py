from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .flux_routed_attention import build_routed_flux_processors


@dataclass
class FluxDecoderConfig:
    """Configuration for the Diffusers FLUX transformer backend."""

    pretrained_model_name_or_path: str
    hidden_dim: int = 4096
    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 512
    torch_dtype: str = "bfloat16"
    device: str | None = None
    use_routed_conditioning: bool = True
    routed_initial_gates: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported FLUX dtype: {name}")


class FluxImageDecoder(nn.Module):
    """Token-level FLUX transformer wrapper.

    Unified forward contract:

    ``forward(latents, text_tokens, style_tokens, glyph_tokens, timestep, ids)``

    Shapes:
        latents: ``[B, L_img, C]`` packed FLUX latents.
        text_tokens: ``[B, L_text, D]`` FLUX text embeddings.
        style_tokens: ``[B, L_style, D_style]`` style condition tokens.
        glyph_tokens: optional ``[B, L_glyph, D_glyph]`` glyph prior tokens.
            In mode A this is ``None`` and no glyph branch is added.
        timestep: ``[B]`` or scalar timestep.
        ids: dictionary containing at least ``img_ids`` and optionally
            ``txt_ids``, ``pooled_projections`` and ``guidance``.

    Output:
        predicted noise/velocity tokens with shape matching ``latents``.
    """

    def __init__(self, config: FluxDecoderConfig) -> None:
        super().__init__()
        if not config.pretrained_model_name_or_path:
            raise ValueError("FluxImageDecoder requires a real Diffusers FLUX checkpoint path.")

        self.config = config
        dtype = _resolve_dtype(config.torch_dtype)
        try:
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImportError("FluxImageDecoder requires diffusers with FluxPipeline support.") from exc

        self.pipe = FluxPipeline.from_pretrained(config.pretrained_model_name_or_path, torch_dtype=dtype)
        if config.device is not None:
            self.pipe.to(config.device)

        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae

        flux_condition_dim = int(self.transformer.config.joint_attention_dim)
        self.to_flux_condition = (
            nn.Identity() if config.hidden_dim == flux_condition_dim else nn.Linear(config.hidden_dim, flux_condition_dim)
        )
        self.pooled_projection_dim = int(self.transformer.config.pooled_projection_dim)
        self.use_routed_conditioning = config.use_routed_conditioning
        self.routed_processors = None
        if self.use_routed_conditioning:
            self.routed_processors = build_routed_flux_processors(
                list(self.transformer.attn_processors.keys()),
                initial_gates=config.routed_initial_gates,
            )
            self.transformer.set_attn_processor(dict(self.routed_processors))

    @property
    def device(self) -> torch.device:
        return self.pipe._execution_device

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    def set_trainable(
        self,
        transformer: bool = True,
        text_encoders: bool = False,
        vae: bool = False,
        condition_projection: bool = True,
        routed_conditioning: bool = True,
    ) -> None:
        """Select trainable FLUX components for fine-tuning."""

        for param in self.transformer.parameters():
            param.requires_grad_(transformer)
        if self.routed_processors is not None:
            for processor in self.routed_processors.values():
                for param in processor.parameters():
                    param.requires_grad_(routed_conditioning)
        if self.pipe.text_encoder is not None:
            for param in self.pipe.text_encoder.parameters():
                param.requires_grad_(text_encoders)
        if self.pipe.text_encoder_2 is not None:
            for param in self.pipe.text_encoder_2.parameters():
                param.requires_grad_(text_encoders)
        for param in self.vae.parameters():
            param.requires_grad_(vae)
        for param in self.to_flux_condition.parameters():
            param.requires_grad_(condition_projection)

        self.transformer.train(transformer)
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.train(text_encoders)
        if self.pipe.text_encoder_2 is not None:
            self.pipe.text_encoder_2.train(text_encoders)
        self.vae.train(vae)

    def empty_glyph_tokens(self, batch_size: int, dim: int | None = None, device=None, dtype=None) -> torch.Tensor:
        """Create required glyph placeholder with shape ``[B, 0, D]``."""

        dim = dim or int(self.transformer.config.joint_attention_dim)
        return torch.empty(batch_size, 0, dim, device=device or self.device, dtype=dtype or self.dtype)

    def encode_text(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        lora_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use Diffusers FLUX prompt encoders to create text tokens.

        Returns:
            text_tokens, pooled_projections, txt_ids.
        """

        prompt_tokens, pooled, txt_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length or self.config.max_sequence_length,
            lora_scale=lora_scale,
        )
        return prompt_tokens, pooled, txt_ids

    def prepare_latents(
        self,
        batch_size: int,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create packed FLUX latents and image ids via Diffusers helpers."""

        num_channels_latents = self.transformer.config.in_channels // 4
        return self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height or self.config.height,
            width or self.config.width,
            self.dtype,
            self.device,
            generator,
            latents,
        )

    def _project_optional_tokens(self, tokens: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, L, D], got {tuple(tokens.shape)}")
        if tokens.shape[0] != batch_size:
            raise ValueError(f"{name} batch {tokens.shape[0]} does not match latent batch {batch_size}")
        if tokens.shape[1] == 0:
            return tokens.to(device=self.device, dtype=self.dtype)
        return self.to_flux_condition(tokens.to(device=self.device, dtype=self.dtype))

    def _condition_ids(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(length, 3, device=device, dtype=dtype)

    def _build_condition_tokens(
        self,
        text_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        glyph_tokens: torch.Tensor | None,
        ids: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        batch_size = text_tokens.shape[0]
        text_tokens = text_tokens.to(device=self.device, dtype=self.dtype)
        style_tokens = self._project_optional_tokens(style_tokens, batch_size, "style_tokens")
        if glyph_tokens is None:
            glyph_tokens = torch.empty(batch_size, 0, text_tokens.shape[-1], device=self.device, dtype=self.dtype)
        else:
            glyph_tokens = self._project_optional_tokens(glyph_tokens, batch_size, "glyph_tokens")

        branch_lengths = (text_tokens.shape[1], style_tokens.shape[1], glyph_tokens.shape[1])
        condition_tokens = torch.cat([text_tokens, style_tokens, glyph_tokens], dim=1)

        txt_ids = ids.get("txt_ids")
        if txt_ids is None:
            txt_ids = self._condition_ids(text_tokens.shape[1], self.device, self.dtype)
        else:
            txt_ids = txt_ids.to(device=self.device, dtype=self.dtype)

        extra_len = style_tokens.shape[1] + glyph_tokens.shape[1]
        if extra_len:
            txt_ids = torch.cat([txt_ids, self._condition_ids(extra_len, self.device, txt_ids.dtype)], dim=0)
        return condition_tokens, txt_ids, branch_lengths

    def forward(
        self,
        latents: torch.Tensor,
        text_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        glyph_tokens: torch.Tensor | None,
        timestep: torch.Tensor,
        ids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run one FLUX transformer denoising step with unified conditions."""

        latents = latents.to(device=self.device, dtype=self.dtype)
        condition_tokens, txt_ids, branch_lengths = self._build_condition_tokens(
            text_tokens, style_tokens, glyph_tokens, ids
        )

        img_ids = ids.get("img_ids")
        if img_ids is None:
            raise ValueError("ids must include img_ids for FLUX positional encoding.")
        img_ids = img_ids.to(device=self.device, dtype=self.dtype)

        pooled = ids.get("pooled_projections")
        if pooled is None:
            pooled = torch.zeros(latents.shape[0], self.pooled_projection_dim, device=self.device, dtype=self.dtype)
        else:
            pooled = pooled.to(device=self.device, dtype=self.dtype)

        guidance = ids.get("guidance")
        if guidance is None and self.transformer.config.guidance_embeds:
            guidance = torch.full([latents.shape[0]], 1.0, device=self.device, dtype=torch.float32)
        elif guidance is not None:
            guidance = guidance.to(device=self.device, dtype=torch.float32)

        timestep = timestep.to(device=self.device, dtype=self.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])

        joint_attention_kwargs = dict(ids.get("joint_attention_kwargs", {}))
        if self.use_routed_conditioning:
            joint_attention_kwargs["branch_lengths"] = branch_lengths

        return self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled,
            encoder_hidden_states=condition_tokens,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
