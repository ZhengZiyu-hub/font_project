from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class FluxDecoderConfig:
    """Configuration for the Diffusers FLUX backend."""

    pretrained_model_name_or_path: str
    image_channels: int = 3
    hidden_dim: int = 4096
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    max_sequence_length: int = 512
    torch_dtype: str = "bfloat16"
    device: str | None = None


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported FLUX dtype: {name}")


class FluxImageDecoder(nn.Module):
    """Full Diffusers FLUX backend.

    This module wraps the actual FLUX pipeline components:

    prompt/style/content conditions
        -> T5/CLIP prompt embeddings + extra condition tokens
        -> packed FLUX latents
        -> FlowMatch scheduler denoising loop
        -> VAE decode
        -> image tensor ``[B, 3, H, W]``.

    ``extra_condition_tokens`` may contain style tokens, retrieved-content
    tokens, or both. They are concatenated to the official FLUX text embeddings
    along the sequence dimension and enter ``FluxTransformer2DModel`` through
    ``encoder_hidden_states``.
    """

    def __init__(self, config: FluxDecoderConfig) -> None:
        super().__init__()
        if not config.pretrained_model_name_or_path:
            raise ValueError("FluxImageDecoder now requires a real Diffusers FLUX checkpoint path.")

        self.config = config
        dtype = _resolve_dtype(config.torch_dtype)
        try:
            from diffusers import FluxImg2ImgPipeline, FluxPipeline
        except ImportError as exc:
            raise ImportError("FluxImageDecoder requires diffusers with FluxPipeline support.") from exc

        self.pipe = FluxPipeline.from_pretrained(config.pretrained_model_name_or_path, torch_dtype=dtype)
        self.img2img_pipe = FluxImg2ImgPipeline(**self.pipe.components)
        if config.device is not None:
            self.pipe.to(config.device)
            self.img2img_pipe.to(config.device)

        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae

        flux_condition_dim = int(self.transformer.config.joint_attention_dim)
        self.to_flux_condition = (
            nn.Identity() if config.hidden_dim == flux_condition_dim else nn.Linear(config.hidden_dim, flux_condition_dim)
        )

    def set_trainable(
        self,
        transformer: bool = True,
        text_encoders: bool = False,
        vae: bool = False,
        condition_projection: bool = True,
    ) -> None:
        """Select trainable FLUX components for fine-tuning.

        Full FLUX fine-tuning usually trains ``transformer`` or LoRA adapters
        on it, keeps text encoders frozen, and keeps the VAE frozen. This method
        only toggles ``requires_grad``; optimizer construction remains in the
        training script.
        """

        for param in self.transformer.parameters():
            param.requires_grad_(transformer)
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

    @property
    def device(self) -> torch.device:
        return self.pipe._execution_device

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    def _condition_ids(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build neutral FLUX ids for additional non-image condition tokens."""

        return torch.zeros(length, 3, device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: str | list[str],
        extra_condition_tokens: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        lora_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text and append project-specific condition tokens.

        Shapes:
            prompt_embeds: ``[B, T_text, 4096]``
            extra_condition_tokens: ``[B, T_extra, D]``
            output prompt embeds: ``[B, T_text + T_extra, 4096]``
        """

        device = self.device
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length or self.config.max_sequence_length,
            lora_scale=lora_scale,
        )

        if extra_condition_tokens is None:
            return prompt_embeds, pooled_prompt_embeds, text_ids

        extra_condition_tokens = extra_condition_tokens.to(device=device, dtype=prompt_embeds.dtype)
        extra_condition_tokens = self.to_flux_condition(extra_condition_tokens)
        if extra_condition_tokens.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(
                f"extra condition batch {extra_condition_tokens.shape[0]} does not match prompt batch {prompt_embeds.shape[0]}"
            )

        extra_ids = self._condition_ids(extra_condition_tokens.shape[1], device, text_ids.dtype)
        prompt_embeds = torch.cat([prompt_embeds, extra_condition_tokens], dim=1)
        text_ids = torch.cat([text_ids, extra_ids], dim=0)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    @torch.no_grad()
    def forward(
        self,
        prompt: str | list[str],
        extra_condition_tokens: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        strength: float = 0.6,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run the real FLUX denoising loop and return image tensor."""

        if image is not None:
            return self.forward_img2img(
                prompt=prompt,
                image=image,
                extra_condition_tokens=extra_condition_tokens,
                height=height,
                width=width,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                latents=latents,
                output_type=output_type,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        joint_attention_kwargs = joint_attention_kwargs or {}

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)
        device = self.device

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt_list,
            extra_condition_tokens=extra_condition_tokens,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([latents.shape[0]], guidance_scale, device=device, dtype=torch.float32)

        self.scheduler.set_begin_index(0)
        for timestep_value in timesteps:
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

        if output_type == "latent":
            return latents

        latents = self.pipe._unpack_latents(latents, height, width, self.pipe.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image, output_type=output_type)

    @torch.no_grad()
    def forward_img2img(
        self,
        prompt: str | list[str],
        image: torch.Tensor,
        extra_condition_tokens: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        strength: float = 0.6,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run official FLUX img2img using content image as latent prior."""

        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, pooled_prompt_embeds, _ = self.encode_prompt(
            prompt_list,
            extra_condition_tokens=extra_condition_tokens,
        )

        # Diffusers img2img accepts tensors in [0, 1]. Project images in
        # [-1, 1] into that range while leaving [0, 1] tensors untouched.
        image = image.to(device=self.device, dtype=prompt_embeds.dtype)
        if image.detach().amin() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0.0, 1.0)

        result = self.img2img_pipe(
            prompt=None,
            prompt_2=None,
            image=image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type=output_type,
            return_dict=True,
            joint_attention_kwargs=joint_attention_kwargs or {},
            max_sequence_length=self.config.max_sequence_length,
        )
        return result.images

    def training_denoise_forward(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Trainable transformer forward for fine-tuning loops.

        This is the hook a training script should call after preparing noisy
        packed latents and flow-matching targets. It returns predicted velocity
        or noise with the same shape as ``latents``.
        """

        guidance = None
        if self.transformer.config.guidance_embeds:
            value = guidance_scale if guidance_scale is not None else self.config.guidance_scale
            guidance = torch.full([latents.shape[0]], value, device=latents.device, dtype=torch.float32)

        return self.transformer(
            hidden_states=latents,
            timestep=timestep.to(device=latents.device, dtype=latents.dtype) / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs or {},
            return_dict=False,
        )[0]
