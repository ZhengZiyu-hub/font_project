from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.font_model import FontModel
from src.utils.config import load_config


@dataclass
class InferenceResult:
    image: Image.Image
    metadata: dict[str, Any]


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_style_image(file_bytes: bytes, image_size: int = 384) -> torch.Tensor:
    """Decode uploaded style image to ``[1, 3, H, W]`` in ``[-1, 1]``."""

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


class FluxCustomInferenceRunner:
    """Real FLUX inference runner for branch A and branch B.

    Branch A:
        style image -> StyleEncoder
        text -> FLUX text encoder
        glyph_tokens=None
        FLUX denoising -> VAE decode

    Branch B:
        style image -> FontRetriever
        text + top-k fonts -> rendered content prior
        rendered prior -> VAE latent -> packed initial FLUX latents
        FLUX denoising -> VAE decode
    """

    def __init__(self, config_path: str | Path = "configs/base.yaml", device: str | None = None) -> None:
        config = load_config(config_path)
        model_cfg = dict(config.get("model", {}))
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(model_cfg.get("image_size", 384))
        self.output_size = int(model_cfg.get("content_prior_image_size") or model_cfg.get("output_size", 1024))
        self._models: dict[str, FontModel] = {}

    def _model_kwargs(self, branch: str) -> dict[str, Any]:
        model_cfg = dict(self.config.get("model", {}))
        style_encoder_cfg = dict(self.config.get("style_encoder", {}))
        allowed = {
            "mode",
            "image_encoder_path",
            "dino_model_path",
            "condition_dim",
            "image_embedding_dim",
            "condition_tokens",
            "condition_heads",
            "condition_query_tokens",
            "image_size",
            "output_size",
            "flux_model_path",
            "flux_dtype",
            "use_routed_conditioning",
            "routed_initial_gates",
            "font_db_path",
            "retrieval_top_k",
            "retrieval_coarse_k",
            "content_prior_image_size",
            "content_prior_font_root",
            "content_prior_blend_temperature",
            "content_prior_cache",
            "glyph_tokens",
            "glyph_max_length",
            "glyph_heads",
            "glyph_layers",
            "radical_map_path",
        }
        kwargs = {key: value for key, value in model_cfg.items() if key in allowed}
        if "pretrained_path" in style_encoder_cfg:
            kwargs["style_encoder_pretrained_path"] = style_encoder_cfg.get("pretrained_path")
        if "freeze" in style_encoder_cfg:
            kwargs["style_encoder_freeze"] = bool(style_encoder_cfg.get("freeze"))
        kwargs["mode"] = branch
        if branch == "B" and not kwargs.get("font_db_path"):
            raise RuntimeError("Branch B requires model.font_db_path in configs/base.yaml.")
        return kwargs

    def get_model(self, branch: str) -> FontModel:
        branch = branch.upper()
        if branch not in {"A", "B"}:
            raise ValueError("branch must be A or B.")
        if branch not in self._models:
            model = FontModel(**self._model_kwargs(branch))
            model.eval()
            model.to(self.device)
            self._models[branch] = model
        return self._models[branch]

    def _pack_prior_latent(self, model: FontModel, prior_latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
        pipe = model.decoder.pipe
        num_channels_latents = model.decoder.transformer.config.in_channels // 4
        latent_height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
        prior_latent = F.interpolate(prior_latent.float(), size=(latent_height, latent_width), mode="bilinear")
        prior_latent = prior_latent.to(device=model.decoder.device, dtype=model.decoder.dtype)
        return pipe._pack_latents(prior_latent, prior_latent.shape[0], num_channels_latents, latent_height, latent_width)

    @torch.no_grad()
    def infer(
        self,
        branch: str,
        style_image: torch.Tensor,
        text_prompt: str,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> InferenceResult:
        branch = branch.upper()
        model = self.get_model(branch)
        pipe = model.decoder.pipe
        device = model.decoder.device
        dtype = model.decoder.dtype
        height = height or self.output_size
        width = width or self.output_size
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        style_image = style_image.to(device=device, dtype=torch.float32)
        text_tokens, pooled, txt_ids = model.encode_text([text_prompt])
        style_tokens = model.encode_style(style_image)
        metadata: dict[str, Any] = {
            "branch": branch,
            "text": text_prompt,
            "text_tokens": list(text_tokens.shape),
            "style_tokens": list(style_tokens.shape),
        }

        initial_latents = None
        if branch == "A":
            glyph_tokens = None
            metadata["glyph_tokens"] = None
            print("glyph_tokens=None")
        else:
            glyph_tokens, glyph_metadata = model.prepare_experiment_glyph_tokens(
                text_prompt=[text_prompt],
                style_image=style_image,
                mode="B",
            )
            prior = glyph_metadata["content_prior"]
            retrieval_results = glyph_metadata["retrieval_results"]
            if prior.latent is None:
                raise RuntimeError("Branch B content prior generator did not return a VAE latent.")
            initial_latents = self._pack_prior_latent(model, prior.latent, height, width)
            glyph_tokens = glyph_tokens.to(device=device, dtype=dtype)
            metadata["glyph_tokens"] = list(glyph_tokens.shape)
            metadata["retrieved_fonts"] = [
                {"font_ids": result.font_ids, "scores": result.scores} for result in retrieval_results
            ]
            metadata["content_prior_image"] = image_to_base64(
                pipe.image_processor.postprocess(prior.fused_image, output_type="pil")[0]
            )

        text_tokens = text_tokens.to(device=device, dtype=dtype)
        pooled = pooled.to(device=device, dtype=dtype)
        txt_ids = txt_ids.to(device=device, dtype=dtype)
        condition_tokens, routed_txt_ids, branch_lengths = model.decoder._build_condition_tokens(
            text_tokens, style_tokens, glyph_tokens, {"txt_ids": txt_ids}
        )

        batch_size = 1
        num_channels_latents = model.decoder.transformer.config.in_channels // 4
        latents, img_ids = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            initial_latents,
        )

        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        guidance = None
        if model.decoder.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)

        joint_attention_kwargs = {"branch_lengths": branch_lengths}
        pipe.scheduler.set_begin_index(0)
        for timestep_value in timesteps:
            timestep = timestep_value.expand(latents.shape[0]).to(dtype)
            noise_pred = model.decoder.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=condition_tokens,
                txt_ids=routed_txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

        unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        unpacked = (unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(unpacked, return_dict=False)[0]
        pil_image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        metadata["output_size"] = [pil_image.width, pil_image.height]
        return InferenceResult(image=pil_image, metadata=metadata)
