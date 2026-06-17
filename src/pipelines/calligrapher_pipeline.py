from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.fusion.style_injection import (
    DEFAULT_CALLIGRAPHER_ROOT,
    DEFAULT_CALLIGRAPHER_WEIGHTS,
    DEFAULT_FLUX_DEV_PATH,
    DEFAULT_FLUX_FILL_PATH,
    DEFAULT_SIGLIP_PATH,
    inspect_local_resources,
    is_complete_flux_model,
    require_calligrapher_on_path,
)


@dataclass(frozen=True)
class CalligrapherPaths:
    base_model_path: Path = DEFAULT_FLUX_DEV_PATH
    inpaint_model_path: Path = DEFAULT_FLUX_FILL_PATH
    image_encoder_path: Path = DEFAULT_SIGLIP_PATH
    calligrapher_path: Path = DEFAULT_CALLIGRAPHER_WEIGHTS
    calligrapher_root: Path = DEFAULT_CALLIGRAPHER_ROOT


def build_chinese_prompt(text: str, template: str = 'The Chinese text is "{text}".') -> str:
    return template.format(text=text)


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    return Image.open(path).convert(mode)


def binarize_mask(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask.convert("L"))
    mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def resize_img_and_pad(input_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    cropped_width, cropped_height = input_image.size
    target_width, target_height = target_size
    scale = min(target_width / cropped_width, target_height / cropped_height)
    new_width = max(1, int(cropped_width * scale))
    new_height = max(1, int(cropped_height * scale))
    resized_image = input_image.resize((new_width, new_height), Image.BILINEAR)
    padded_image = Image.new("RGB", target_size, (0, 0, 0))
    padded_image.paste(
        resized_image,
        ((target_width - new_width) // 2, (target_height - new_height) // 2),
    )
    return padded_image


def get_bbox_from_mask(mask_image: Image.Image) -> tuple[int, int, int, int] | None:
    mask = np.array(mask_image.convert("L")) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_context_reference(reference_image: Image.Image, width: int) -> Image.Image:
    ref_width, ref_height = reference_image.size
    new_height = max(16, round((width / ref_width) * ref_height / 16) * 16)
    return reference_image.resize((width, new_height), Image.BILINEAR)


def prepare_inference_images(
    image: Image.Image,
    mask: Image.Image,
    reference: Image.Image,
    width: int,
    height: int,
    use_context: bool = True,
) -> tuple[Image.Image, Image.Image, Image.Image, int]:
    source = image.convert("RGB").resize((width, height), Image.BILINEAR)
    mask_image = binarize_mask(mask).resize((width, height), Image.NEAREST)
    reference_to_encoder = resize_img_and_pad(reference.convert("RGB"), (512, 512))

    if not use_context:
        return source, mask_image, reference_to_encoder, 0

    reference_context = make_context_reference(reference.convert("RGB"), width)
    source_with_context = Image.new("RGB", (width, reference_context.height + height))
    source_with_context.paste(reference_context, (0, 0))
    source_with_context.paste(source, (0, reference_context.height))

    mask_with_context = Image.new("L", source_with_context.size, 0)
    mask_with_context.paste(mask_image, (0, reference_context.height))
    return source_with_context, mask_with_context, reference_to_encoder, reference_context.height


def smoke_inpaint(
    image: Image.Image,
    mask: Image.Image,
    reference: Image.Image,
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Cheap deterministic fallback for data-path validation.

    This is not a diffusion decoder. It copies a resized reference texture into
    the mask and preserves the unmasked background, so tests can verify
    reference_patch + masked_image + mask + target_text plumbing without large
    model weights.
    """

    base = image.convert("RGB")
    if output_size is not None:
        base = base.resize(output_size, Image.BILINEAR)
    mask_l = binarize_mask(mask).resize(base.size, Image.NEAREST)
    bbox = get_bbox_from_mask(mask_l)
    if bbox is None:
        return base
    x1, y1, x2, y2 = bbox
    ref = reference.convert("RGB").resize((x2 - x1, y2 - y1), Image.BILINEAR)
    patch_layer = Image.new("RGB", base.size)
    patch_layer.paste(ref, (x1, y1))
    return Image.composite(patch_layer, base, mask_l)


class CalligrapherGenerationPipeline:
    """Reference image + target text -> clean stylized Chinese text image."""

    def __init__(
        self,
        paths: CalligrapherPaths | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_tokens: int = 128,
        backend: str = "auto",
        mode: str = "generation",
    ) -> None:
        if backend not in {"auto", "flux", "smoke"}:
            raise ValueError("backend must be one of: auto, flux, smoke")
        if mode not in {"generation", "inpaint"}:
            raise ValueError("mode must be one of: generation, inpaint")
        self.paths = paths or CalligrapherPaths()
        self.device = device
        self.dtype = dtype
        self.num_tokens = num_tokens
        self.backend = backend
        self.mode = mode
        self.model = None
        self.pipe = None
        self.last_style_token_shape: tuple[int, ...] | None = None

    def resource_report(self) -> dict:
        return inspect_local_resources(
            calligrapher_root=self.paths.calligrapher_root,
            flux_fill_path=self.paths.inpaint_model_path,
            flux_dev_path=self.paths.base_model_path,
            siglip_path=self.paths.image_encoder_path,
            calligrapher_weights=self.paths.calligrapher_path,
        ).to_dict()

    def _validate_flux_resources(self) -> None:
        report = inspect_local_resources(
            calligrapher_root=self.paths.calligrapher_root,
            flux_fill_path=self.paths.inpaint_model_path,
            flux_dev_path=self.paths.base_model_path,
            siglip_path=self.paths.image_encoder_path,
            calligrapher_weights=self.paths.calligrapher_path,
        )
        missing = []
        if self.mode == "generation" and not is_complete_flux_model(self.paths.base_model_path):
            missing.append(f"complete FLUX generation model at {self.paths.base_model_path}")
        if self.mode == "inpaint" and not report.flux_fill_complete:
            missing.append(f"complete FLUX-Fill model at {self.paths.inpaint_model_path}")
        if not self.paths.calligrapher_path.is_file():
            missing.append(f"Calligrapher adapter weights at {self.paths.calligrapher_path}")
        if not self.paths.image_encoder_path.exists():
            missing.append(f"SigLIP image encoder at {self.paths.image_encoder_path}")
        if missing:
            raise FileNotFoundError("Missing required resources: " + "; ".join(missing))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FLUX Calligrapher inference.")

    def load(self) -> None:
        if self.backend == "smoke":
            return
        try:
            self._validate_flux_resources()
        except Exception:
            if self.backend == "auto":
                self.backend = "smoke"
                return
            raise

        require_calligrapher_on_path(self.paths.calligrapher_root)
        try:
            import diffusers.utils.import_utils as diffusers_import_utils

            diffusers_import_utils._xformers_available = False
        except Exception:
            pass
        from models.calligrapher import Calligrapher
        from models.transformer_flux_inpainting import FluxTransformer2DModel
        from pipeline_calligrapher import CalligrapherPipeline

        class StyledFluxGenerationPipeline(CalligrapherPipeline):
            @torch.no_grad()
            def __call__(
                self,
                prompt=None,
                prompt_2=None,
                height=None,
                width=None,
                num_inference_steps=50,
                guidance_scale=3.5,
                num_images_per_prompt=1,
                generator=None,
                latents=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                image_emb=None,
                output_type="pil",
                return_dict=True,
                joint_attention_kwargs=None,
                max_sequence_length=512,
            ):
                height = height or self.default_sample_size * self.vae_scale_factor
                width = width or self.default_sample_size * self.vae_scale_factor
                self.check_inputs(
                    prompt,
                    prompt_2,
                    height,
                    width,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    callback_on_step_end_tensor_inputs=["latents"],
                    max_sequence_length=max_sequence_length,
                )
                self._guidance_scale = guidance_scale
                self._joint_attention_kwargs = joint_attention_kwargs
                self._interrupt = False
                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    batch_size = prompt_embeds.shape[0]
                device = self._execution_device
                prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
                num_channels_latents = self.vae.config.latent_channels
                latents, latent_image_ids = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )
                import numpy as _np
                from pipeline_calligrapher import calculate_shift, retrieve_timesteps

                sigmas = _np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                mu = calculate_shift(
                    latents.shape[1],
                    self.scheduler.config.base_image_seq_len,
                    self.scheduler.config.max_image_seq_len,
                    self.scheduler.config.base_shift,
                    self.scheduler.config.max_shift,
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
                    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                    guidance = guidance.expand(latents.shape[0])
                self._num_timesteps = len(timesteps)
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for t in timesteps:
                        if self.interrupt:
                            continue
                        timestep = t.expand(latents.shape[0]).to(latents.dtype)
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            image_emb=image_emb,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        progress_bar.update()
                if output_type == "latent":
                    image = latents
                else:
                    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(latents, return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                self.maybe_free_model_hooks()
                if not return_dict:
                    return (image,)
                from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

                return FluxPipelineOutput(images=image)

        base_path = self.paths.base_model_path if self.mode == "generation" else self.paths.inpaint_model_path
        transformer = FluxTransformer2DModel.from_pretrained(
            str(base_path),
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        pipe_cls = StyledFluxGenerationPipeline if self.mode == "generation" else CalligrapherPipeline
        pipe = pipe_cls.from_pretrained(
            str(base_path),
            transformer=transformer,
            torch_dtype=self.dtype,
        ).to(self.device)
        for module_name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()
        self.model = Calligrapher(
            pipe,
            str(self.paths.image_encoder_path),
            str(self.paths.calligrapher_path),
            device=self.device,
            num_tokens=self.num_tokens,
        )
        transformer_dtype = next(pipe.transformer.parameters()).dtype
        for processor in pipe.transformer.attn_processors.values():
            if hasattr(processor, "to"):
                processor.to(device=self.device, dtype=transformer_dtype)
        self.pipe = pipe
        self.backend = "flux"

    @torch.inference_mode()
    def get_style_tokens(self, reference_patch: Image.Image) -> torch.Tensor:
        if self.model is None and self.backend != "smoke":
            self.load()
        if self.backend == "smoke":
            raise RuntimeError("Smoke backend does not produce diffusion style tokens.")
        ref_image = resize_img_and_pad(reference_patch.convert("RGB"), (512, 512))
        style_tokens = self.model.get_image_embeds(pil_image=ref_image)
        transformer_dtype = next(self.model.pipe.transformer.parameters()).dtype
        style_tokens = style_tokens.to(device=self.device, dtype=transformer_dtype)
        self.last_style_token_shape = tuple(style_tokens.shape)
        print("style injection enabled")
        print(f"style token shape: {tuple(style_tokens.shape)}")
        print(f"loaded style checkpoint: {self.paths.calligrapher_path}")
        return style_tokens

    @torch.inference_mode()
    def generate_text_image(
        self,
        reference_patch: Image.Image,
        target_text: str,
        prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        seed: int = 2025,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        if self.model is None and self.backend != "smoke":
            self.load()
        if self.backend == "smoke":
            raise RuntimeError("Smoke backend is only for tests; generation requires a real diffusion backend.")
        prompt = prompt or build_chinese_prompt(target_text)
        style_tokens = self.get_style_tokens(reference_patch)
        generator = torch.Generator(self.device).manual_seed(seed)
        result = self.model.pipe(
            prompt=prompt,
            image_emb=style_tokens,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        return result

    @torch.inference_mode()
    def generate_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        reference_patch: Image.Image,
        target_text: str,
        prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        seed: int = 2025,
        scale: float = 1.0,
        use_context: bool = True,
    ) -> Image.Image:
        if self.model is None and self.backend != "smoke":
            self.load()
        prompt = prompt or build_chinese_prompt(target_text)
        if self.backend == "smoke":
            return smoke_inpaint(image, mask, reference_patch, output_size=(width, height))

        source, mask_image, ref_image, context_height = prepare_inference_images(
            image=image,
            mask=mask,
            reference=reference_patch,
            width=width,
            height=height,
            use_context=use_context,
        )
        result = self.model.generate(
            image=source,
            mask_image=mask_image,
            ref_image=ref_image,
            prompt=prompt,
            scale=scale,
            num_inference_steps=steps,
            width=source.width,
            height=source.height,
            seed=seed,
        )[0]
        if context_height:
            result = result.crop((0, context_height, result.width, result.height))
        return result


def write_generation_metadata(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
