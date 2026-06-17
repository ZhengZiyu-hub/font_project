from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.calligrapher_pipeline import (
    CalligrapherGenerationPipeline,
    CalligrapherPaths,
    build_chinese_prompt,
    resize_img_and_pad,
)


def first_existing(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


class GenerationJsonlDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        data_root: str | Path,
        prompt_template: str,
        max_samples: int | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.data_root = Path(data_root)
        self.prompt_template = prompt_template
        self.records: list[dict[str, Any]] = []
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.records.append(json.loads(line))
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        if not self.records:
            raise ValueError(f"No records found in {self.annotation_file}")

    def __len__(self) -> int:
        return len(self.records)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        reference_key = first_existing(
            record,
            ("reference", "reference_patch", "ref_image", "text_patch", "style_patch"),
        )
        target_key = first_existing(record, ("target_image", "image", "image_path"))
        target_text = first_existing(record, ("target_text", "text", "caption"))
        prompt = first_existing(record, ("prompt", "instruction"))
        if reference_key is None or target_key is None or target_text is None:
            raise KeyError("Each sample needs reference, target_text/text, and target_image/image.")
        return {
            "reference": Image.open(self.resolve_path(reference_key)).convert("RGB"),
            "target_image": Image.open(self.resolve_path(target_key)).convert("RGB"),
            "target_text": str(target_text),
            "prompt": str(prompt or build_chinese_prompt(str(target_text), self.prompt_template)),
            "record": record,
        }


def collate_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen


def set_trainable_modules(model) -> list[torch.nn.Parameter]:
    pipe = model.pipe
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, model.image_encoder):
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    train_modules = [model.image_proj_mlp, model.image_proj_qformer]
    train_modules.extend(list(pipe.transformer.attn_processors.values()))
    params: list[torch.nn.Parameter] = []
    for module in train_modules:
        module.train()
        module.requires_grad_(True)
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def get_trainable_style_tokens(model, references: list[Image.Image]) -> torch.Tensor:
    references = [resize_img_and_pad(ref.convert("RGB"), (512, 512)) for ref in references]
    clip_image = model.clip_image_processor(images=references, return_tensors="pt").pixel_values
    clip_image = clip_image.to(model.device, dtype=model.image_encoder.dtype)
    with torch.no_grad():
        clip_image_embeds = model.image_encoder(clip_image).pooler_output
    clip_image_embeds = clip_image_embeds.to(dtype=model.image_proj_mlp.proj[0].weight.dtype)
    return model.image_proj_mlp(clip_image_embeds) + model.image_proj_qformer(clip_image_embeds)


def encode_targets_to_packed_latents(pipe, images, width: int, height: int, dtype, device, generator):
    image_tensor = pipe.image_processor.preprocess(images, height=height, width=width)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latent_h, latent_w = latents.shape[-2:]
    return pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latent_h, latent_w)


def prepare_wrapper(args: argparse.Namespace) -> CalligrapherGenerationPipeline:
    paths = CalligrapherPaths(
        base_model_path=Path(args.base_model_path),
        inpaint_model_path=Path(args.inpaint_model_path),
        image_encoder_path=Path(args.image_encoder_path),
        calligrapher_path=Path(args.calligrapher_weights),
        calligrapher_root=Path(args.calligrapher_root),
    )
    wrapper = CalligrapherGenerationPipeline(
        paths=paths,
        device=args.device,
        backend="flux",
        mode="generation",
    )
    wrapper.load()
    return wrapper


def run_dry_run(args: argparse.Namespace, wrapper: CalligrapherGenerationPipeline, loader: DataLoader) -> None:
    model = wrapper.model
    params = set_trainable_modules(model)
    batch = next(iter(loader))
    style_tokens = get_trainable_style_tokens(model, batch["reference"])
    target_tensor = model.pipe.image_processor.preprocess(
        batch["target_image"], height=args.height, width=args.width
    )
    trainable, frozen = count_parameters(model.pipe.transformer)
    mlp_trainable, _ = count_parameters(model.image_proj_mlp)
    qformer_trainable, _ = count_parameters(model.image_proj_qformer)
    print(f"loaded base model: {args.base_model_path}")
    print(f"loaded style checkpoint: {args.calligrapher_weights}")
    print("style injection enabled")
    print(f"trainable adapter parameters: {sum(p.numel() for p in params)}")
    print(f"trainable transformer-attn parameters: {trainable}")
    print(f"frozen transformer parameters: {frozen}")
    print(f"trainable mlp parameters: {mlp_trainable}")
    print(f"trainable qformer parameters: {qformer_trainable}")
    print(f"one batch target tensor shape: {tuple(target_tensor.shape)}")
    print(f"one batch prompts: {batch['prompt'][:2]}")
    print(f"style token shape: {tuple(style_tokens.shape)}")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    dataset = GenerationJsonlDataset(
        annotation_file=args.annotation_file,
        data_root=args.data_root,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.dry_run,
        num_workers=args.num_workers,
        collate_fn=collate_records,
    )
    wrapper = prepare_wrapper(args)
    model = wrapper.model
    pipe = model.pipe
    params = set_trainable_modules(model)
    if args.dry_run:
        run_dry_run(args, wrapper, loader)
        return

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    device = torch.device(args.device)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            global_step += 1
            generator = torch.Generator(device=device).manual_seed(args.seed + global_step)
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=batch["prompt"],
                prompt_2=batch["prompt"],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
            target_latents = encode_targets_to_packed_latents(
                pipe,
                batch["target_image"],
                width=args.width,
                height=args.height,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
            )
            noise = torch.randn_like(target_latents)
            sigmas = torch.rand(target_latents.shape[0], 1, 1, device=device, dtype=target_latents.dtype)
            noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise
            timesteps = sigmas.flatten()
            style_tokens = get_trainable_style_tokens(model, batch["reference"])
            guidance = None
            if pipe.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [target_latents.shape[0]],
                    args.guidance_scale,
                    device=device,
                    dtype=torch.float32,
                )
            latent_image_ids = pipe._prepare_latent_image_ids(
                target_latents.shape[0],
                args.height // (pipe.vae_scale_factor * 2),
                args.width // (pipe.vae_scale_factor * 2),
                device,
                prompt_embeds.dtype,
            )
            pred = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                image_emb=style_tokens,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            target = noise - target_latents
            loss = F.mse_loss(pred.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                record = {"epoch": epoch, "step": global_step, "loss": float(loss.detach().cpu())}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"epoch={epoch} step={global_step} loss={record['loss']:.6f}")
            if global_step % args.save_every == 0:
                save_adapter_checkpoint(output_dir / f"adapter_step_{global_step}.pt", model, optimizer, args)
    save_adapter_checkpoint(output_dir / "adapter_final.pt", model, optimizer, args)


def save_adapter_checkpoint(path: Path, model, optimizer, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_proj_mlp": model.image_proj_mlp.state_dict(),
            "image_proj_qformer": model.image_proj_qformer.state_dict(),
            "attn_adapter": torch.nn.ModuleList(model.pipe.transformer.attn_processors.values()).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean Chinese style text generation adapters.")
    parser.add_argument("--mode", choices=["generation"], default="generation")
    parser.add_argument("--annotation-file", default=PROJECT_ROOT / "datasets/generation_dataset/annotations.jsonl")
    parser.add_argument("--data-root", default=PROJECT_ROOT / "datasets/generation_dataset")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "outputs/calligrapher_generation")
    parser.add_argument("--base-model-path", default="/data/zhengziyu/models/FLUX.1-dev")
    parser.add_argument("--inpaint-model-path", default="/data/zhengziyu/Calligrapher/pretrained/FLUX.1-Fill-dev")
    parser.add_argument("--image-encoder-path", default="/data/zhengziyu/Calligrapher/pretrained/siglip-so400m-patch14-384")
    parser.add_argument("--calligrapher-weights", default="/data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin")
    parser.add_argument("--calligrapher-root", default="/data/zhengziyu/Calligrapher")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--prompt-template", default='The Chinese text is "{text}".')
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-bg-only", action="store_true", help="Dataset should already contain clean targets.")
    parser.add_argument("--auto-clean-bg", action="store_true", help="Use data_engine/build_clean_generation_dataset.py before training.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
