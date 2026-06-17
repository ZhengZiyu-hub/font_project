from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.calligrapher_pipeline import (
    CalligrapherGenerationPipeline,
    CalligrapherPaths,
    build_chinese_prompt,
    load_image,
    write_generation_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chinese Calligrapher-style text generation.")
    parser.add_argument("--mode", choices=["generation", "inpaint"], default="generation")
    parser.add_argument("--reference", required=True, help="Style reference patch.")
    parser.add_argument("--text", required=True, help="Target Chinese text, e.g. 生日快乐.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--image", default=None, help="Experimental inpaint mode source image.")
    parser.add_argument("--mask", default=None, help="Experimental inpaint mode mask image.")
    parser.add_argument("--prompt", default=None, help="Optional full prompt.")
    parser.add_argument("--prompt-template", default='The Chinese text is "{text}".')
    parser.add_argument("--backend", choices=["auto", "flux", "smoke"], default="flux")
    parser.add_argument(
        "--base-model-path",
        default="/data/zhengziyu/models/FLUX.1-dev",
        help="Local FLUX.1-dev directory for clean generation.",
    )
    parser.add_argument(
        "--inpaint-model-path",
        default="/data/zhengziyu/Calligrapher/pretrained/FLUX.1-Fill-dev",
        help="Optional FLUX-Fill directory for experimental inpainting.",
    )
    parser.add_argument(
        "--image-encoder-path",
        default="/data/zhengziyu/Calligrapher/pretrained/siglip-so400m-patch14-384",
        help="Local SigLIP image encoder directory.",
    )
    parser.add_argument(
        "--calligrapher-weights",
        default="/data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin",
        help="Calligrapher QFormer/MLP/attention adapter checkpoint.",
    )
    parser.add_argument("--calligrapher-root", default="/data/zhengziyu/Calligrapher")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--scale", type=float, default=1.0, help="Style scale for experimental inpaint mode.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--no-context", action="store_true", help="Experimental inpaint mode only.")
    parser.add_argument("--no-remove-background", action="store_true")
    parser.add_argument("--background-threshold", type=int, default=245)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def remove_near_white_background(image: Image.Image, threshold: int = 245) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = pixels[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                pixels[x, y] = (r, g, b, 0)
            else:
                pixels[x, y] = (r, g, b, a)
    return rgba


def main() -> None:
    args = parse_args()
    if args.mode == "generation" and args.backend == "smoke":
        raise ValueError("Smoke backend is only for unit tests; generation requires real diffusion output.")
    if args.mode == "inpaint" and (not args.image or not args.mask):
        raise ValueError("--image and --mask are required only when --mode inpaint.")

    paths = CalligrapherPaths(
        base_model_path=Path(args.base_model_path),
        inpaint_model_path=Path(args.inpaint_model_path),
        image_encoder_path=Path(args.image_encoder_path),
        calligrapher_path=Path(args.calligrapher_weights),
        calligrapher_root=Path(args.calligrapher_root),
    )
    pipeline = CalligrapherGenerationPipeline(
        paths=paths,
        device=args.device,
        dtype=dtype_from_name(args.dtype),
        backend=args.backend,
        mode=args.mode,
    )
    reference = load_image(args.reference, "RGB")
    prompt = args.prompt or build_chinese_prompt(args.text, args.prompt_template)

    if args.mode == "generation":
        result = pipeline.generate_text_image(
            reference_patch=reference,
            target_text=args.text,
            prompt=prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
    else:
        image = load_image(args.image, "RGB")
        mask = load_image(args.mask, "L")
        result = pipeline.generate_inpaint(
            image=image,
            mask=mask,
            reference_patch=reference,
            target_text=args.text,
            prompt=prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            scale=args.scale,
            use_context=not args.no_context,
        )

    if args.mode == "generation" and not args.no_remove_background:
        result = remove_near_white_background(result, threshold=args.background_threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    write_generation_metadata(
        output_path.with_suffix(output_path.suffix + ".json"),
        {
            "mode": args.mode,
            "backend": pipeline.backend,
            "style_injection": "enabled" if pipeline.backend == "flux" else "disabled",
            "style_token_shape": pipeline.last_style_token_shape,
            "prompt": prompt,
            "target_text": args.text,
            "reference": args.reference,
            "image": args.image,
            "mask": args.mask,
            "output": str(output_path),
            "resources": pipeline.resource_report(),
        },
    )
    print(f"saved {output_path} using mode={args.mode} backend={pipeline.backend}")


if __name__ == "__main__":
    main()
