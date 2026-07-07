from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.models.experiment_modes import add_experiment_mode_arg, apply_experiment_mode_to_config
from src.pipelines.flux_infer import FluxCustomInferenceRunner, load_style_image
from src.utils.config import load_config


def _default_style_image() -> bytes:
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.text((116, 126), "风格", fill="black")
    array = np.asarray(image)
    output = Image.fromarray(array)
    import io

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real branch inference.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--style_image", default=None)
    parser.add_argument("--text", default="生日快乐")
    parser.add_argument("--output", default="outputs/inference_mode_a.png")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default=None)
    add_experiment_mode_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_experiment_mode_to_config(config, args.mode)

    if args.style_image:
        style_bytes = Path(args.style_image).read_bytes()
    else:
        style_bytes = _default_style_image()

    style_tensor = load_style_image(style_bytes)
    runner = FluxCustomInferenceRunner(config_path=args.config, device=args.device)
    # Force CLI mode override even when the YAML default is baseline.
    runner.config = config
    result = runner.infer(
        branch=args.mode,
        style_image=style_tensor,
        text_prompt=args.text,
        num_inference_steps=args.steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path)
    print(f"saved={output_path}")
    print(f"glyph_tokens={result.metadata.get('glyph_tokens')}")


if __name__ == "__main__":
    main()
