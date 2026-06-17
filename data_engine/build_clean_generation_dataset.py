from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.calligrapher_pipeline import build_chinese_prompt


def first_existing(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def resolve_path(data_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def mask_bbox(mask: Image.Image, fallback_bbox: list[int] | None = None) -> tuple[int, int, int, int] | None:
    mask_np = np.array(mask.convert("L")) > 127
    ys, xs = np.where(mask_np)
    if len(xs) and len(ys):
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if fallback_bbox and len(fallback_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in fallback_bbox]
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return None


def make_clean_target(
    image: Image.Image,
    mask: Image.Image,
    bbox: tuple[int, int, int, int],
    size: int,
    background: str,
    padding: int,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    image_crop = image.convert("RGBA").crop((x1, y1, x2, y2))
    mask_crop = mask.convert("L").crop((x1, y1, x2, y2)).point(lambda p: 255 if p > 127 else 0)

    if background == "transparent":
        clean = Image.new("RGBA", image_crop.size, (255, 255, 255, 0))
        clean.paste(image_crop, (0, 0), mask_crop)
    else:
        clean = Image.new("RGBA", image_crop.size, (255, 255, 255, 255))
        clean.paste(image_crop, (0, 0), mask_crop)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0 if background == "transparent" else 255))
    scale = min(size / clean.width, size / clean.height)
    new_size = (max(1, int(clean.width * scale)), max(1, int(clean.height * scale)))
    resized = clean.resize(new_size, Image.BILINEAR)
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2), resized)
    return canvas if background == "transparent" else canvas.convert("RGB")


def load_plain_font(font_path: str | Path | None, font_size: int) -> ImageFont.FreeTypeFont:
    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def render_plain_text_reference(
    text: str,
    size: tuple[int, int],
    font_path: str | Path | None = None,
    font_size: int | None = None,
) -> Image.Image:
    width, height = size
    font_size = font_size or max(24, int(height * 0.55))
    font = load_plain_font(font_path, font_size)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # Shrink until the text fits comfortably.
    for candidate_size in range(font_size, 11, -2):
        font = load_plain_font(font_path, candidate_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= width * 0.9 and text_h <= height * 0.8:
            break
    x = (width - text_w) // 2 - bbox[0]
    y = (height - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill="black", font=font)
    return canvas


def build_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    reference_dir = output_root / "references"
    image_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    output_ann = output_root / "annotations.jsonl"

    written = 0
    skipped = 0
    with Path(args.annotation_file).open("r", encoding="utf-8") as src, output_ann.open(
        "w", encoding="utf-8"
    ) as dst:
        for index, line in enumerate(src):
            if args.max_samples is not None and written >= args.max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            image_key = first_existing(record, ("target_image", "image", "image_path"))
            mask_key = first_existing(record, ("mask", "mask_path", "mask_image"))
            reference_key = first_existing(
                record,
                ("reference", "reference_patch", "ref_image", "text_patch", "style_patch"),
            )
            target_text = first_existing(record, ("target_text", "text", "caption"))
            if image_key is None or target_text is None:
                skipped += 1
                continue

            image_path = resolve_path(data_root, image_key)
            mask_path = resolve_path(data_root, mask_key) if mask_key is not None else None
            reference_path = resolve_path(data_root, reference_key) if reference_key is not None else None
            if not image_path.is_file():
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            sample_id = f"{index:06d}"
            image_ext = ".png"
            target_rel = Path("images") / f"{sample_id}{image_ext}"
            ref_rel = Path("references") / f"{sample_id}{image_ext}"

            if args.target_mode == "original":
                shutil.copy2(image_path, output_root / target_rel)
                bbox = record.get("bbox")
            else:
                if mask_path is None or not mask_path.is_file():
                    skipped += 1
                    continue
                mask = Image.open(mask_path).convert("L")
                bbox = mask_bbox(mask, record.get("bbox"))
                if bbox is None:
                    skipped += 1
                    continue
                clean = make_clean_target(
                    image=image,
                    mask=mask,
                    bbox=bbox,
                    size=args.size,
                    background=args.background,
                    padding=args.padding,
                )
                clean.save(output_root / target_rel)

            if args.reference_mode == "plain-text":
                ref_size = (
                    args.reference_width or image.width,
                    args.reference_height or image.height,
                )
                plain_reference = render_plain_text_reference(
                    str(target_text),
                    ref_size,
                    font_path=args.plain_font,
                    font_size=args.plain_font_size,
                )
                plain_reference.save(output_root / ref_rel)
            else:
                if reference_path is None or not reference_path.is_file():
                    skipped += 1
                    continue
                shutil.copy2(reference_path, output_root / ref_rel)

            out_record = {
                "target_image": str(target_rel),
                "reference": str(ref_rel),
                "target_text": str(target_text),
                "prompt": first_existing(record, ("prompt", "instruction"))
                or build_chinese_prompt(str(target_text), args.prompt_template),
                "source_image": str(image_key),
                "source_mask": str(mask_key) if mask_key is not None else None,
                "source_reference": str(reference_key),
                "bbox": list(bbox) if bbox is not None else None,
                "target_mode": args.target_mode,
                "reference_mode": args.reference_mode,
            }
            dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} samples to {output_root}; skipped {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference/text generation dataset.")
    parser.add_argument("--annotation-file", default=PROJECT_ROOT / "datasets/dataset_3000/annotations.jsonl")
    parser.add_argument("--data-root", default=PROJECT_ROOT / "datasets/dataset_3000")
    parser.add_argument("--output-root", default=PROJECT_ROOT / "datasets/generation_dataset")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--target-mode", choices=["original", "clean"], default="original")
    parser.add_argument("--reference-mode", choices=["plain-text", "text-patch"], default="plain-text")
    parser.add_argument("--reference-width", type=int, default=None)
    parser.add_argument("--reference-height", type=int, default=None)
    parser.add_argument("--plain-font", default=None)
    parser.add_argument("--plain-font-size", type=int, default=None)
    parser.add_argument("--background", choices=["white", "transparent"], default="white")
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--prompt-template", default='The Chinese text is "{text}".')
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
