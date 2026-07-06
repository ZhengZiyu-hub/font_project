from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from fontTools.ttLib import TTCollection, TTFont
except ImportError:  # pragma: no cover
    TTCollection = None
    TTFont = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FONT_EXTS = {".ttf", ".otf", ".ttc"}
CJK_RE = re.compile(r"[\u3400-\u9fff]")
_CMAP_CACHE: dict[Path, set[int] | None] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate Chinese text images on DTD, water, wood, paper, cloth, or custom textures."
    )
    parser.add_argument("--out-dir", default="outputs/chinese_texture")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fonts-dir", action="append", default=[], help="Font folder. Can be used multiple times.")
    parser.add_argument("--text-file", default=str(default_data_root() / "data" / "text_pools" / "chinese_phrases_extended.txt"))
    parser.add_argument("--background-dir", default="", help="Optional folder of texture images to mix in.")
    parser.add_argument("--backgrounds", default="custom,water,wood,paper,cloth")
    parser.add_argument("--custom-background-ratio", type=float, default=0.75)
    parser.add_argument("--standard-font-ratio", type=float, default=0.15)
    parser.add_argument("--max-text-len", type=int, default=8)
    parser.add_argument("--min-font-size", type=int, default=42)
    parser.add_argument("--max-font-size", type=int, default=112)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    font_roots = [Path(item) for item in args.fonts_dir] if args.fonts_dir else default_font_dirs()
    fonts = find_fonts(font_roots)
    stylized_fonts, standard_fonts = split_font_pool(fonts)
    texts = load_texts(Path(args.text_file), args.max_text_len)
    background_root = Path(args.background_dir) if args.background_dir else default_background_dir()
    custom_backgrounds = find_images(background_root) if background_root else []
    bg_kinds = [item.strip().lower() for item in args.backgrounds.split(",") if item.strip()]

    if not fonts:
        raise RuntimeError(f"No .ttf/.otf/.ttc fonts found under: {', '.join(str(path) for path in font_roots)}")
    if not texts:
        raise RuntimeError(f"No usable Chinese text found in: {args.text_file}")
    if "custom" in bg_kinds and not custom_backgrounds:
        bg_kinds = [kind for kind in bg_kinds if kind != "custom"]
    if not bg_kinds:
        bg_kinds = ["water", "wood", "paper", "cloth"]

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    patch_dir = out_dir / "text_patches"
    for directory in (image_dir, mask_dir, patch_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ann_path = out_dir / "annotations.jsonl"
    with ann_path.open("w", encoding="utf-8") as ann:
        for idx in range(args.count):
            record, image, mask, patch = generate_one(
                args,
                stylized_fonts,
                standard_fonts,
                texts,
                custom_backgrounds,
                bg_kinds,
                rng,
                np_rng,
            )
            stem = f"{idx:06d}"
            image_path = image_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            patch_path = patch_dir / f"{stem}.png"
            image.save(image_path)
            mask.save(mask_path)
            patch.save(patch_path)
            record.update(
                {
                    "image": image_path.relative_to(out_dir).as_posix(),
                    "mask": mask_path.relative_to(out_dir).as_posix(),
                    "text_patch": patch_path.relative_to(out_dir).as_posix(),
                }
            )
            ann.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"generated {args.count} images -> {out_dir.resolve()}")
    print(f"annotations -> {ann_path.resolve()}")


def default_data_root() -> Path:
    candidates = [
        Path.cwd().parent / "\u6570\u636e",
        Path("D:/Documents") / "\u6570\u636e",
        Path.home() / "Documents" / "\u6570\u636e",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_font_dirs() -> list[Path]:
    return [Path("assets/fonts/stylized"), default_data_root() / "fonts"]


def default_background_dir() -> Path | None:
    dtd_images = Path("assets/backgrounds/dtd/dtd/images")
    return dtd_images if dtd_images.exists() else None


def find_fonts(roots: list[Path]) -> list[Path]:
    fonts: list[Path] = []
    for root in roots:
        if root.exists():
            fonts.extend(
                path
                for path in root.rglob("*")
                if path.suffix.lower() in FONT_EXTS and not any(part.startswith("_") for part in path.parts)
            )
    return sorted(set(fonts))


def split_font_pool(fonts: list[Path]) -> tuple[list[Path], list[Path]]:
    standard_markers = ("notosans", "notoserif", "sourcehan", "msyh", "simsun", "simhei")
    standard = [path for path in fonts if any(marker in path.name.lower() for marker in standard_markers)]
    stylized = [path for path in fonts if path not in standard]
    return stylized or fonts, standard or fonts


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def load_texts(path: Path, max_len: int) -> list[str]:
    raw = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    texts: list[str] = []
    seen = set()
    for line in raw.splitlines():
        line = re.sub(r"\s+", "", line.strip())
        line = re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", line)
        if not CJK_RE.search(line):
            continue
        if len(line) > max_len:
            line = line[:max_len]
        if line and line not in seen:
            seen.add(line)
            texts.append(line)
    return texts


def generate_one(
    args: argparse.Namespace,
    stylized_fonts: list[Path],
    standard_fonts: list[Path],
    texts: list[str],
    custom_backgrounds: list[Path],
    bg_kinds: list[str],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[dict, Image.Image, Image.Image, Image.Image]:
    bg_kind = choose_background_kind(bg_kinds, bool(custom_backgrounds), args.custom_background_ratio, rng)
    if bg_kind == "custom" and custom_backgrounds:
        bg_path = rng.choice(custom_backgrounds)
        background = crop_texture(bg_path, args.width, args.height, rng)
        bg_source = str(bg_path)
    else:
        background = procedural_background(bg_kind, args.width, args.height, rng, np_rng)
        bg_source = f"procedural:{bg_kind}"

    text = rng.choice(texts)
    if len(text) > 2 and rng.random() < 0.55:
        take = rng.randint(2, min(args.max_text_len, len(text)))
        start = rng.randint(0, max(0, len(text) - take))
        text = text[start : start + take]

    font_path, font_group = choose_font_for_text(stylized_fonts, standard_fonts, text, args.standard_font_ratio, rng)
    patch, text_mask, text_bbox, font_size = fit_text_patch(
        text=text,
        font_path=font_path,
        canvas_size=(args.width, args.height),
        min_size=args.min_font_size,
        max_size=args.max_font_size,
        rng=rng,
    )

    angle = rng.uniform(-8.0, 8.0)
    patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    text_mask = text_mask.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    alpha_bbox = text_mask.getbbox()
    if alpha_bbox:
        patch = patch.crop(alpha_bbox)
        text_mask = text_mask.crop(alpha_bbox)

    x = max(0, (args.width - patch.width) // 2 + rng.randint(-18, 18))
    y = max(0, (args.height - patch.height) // 2 + rng.randint(-10, 10))
    x = min(x, max(0, args.width - patch.width))
    y = min(y, max(0, args.height - patch.height))

    composed = background.convert("RGBA")
    composed.alpha_composite(patch, (x, y))

    mask = Image.new("L", (args.width, args.height), 0)
    mask.paste(text_mask, (x, y), text_mask)
    if rng.random() < 0.35:
        composed = add_sensor_noise(composed, np_rng, strength=rng.uniform(2.0, 7.0))

    bbox = mask.getbbox() or (x, y, x + patch.width, y + patch.height)
    record = {
        "text": text,
        "font": str(font_path),
        "font_group": font_group,
        "font_size": font_size,
        "background": bg_source,
        "width": args.width,
        "height": args.height,
        "bbox": list(bbox),
        "angle": round(angle, 3),
        "text_bbox_before_rotation": list(text_bbox),
    }
    return record, composed.convert("RGB"), mask, patch


def choose_background_kind(bg_kinds: list[str], has_custom: bool, custom_ratio: float, rng: random.Random) -> str:
    if has_custom and "custom" in bg_kinds and rng.random() < max(0.0, min(1.0, custom_ratio)):
        return "custom"
    choices = [kind for kind in bg_kinds if kind != "custom"]
    if not choices:
        choices = ["custom"] if has_custom else ["water", "wood", "paper", "cloth"]
    return rng.choice(choices)


def choose_font_for_text(
    stylized_fonts: list[Path],
    standard_fonts: list[Path],
    text: str,
    standard_ratio: float,
    rng: random.Random,
) -> tuple[Path, str]:
    use_standard = rng.random() < max(0.0, min(1.0, standard_ratio))
    primary = standard_fonts if use_standard else stylized_fonts
    fallback = stylized_fonts if use_standard else standard_fonts
    groups = (("standard" if use_standard else "stylized", primary), ("stylized" if use_standard else "standard", fallback))
    for group_name, pool in groups:
        candidates = pool[:]
        rng.shuffle(candidates)
        for font_path in candidates:
            if font_supports_text(font_path, text) and font_can_render(font_path, text):
                return font_path, group_name
    return rng.choice(primary or fallback), "unchecked"


def font_supports_text(font_path: Path, text: str) -> bool:
    cmap = font_cmap(font_path)
    if cmap is None:
        return True
    return all(ord(ch) in cmap for ch in text if ch.strip())


def font_can_render(font_path: Path, text: str) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), size=48)
        ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
        return True
    except Exception:
        return False


def font_cmap(font_path: Path) -> set[int] | None:
    font_path = font_path.resolve()
    if font_path in _CMAP_CACHE:
        return _CMAP_CACHE[font_path]
    if TTFont is None:
        _CMAP_CACHE[font_path] = None
        return None
    try:
        cmaps: set[int] = set()
        if font_path.suffix.lower() == ".ttc":
            collection = TTCollection(str(font_path))
            fonts = collection.fonts
        else:
            fonts = [TTFont(str(font_path), lazy=True)]
        for font in fonts:
            for table in font["cmap"].tables:
                cmaps.update(table.cmap.keys())
            font.close()
        _CMAP_CACHE[font_path] = cmaps
        return cmaps
    except Exception:
        _CMAP_CACHE[font_path] = None
        return None


def fit_text_patch(
    text: str,
    font_path: Path,
    canvas_size: tuple[int, int],
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], int]:
    width, height = canvas_size
    target_w = int(width * rng.uniform(0.70, 0.94))
    target_h = int(height * rng.uniform(0.48, 0.80))
    chosen_font = None
    chosen_bbox = (0, 0, 1, 1)
    chosen_size = min_size
    for size in range(max_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font, stroke_width=max(1, size // 28))
        if bbox[2] - bbox[0] <= target_w and bbox[3] - bbox[1] <= target_h:
            chosen_font = font
            chosen_bbox = bbox
            chosen_size = size
            break
    if chosen_font is None:
        chosen_font = ImageFont.truetype(str(font_path), size=min_size)
        chosen_bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=chosen_font)
        chosen_size = min_size

    pad = max(16, chosen_size // 3)
    tw = chosen_bbox[2] - chosen_bbox[0]
    th = chosen_bbox[3] - chosen_bbox[1]
    patch = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    mask = Image.new("L", patch.size, 0)
    draw = ImageDraw.Draw(patch)
    mask_draw = ImageDraw.Draw(mask)

    fill = rng.choice([(178, 69, 30), (35, 44, 48), (232, 226, 205), (28, 91, 133), (206, 42, 45)])
    stroke_fill = rng.choice([(20, 20, 18), (245, 242, 218), (94, 48, 23)])
    stroke_width = rng.randint(0, max(1, chosen_size // 18))
    shadow_offset = (rng.randint(1, 5), rng.randint(1, 5))
    shadow = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (pad - chosen_bbox[0] + shadow_offset[0], pad - chosen_bbox[1] + shadow_offset[1]),
        text,
        font=chosen_font,
        fill=(0, 0, 0, rng.randint(45, 105)),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, rng.randint(40, 90)),
    )
    patch.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.2))))

    pos = (pad - chosen_bbox[0], pad - chosen_bbox[1])
    draw.text(pos, text, font=chosen_font, fill=(*fill, rng.randint(215, 255)), stroke_width=stroke_width, stroke_fill=stroke_fill)
    mask_draw.text(pos, text, font=chosen_font, fill=255, stroke_width=stroke_width, stroke_fill=255)

    if rng.random() < 0.45:
        patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    return patch, mask, chosen_bbox, chosen_size


def crop_texture(path: Path, width: int, height: int, rng: random.Random) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    new_size = (max(width, int(image.width * scale)), max(height, int(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    x = rng.randint(0, image.width - width)
    y = rng.randint(0, image.height - height)
    return image.crop((x, y, x + width, y + height))


def procedural_background(kind: str, width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    if kind == "wood":
        return wood_texture(width, height, rng, np_rng)
    if kind == "paper":
        return paper_texture(width, height, rng, np_rng)
    if kind == "cloth":
        return cloth_texture(width, height, rng, np_rng)
    return water_texture(width, height, rng, np_rng)


def water_texture(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    waves = (
        np.sin(x / rng.uniform(8, 18) + y / rng.uniform(18, 35))
        + np.sin((x + y) / rng.uniform(14, 28))
        + np.sin(y / rng.uniform(5, 11))
    )
    noise = np_rng.normal(0, 0.25, (height, width))
    v = normalize(waves + noise)
    rgb = np.dstack((25 + v * 45, 82 + v * 90, 105 + v * 110)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(0.35))


def wood_texture(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    grain = np.sin((x + 9 * np.sin(y / 23.0)) / rng.uniform(7, 16)) + np_rng.normal(0, 0.2, (height, width))
    v = normalize(grain)
    rgb = np.dstack((92 + v * 95, 54 + v * 55, 25 + v * 30)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(0.45))


def paper_texture(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    base = np_rng.normal(0, 1, (height, width))
    y, x = np.mgrid[0:height, 0:width]
    folds = 0.9 * np.sin((x + y) / rng.uniform(35, 70)) + 0.6 * np.sin((x - y) / rng.uniform(45, 90))
    v = normalize(base * 0.45 + folds)
    tint = rng.choice([(214, 205, 184), (232, 225, 205), (198, 206, 196)])
    rgb = np.dstack([np.clip(channel - 30 + v * 58, 0, 255) for channel in tint]).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(0.25))


def cloth_texture(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    weave = np.sin(x / 2.6) * 0.35 + np.sin(y / 3.1) * 0.35 + np_rng.normal(0, 0.18, (height, width))
    folds = np.sin((x + y) / rng.uniform(24, 50))
    v = normalize(weave + folds)
    rgb = np.dstack((54 + v * 70, 68 + v * 65, 73 + v * 58)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(0.25))


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    low, high = np.percentile(arr, [1, 99])
    return np.clip((arr - low) / max(1e-6, high - low), 0, 1)


def add_sensor_noise(image: Image.Image, np_rng: np.random.Generator, strength: float) -> Image.Image:
    arr = np.asarray(image.convert("RGBA")).astype("float32")
    arr[:, :, :3] += np_rng.normal(0, strength, arr[:, :, :3].shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


if __name__ == "__main__":
    main()
