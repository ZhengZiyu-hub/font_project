from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _resolve_font_path(font_id: str, font_root: str | Path | None = None) -> Path | None:
    direct_path = Path(font_id)
    if direct_path.is_file():
        return direct_path

    if font_root is None:
        return None

    root = Path(font_root)
    candidate = root / font_id
    if candidate.is_file():
        return candidate

    for suffix in (".ttf", ".otf", ".ttc"):
        candidate = root / f"{font_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def render_text_with_font(
    text: str,
    font_id: str,
    font_root: str | Path | None = None,
    image_size: int = 64,
    font_size: int | None = None,
) -> torch.Tensor:
    """Render text with a retrieved font.

    Args:
        text: glyph text to render.
        font_id: font path or id inside ``font_root``.
        font_root: optional font directory.
        image_size: square canvas size.
        font_size: optional font size. Defaults to roughly 70 percent of the
            canvas height.

    Returns:
        rendered glyph tensor with shape ``[3, image_size, image_size]`` in
        ``[-1, 1]``. Black glyphs are rendered on a white background.
    """

    font_size = font_size or max(8, int(image_size * 0.7))
    font_path = _resolve_font_path(font_id, font_root)
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(str(font_path), size=font_size)

    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size - text_width) / 2 - bbox[0]
    y = (image_size - text_height) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0
