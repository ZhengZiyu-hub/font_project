from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _prepare_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Convert one tensor image to ``[H, W, C]`` uint8-ready values in ``[0, 1]``."""
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")

    image = image.detach().float().cpu()
    if image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[0]}")

    # Support common model ranges: [-1, 1] and [0, 1].
    if image.min().item() < 0:
        image = (image + 1.0) * 0.5

    image = image.clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image.permute(1, 2, 0)


def save_tensor_image(
    image: torch.Tensor,
    filename: str = "image.png",
    output_dir: str | Path = "outputs",
) -> Path:
    """Save a tensor image under ``output_dir``.

    Accepts ``[C, H, W]`` or ``[B, C, H, W]``; for a batch, the first image is
    saved. Returns the final file path.
    """
    if image.ndim == 4:
        image = image[0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename

    array = (_prepare_image_tensor(image).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array).save(path)
    return path
