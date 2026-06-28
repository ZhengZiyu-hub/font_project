from __future__ import annotations

import os

import torch

from src.models.font_model import FontModel


def main() -> None:
    batch_size = 1
    height = width = 32

    model = FontModel(
        image_channels=3,
        image_encoder_path=os.environ.get("FONT_IMAGE_ENCODER_PATH"),
        condition_dim=32,
        condition_tokens=2,
        condition_heads=2,
        condition_query_tokens=4,
        num_heads=2,
        decoder_blocks=1,
        decoder_single_blocks=1,
        flux_model_path=None,
    )
    model.eval()

    content_image = torch.randn(batch_size, 3, height, width)
    style_image = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        output_image = model(content_image, style_image)

    print(f"flux output shape: {tuple(output_image.shape)}")
    assert output_image.shape == (batch_size, 3, height, width)


if __name__ == "__main__":
    main()
