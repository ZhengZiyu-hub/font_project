from __future__ import annotations

import torch

from src.models.font_model import FontModel


def main() -> None:
    batch_size = 2
    height = width = 64

    model = FontModel(image_channels=3, encoder_dim=64, hidden_dim=64, style_tokens=4)
    model.eval()

    content_image = torch.randn(batch_size, 3, height, width)
    style_image = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        output_image = model(content_image, style_image)

    print(f"output shape: {tuple(output_image.shape)}")
    assert output_image.shape == (batch_size, 3, height, width)


if __name__ == "__main__":
    main()
