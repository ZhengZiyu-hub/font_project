from __future__ import annotations

import torch

from src.models.font_model import FontModel


def main() -> None:
    batch_size = 1

    model = FontModel(use_glyph_prior=False, use_retrieval_prior=False)
    model.eval()

    text_tokens, pooled, txt_ids = model.encode_text(["baseline glyph"])
    latents, img_ids = model.prepare_latents(batch_size)
    style_tokens = torch.empty(batch_size, 0, text_tokens.shape[-1], device=text_tokens.device, dtype=text_tokens.dtype)
    timestep = torch.zeros(batch_size, device=text_tokens.device, dtype=text_tokens.dtype)
    ids = {
        "txt_ids": txt_ids,
        "img_ids": img_ids,
        "pooled_projections": pooled,
    }

    with torch.no_grad():
        output = model.forward_baseline(latents, text_tokens, style_tokens, timestep, ids)

    print(f"flux output shape: {tuple(output.shape)}")
    assert output.shape == latents.shape


if __name__ == "__main__":
    main()
