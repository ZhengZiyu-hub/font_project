from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FluxDecoderConfig:
    """Configuration for the optional FLUX transformer decoder backend.

    The adapter keeps the project interface unchanged:
    latent image + content tokens + style tokens -> RGB image.
    """

    image_channels: int = 3
    hidden_dim: int = 128
    patch_size: int = 2
    num_layers: int = 1
    num_single_layers: int = 1
    num_attention_heads: int = 4
    attention_head_dim: int = 32
    pooled_projection_dim: int = 128
    pretrained_model_name_or_path: str | None = None
    subfolder: str | None = "transformer"
    guidance_scale: float = 1.0


def _default_rope_axes(head_dim: int) -> tuple[int, int, int]:
    """Split one attention head into 3 rotary axes used by FLUX ids."""
    first = max(2, head_dim // 4)
    second = max(2, (head_dim - first) // 2)
    third = head_dim - first - second
    if third <= 0:
        raise ValueError(f"attention_head_dim is too small for 3-axis rotary embedding: {head_dim}")
    return first, second, third


class FluxImageDecoder(nn.Module):
    """Project-native adapter around Diffusers' FLUX transformer.

    Inputs:
        latent: ``[B, C, H, W]`` image-like latent canvas.
        content_tokens: ``[B, N_content, D]`` structural tokens.
        style_tokens: ``[B, N_style, D]`` style tokens.

    Output:
        image: ``[B, 3, H, W]``.

    This module does not make the reference repository a runtime dependency.
    It uses Diffusers directly. If no pretrained path is provided, it creates a
    small randomly initialized FLUX transformer so imports and dummy forward can
    be tested without downloading weights.
    """

    def __init__(self, config: FluxDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_dim = config.image_channels * config.patch_size * config.patch_size

        try:
            from diffusers import FluxTransformer2DModel
        except ImportError as exc:
            raise ImportError("FluxImageDecoder requires diffusers. Install project requirements first.") from exc

        if config.pretrained_model_name_or_path:
            self.transformer = FluxTransformer2DModel.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder=config.subfolder,
            )
        else:
            self.transformer = FluxTransformer2DModel(
                patch_size=1,
                in_channels=self.patch_dim,
                out_channels=self.patch_dim,
                num_layers=config.num_layers,
                num_single_layers=config.num_single_layers,
                attention_head_dim=config.attention_head_dim,
                num_attention_heads=config.num_attention_heads,
                joint_attention_dim=config.hidden_dim,
                pooled_projection_dim=config.pooled_projection_dim,
                axes_dims_rope=_default_rope_axes(config.attention_head_dim),
            )
        transformer_config = self.transformer.config
        flux_in_channels = int(transformer_config.in_channels)
        flux_out_channels = int(getattr(transformer_config, "out_channels", flux_in_channels) or flux_in_channels)
        flux_condition_dim = int(transformer_config.joint_attention_dim)
        flux_pooled_dim = int(transformer_config.pooled_projection_dim)

        # Local project tokens are intentionally small. These adapters bridge
        # them to the dimensions expected by a real FLUX checkpoint.
        self.to_flux_channels = (
            nn.Identity() if self.patch_dim == flux_in_channels else nn.Linear(self.patch_dim, flux_in_channels)
        )
        self.from_flux_channels = (
            nn.Identity() if flux_out_channels == self.patch_dim else nn.Linear(flux_out_channels, self.patch_dim)
        )
        self.to_flux_condition = (
            nn.Identity() if config.hidden_dim == flux_condition_dim else nn.Linear(config.hidden_dim, flux_condition_dim)
        )
        self.pooled_projection_dim = flux_pooled_dim
        self.uses_guidance = bool(getattr(transformer_config, "guidance_embeds", False))

    def _patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Convert ``[B, C, H, W]`` to patch tokens ``[B, h*w, C*p*p]``."""
        batch_size, channels, height, width = image.shape
        patch = self.config.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Image size {(height, width)} must be divisible by patch_size={patch}")

        grid_h = height // patch
        grid_w = width // patch
        tokens = image.view(batch_size, channels, grid_h, patch, grid_w, patch)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, grid_h * grid_w, -1)
        return tokens, grid_h, grid_w

    def _unpatchify(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Convert patch tokens ``[B, h*w, C*p*p]`` back to ``[B, C, H, W]``."""
        batch_size = tokens.shape[0]
        patch = self.config.patch_size
        channels = self.config.image_channels
        image = tokens.view(batch_size, grid_h, grid_w, channels, patch, patch)
        image = image.permute(0, 3, 1, 4, 2, 5).contiguous()
        return image.view(batch_size, channels, grid_h * patch, grid_w * patch)

    def _image_ids(self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build FLUX 3-axis image ids with shape ``[h*w, 3]``."""
        y, x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        zeros = torch.zeros_like(x)
        return torch.stack([zeros, y, x], dim=-1).reshape(grid_h * grid_w, 3)

    def _condition_ids(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build neutral FLUX ids for non-image condition tokens."""
        return torch.zeros(length, 3, device=device, dtype=dtype)

    def forward(
        self,
        latent: torch.Tensor,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patch_tokens, grid_h, grid_w = self._patchify(latent)
        patch_tokens = self.to_flux_channels(patch_tokens)
        condition_tokens = self.to_flux_condition(torch.cat([content_tokens, style_tokens], dim=1))

        batch_size = latent.shape[0]
        device = latent.device
        dtype = patch_tokens.dtype
        if timestep is None:
            timestep = torch.zeros(batch_size, device=device, dtype=dtype)

        pooled = torch.zeros(
            batch_size,
            self.pooled_projection_dim,
            device=device,
            dtype=dtype,
        )

        kwargs = {
            "hidden_states": patch_tokens,
            "encoder_hidden_states": condition_tokens,
            "pooled_projections": pooled,
            "timestep": timestep,
            "img_ids": self._image_ids(grid_h, grid_w, device, dtype),
            "txt_ids": self._condition_ids(condition_tokens.shape[1], device, dtype),
            "return_dict": True,
        }
        if self.uses_guidance:
            kwargs["guidance"] = torch.full_like(timestep, self.config.guidance_scale)

        output = self.transformer(**kwargs)
        output_tokens = output.sample if hasattr(output, "sample") else output[0]
        output_tokens = self.from_flux_channels(output_tokens)
        return torch.tanh(self._unpatchify(output_tokens, grid_h, grid_w))
