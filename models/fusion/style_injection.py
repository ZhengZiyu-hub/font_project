from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_CALLIGRAPHER_ROOT = Path("/data/zhengziyu/Calligrapher")
DEFAULT_FLUX_FILL_PATH = DEFAULT_CALLIGRAPHER_ROOT / "pretrained/FLUX.1-Fill-dev"
DEFAULT_FLUX_DEV_PATH = Path("/data/zhengziyu/models/FLUX.1-dev")
DEFAULT_SIGLIP_PATH = DEFAULT_CALLIGRAPHER_ROOT / "pretrained/siglip-so400m-patch14-384"
DEFAULT_CALLIGRAPHER_WEIGHTS = (
    DEFAULT_CALLIGRAPHER_ROOT / "pretrained/Calligrapher/calligrapher.bin"
)


@dataclass(frozen=True)
class LocalResourceReport:
    calligrapher_root: Path
    flux_fill_path: Path
    flux_dev_path: Path
    siglip_path: Path
    calligrapher_weights: Path
    flux_fill_complete: bool
    flux_dev_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "calligrapher_root": str(self.calligrapher_root),
            "flux_fill_path": str(self.flux_fill_path),
            "flux_dev_path": str(self.flux_dev_path),
            "siglip_path": str(self.siglip_path),
            "calligrapher_weights": str(self.calligrapher_weights),
            "flux_fill_complete": self.flux_fill_complete,
            "flux_dev_complete": self.flux_dev_complete,
        }


def _has_model_index(path: Path) -> bool:
    return (path / "model_index.json").is_file()


def _has_transformer_weights(path: Path) -> bool:
    transformer_dir = path / "transformer"
    return any(transformer_dir.glob("*.safetensors")) or any(transformer_dir.glob("*.bin"))


def is_complete_flux_model(path: str | Path) -> bool:
    path = Path(path)
    return _has_model_index(path) and _has_transformer_weights(path)


def inspect_local_resources(
    calligrapher_root: str | Path = DEFAULT_CALLIGRAPHER_ROOT,
    flux_fill_path: str | Path = DEFAULT_FLUX_FILL_PATH,
    flux_dev_path: str | Path = DEFAULT_FLUX_DEV_PATH,
    siglip_path: str | Path = DEFAULT_SIGLIP_PATH,
    calligrapher_weights: str | Path = DEFAULT_CALLIGRAPHER_WEIGHTS,
) -> LocalResourceReport:
    flux_fill_path = Path(flux_fill_path)
    flux_dev_path = Path(flux_dev_path)
    return LocalResourceReport(
        calligrapher_root=Path(calligrapher_root),
        flux_fill_path=flux_fill_path,
        flux_dev_path=flux_dev_path,
        siglip_path=Path(siglip_path),
        calligrapher_weights=Path(calligrapher_weights),
        flux_fill_complete=_has_model_index(flux_fill_path) and _has_transformer_weights(flux_fill_path),
        flux_dev_complete=_has_model_index(flux_dev_path) and _has_transformer_weights(flux_dev_path),
    )


def require_calligrapher_on_path(calligrapher_root: str | Path = DEFAULT_CALLIGRAPHER_ROOT) -> Path:
    root = Path(calligrapher_root).resolve()
    if not (root / "models/calligrapher.py").is_file():
        raise FileNotFoundError(f"Calligrapher source was not found under {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


class StyleTokenProjectionAdapter(nn.Module):
    """Project arbitrary style tokens into diffusion adapter token space.

    This keeps the current font_project QFormer interface compatible with the
    Calligrapher generation path. The original Calligrapher path uses SigLIP
    pooled features plus its own QFormer/MLP weights; this adapter is for
    training or future experiments that start from the existing Qwen-token
    style encoder.
    """

    def __init__(
        self,
        input_dim: int = 768,
        cross_attention_dim: int = 4096,
        num_tokens: int = 128,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attention_dim),
        )
        self.query = nn.Parameter(torch.randn(num_tokens, cross_attention_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cross_attention_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, style_tokens: torch.Tensor) -> torch.Tensor:
        if style_tokens.ndim != 3:
            raise ValueError(f"Expected style_tokens [B, N, D], got {tuple(style_tokens.shape)}")
        if style_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected style dim {self.input_dim}, got {style_tokens.shape[-1]}")
        kv = self.proj(style_tokens)
        query = self.query.unsqueeze(0).expand(style_tokens.shape[0], -1, -1)
        attended, _ = self.attn(query=query, key=kv, value=kv, need_weights=False)
        return self.norm(query + attended)


def save_resource_report(path: str | Path, report: LocalResourceReport) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# TODO: replace the IP-Adapter-style attention addition in Calligrapher's
# FluxAttnProcessor with strict K/V replacement if reproducing the paper's
# exact style-attention intervention becomes necessary.
