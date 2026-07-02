from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .font_renderer import render_text_with_font


@dataclass
class FontEmbeddingDatabase:
    """In-memory font embedding database.

    Shapes:
        font_ids: list with length ``F``.
        embeddings: ``[F, D]`` normalized font embeddings.
    """

    font_ids: list[str]
    embeddings: torch.Tensor

    def to(self, device: torch.device) -> "FontEmbeddingDatabase":
        return FontEmbeddingDatabase(self.font_ids, self.embeddings.to(device))


def _normalize_database_payload(payload: object) -> FontEmbeddingDatabase:
    if isinstance(payload, FontEmbeddingDatabase):
        return payload

    if isinstance(payload, dict) and "font_ids" in payload and "embeddings" in payload:
        font_ids = [str(font_id) for font_id in payload["font_ids"]]
        embeddings = torch.as_tensor(payload["embeddings"]).float()
        return FontEmbeddingDatabase(font_ids, F.normalize(embeddings, dim=-1))

    if isinstance(payload, dict):
        font_ids = [str(font_id) for font_id in payload.keys()]
        embeddings = torch.stack([torch.as_tensor(payload[font_id]).float().flatten() for font_id in payload.keys()])
        return FontEmbeddingDatabase(font_ids, F.normalize(embeddings, dim=-1))

    raise TypeError("font_db.pt must contain {font_id: embedding} or {'font_ids', 'embeddings'}.")


def load_font_database(path: str | Path) -> FontEmbeddingDatabase:
    """Load ``font_db.pt`` from disk."""

    payload = torch.load(Path(path), map_location="cpu")
    return _normalize_database_payload(payload)


def save_font_database(database: FontEmbeddingDatabase, path: str | Path) -> None:
    """Save a normalized font embedding database."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"font_ids": database.font_ids, "embeddings": database.embeddings.cpu()}, output_path)


def build_font_database(
    font_paths: Iterable[str | Path],
    embedding_fn: Callable[[torch.Tensor], torch.Tensor],
    output_path: str | Path | None = None,
    charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    image_size: int = 64,
) -> FontEmbeddingDatabase:
    """Render fonts and build a normalized embedding database.

    Args:
        font_paths: iterable of font files. The path string is used as
            ``font_id``.
        embedding_fn: callable receiving image batch ``[B, 3, H, W]`` and
            returning embeddings ``[B, D]``.
        output_path: optional destination for ``font_db.pt``.
        charset: fixed text used to render each font.
        image_size: rendered glyph canvas size.

    Returns:
        ``FontEmbeddingDatabase`` with embeddings shape ``[F, D]``.
    """

    font_ids: list[str] = []
    embeddings: list[torch.Tensor] = []

    for font_path in font_paths:
        font_id = str(font_path)
        image = render_text_with_font(charset, font_id, image_size=image_size).unsqueeze(0)
        with torch.no_grad():
            embedding = embedding_fn(image).detach().float().flatten()
        font_ids.append(font_id)
        embeddings.append(embedding)

    if not embeddings:
        raise ValueError("Cannot build font database from an empty font list.")

    database = FontEmbeddingDatabase(font_ids, F.normalize(torch.stack(embeddings), dim=-1))
    if output_path is not None:
        save_font_database(database, output_path)
    return database
