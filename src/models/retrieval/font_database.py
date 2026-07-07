from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .font_renderer import render_text_with_font


@dataclass
class FontEmbeddingDatabase:
    """In-memory font embedding database for two-stage retrieval.

    Shapes:
        font_ids: list with length ``F``.
        clip_embeddings: ``[F, D_clip]`` normalized prototype embeddings for
            coarse retrieval.
        dino_embeddings: ``[F, D_dino]`` normalized prototype embeddings for
            local-structure reranking.
    """

    font_ids: list[str]
    clip_embeddings: torch.Tensor
    dino_embeddings: torch.Tensor
    prototype_texts: list[str] | None = None

    def to(self, device: torch.device) -> "FontEmbeddingDatabase":
        return FontEmbeddingDatabase(
            self.font_ids,
            self.clip_embeddings.to(device),
            self.dino_embeddings.to(device),
            self.prototype_texts,
        )

    @property
    def embeddings(self) -> torch.Tensor:
        """Backward-compatible alias for coarse embeddings."""

        return self.clip_embeddings


def _normalize_database_payload(payload: object) -> FontEmbeddingDatabase:
    if isinstance(payload, FontEmbeddingDatabase):
        return payload

    if isinstance(payload, dict) and {"font_ids", "clip_embeddings", "dino_embeddings"}.issubset(payload.keys()):
        font_ids = [str(font_id) for font_id in payload["font_ids"]]
        clip_embeddings = F.normalize(torch.as_tensor(payload["clip_embeddings"]).float(), dim=-1)
        dino_embeddings = F.normalize(torch.as_tensor(payload["dino_embeddings"]).float(), dim=-1)
        prototype_texts = payload.get("prototype_texts")
        if prototype_texts is not None:
            prototype_texts = [str(text) for text in prototype_texts]
        return FontEmbeddingDatabase(font_ids, clip_embeddings, dino_embeddings, prototype_texts)

    if isinstance(payload, dict) and "font_ids" in payload and "embeddings" in payload:
        font_ids = [str(font_id) for font_id in payload["font_ids"]]
        embeddings = torch.as_tensor(payload["embeddings"]).float()
        embeddings = F.normalize(embeddings, dim=-1)
        return FontEmbeddingDatabase(font_ids, embeddings, embeddings)

    if isinstance(payload, dict):
        font_ids = [str(font_id) for font_id in payload.keys()]
        embeddings = torch.stack([torch.as_tensor(payload[font_id]).float().flatten() for font_id in payload.keys()])
        embeddings = F.normalize(embeddings, dim=-1)
        return FontEmbeddingDatabase(font_ids, embeddings, embeddings)

    raise TypeError(
        "font_db.pt must contain {font_id: embedding}, {'font_ids', 'embeddings'}, "
        "or {'font_ids', 'clip_embeddings', 'dino_embeddings'}."
    )


def load_font_database(path: str | Path) -> FontEmbeddingDatabase:
    """Load ``font_db.pt`` from disk."""

    payload = torch.load(Path(path), map_location="cpu")
    return _normalize_database_payload(payload)


def save_font_database(database: FontEmbeddingDatabase, path: str | Path) -> None:
    """Save a normalized font embedding database."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "font_ids": database.font_ids,
            "clip_embeddings": database.clip_embeddings.cpu(),
            "dino_embeddings": database.dino_embeddings.cpu(),
            "prototype_texts": database.prototype_texts,
        },
        output_path,
    )


DEFAULT_PROTOTYPE_TEXTS = [
    "永",
    "和",
    "风",
    "明",
    "书",
    "山",
    "水",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "天地玄黄",
    "字体风格",
    "春夏秋冬",
]


def _encode_rendered_prototypes(
    images: list[torch.Tensor],
    embedding_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """Encode rendered glyph set and average normalized prototype embeddings."""

    encoded: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size])
        with torch.no_grad():
            embedding = embedding_fn(batch).detach().float()
        encoded.append(embedding.flatten(1))
    embeddings = F.normalize(torch.cat(encoded, dim=0), dim=-1)
    return F.normalize(embeddings.mean(dim=0), dim=-1)


def build_font_database(
    font_paths: Iterable[str | Path],
    embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    output_path: str | Path | None = None,
    charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    image_size: int = 64,
    clip_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    dino_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prototype_texts: Iterable[str] | None = None,
    batch_size: int = 8,
) -> FontEmbeddingDatabase:
    """Render representative glyph sets and build a two-encoder database.

    Args:
        font_paths: iterable of font files. The path string is used as
            ``font_id``.
        embedding_fn: backward-compatible single encoder. If provided without
            ``clip_embedding_fn`` or ``dino_embedding_fn``, it is used for both.
        output_path: optional destination for ``font_db.pt``.
        charset: backward-compatible fixed text. Used only when
            ``prototype_texts`` is not provided.
        image_size: rendered glyph canvas size.
        clip_embedding_fn: coarse encoder receiving image batch ``[B, 3, H, W]``
            and returning ``[B, D_clip]``.
        dino_embedding_fn: rerank encoder receiving image batch ``[B, 3, H, W]``
            and returning ``[B, D_dino]``.
        prototype_texts: 10-20 representative glyph strings rendered per font.

    Returns:
        ``FontEmbeddingDatabase`` with CLIP and DINO prototype embeddings.
    """

    if clip_embedding_fn is None:
        clip_embedding_fn = embedding_fn
    if dino_embedding_fn is None:
        dino_embedding_fn = embedding_fn
    if clip_embedding_fn is None or dino_embedding_fn is None:
        raise ValueError("build_font_database requires clip_embedding_fn and dino_embedding_fn.")

    texts = list(prototype_texts or DEFAULT_PROTOTYPE_TEXTS or [charset])
    if not 10 <= len(texts) <= 20:
        raise ValueError(f"prototype_texts should contain 10-20 glyph sets, got {len(texts)}.")

    font_ids: list[str] = []
    clip_embeddings: list[torch.Tensor] = []
    dino_embeddings: list[torch.Tensor] = []

    for font_path in font_paths:
        font_id = str(font_path)
        images = [render_text_with_font(text, font_id, image_size=image_size) for text in texts]
        font_ids.append(font_id)
        clip_embeddings.append(_encode_rendered_prototypes(images, clip_embedding_fn, batch_size))
        dino_embeddings.append(_encode_rendered_prototypes(images, dino_embedding_fn, batch_size))

    if not clip_embeddings:
        raise ValueError("Cannot build font database from an empty font list.")

    database = FontEmbeddingDatabase(
        font_ids=font_ids,
        clip_embeddings=F.normalize(torch.stack(clip_embeddings), dim=-1),
        dino_embeddings=F.normalize(torch.stack(dino_embeddings), dim=-1),
        prototype_texts=texts,
    )
    if output_path is not None:
        save_font_database(database, output_path)
    return database
