from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .font_database import FontEmbeddingDatabase, load_font_database


class FontRetriever:
    """Retrieve nearest font ids from a font embedding database.

    Input:
        style_image: ``[B, 3, H, W]`` when ``embedding_fn`` is provided, or
        style_embedding: ``[B, D]`` for direct retrieval.

    Output:
        top-k font ids for each sample.
    """

    def __init__(
        self,
        font_db_path: str | Path | None = None,
        database: FontEmbeddingDatabase | None = None,
        embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cache_enabled: bool = True,
    ) -> None:
        if database is None and font_db_path is not None:
            database = load_font_database(font_db_path)

        self.database = database
        self.embedding_fn = embedding_fn
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[float, ...], list[str]] = {}
        self._faiss_index = None
        self._index_device: torch.device | None = None

    def has_database(self) -> bool:
        return self.database is not None and len(self.database.font_ids) > 0

    def _cache_key(self, embedding: torch.Tensor) -> tuple[float, ...]:
        values = embedding.detach().float().cpu().flatten()[:16]
        return tuple(round(float(value), 4) for value in values)

    def _build_faiss_index(self, device: torch.device):
        """Build a FAISS inner-product index over normalized embeddings."""

        if not self.has_database():
            raise RuntimeError("FontRetriever requires a non-empty font database.")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FontRetriever requires faiss-cpu or faiss-gpu for retrieval.") from exc

        database = self.database.to(torch.device("cpu"))
        keys = F.normalize(database.embeddings.float(), dim=-1).cpu().numpy().astype("float32")
        index = faiss.IndexFlatIP(keys.shape[1])
        index.add(np.ascontiguousarray(keys))

        if device.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, device.index or 0, index)

        self._faiss_index = index
        self._index_device = device
        return index

    def retrieve_fonts(self, style_embedding: torch.Tensor, k: int = 1) -> list[list[str]]:
        """Return top-k font ids for each style embedding.

        Shape:
            ``style_embedding`` is ``[B, D]``.
        """

        if not self.has_database():
            raise RuntimeError("FontRetriever requires a non-empty font database.")

        if style_embedding.shape[-1] != self.database.embeddings.shape[-1]:
            raise ValueError(
                f"Query dim {style_embedding.shape[-1]} does not match font database dim {self.database.embeddings.shape[-1]}."
            )

        if self._faiss_index is None or self._index_device != style_embedding.device:
            self._build_faiss_index(style_embedding.device)

        query = F.normalize(style_embedding.detach().float(), dim=-1)
        query_np = np.ascontiguousarray(query.cpu().numpy().astype("float32"))
        _, top_indices_np = self._faiss_index.search(query_np, min(k, len(self.database.font_ids)))

        results: list[list[str]] = []
        for row in top_indices_np:
            results.append([self.database.font_ids[int(index)] for index in row])
        return results

    def __call__(self, style_image: torch.Tensor, k: int = 1) -> list[list[str]]:
        if self.embedding_fn is None:
            raise RuntimeError("FontRetriever(style_image) requires embedding_fn.")

        with torch.no_grad():
            style_embedding = self.embedding_fn(style_image)

        if not self.cache_enabled:
            return self.retrieve_fonts(style_embedding, k=k)

        results: list[list[str]] = []
        missed_embeddings: list[torch.Tensor] = []
        missed_positions: list[int] = []
        for position, embedding in enumerate(style_embedding):
            key = self._cache_key(embedding)
            if key in self._cache:
                results.append(self._cache[key][:k])
            else:
                results.append([])
                missed_embeddings.append(embedding)
                missed_positions.append(position)

        if missed_embeddings:
            retrieved = self.retrieve_fonts(torch.stack(missed_embeddings), k=k)
            for position, font_ids in zip(missed_positions, retrieved):
                key = self._cache_key(style_embedding[position])
                self._cache[key] = font_ids
                results[position] = font_ids

        return results
