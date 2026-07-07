from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .font_database import FontEmbeddingDatabase, load_font_database


@dataclass
class RetrievalResult:
    """Two-stage retrieval output for one query image.

    Attributes:
        font_ids: top-k font ids after DINO reranking.
        scores: final rerank scores, usually cosine similarity in DINO space.
        coarse_font_ids: stage-1 candidate ids from CLIP coarse search.
        coarse_scores: stage-1 cosine scores.
    """

    font_ids: list[str]
    scores: list[float]
    coarse_font_ids: list[str]
    coarse_scores: list[float]


class FontRetriever:
    """Two-stage font retriever for retrieval-augmented content prior.

    Input:
        style_image: ``[B, 3, H, W]`` when encoder functions are provided, or
        direct style embeddings via ``retrieve_with_scores``.

    Output:
        top-k font ids and retrieval scores. ``__call__`` keeps the historical
        behavior and returns only font ids.
    """

    def __init__(
        self,
        font_db_path: str | Path | None = None,
        database: FontEmbeddingDatabase | None = None,
        embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        clip_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dino_embedding_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cache_enabled: bool = True,
        coarse_k: int = 32,
    ) -> None:
        if database is None and font_db_path is not None:
            database = load_font_database(font_db_path)

        self.database = database
        self.clip_embedding_fn = clip_embedding_fn or embedding_fn
        self.dino_embedding_fn = dino_embedding_fn or embedding_fn
        self.cache_enabled = cache_enabled
        self.coarse_k = coarse_k
        self._cache: dict[tuple[float, ...], RetrievalResult] = {}

    def has_database(self) -> bool:
        return self.database is not None and len(self.database.font_ids) > 0

    def _cache_key(self, clip_embedding: torch.Tensor, dino_embedding: torch.Tensor, k: int) -> tuple[float, ...]:
        values = torch.cat(
            [
                clip_embedding.detach().float().cpu().flatten()[:8],
                dino_embedding.detach().float().cpu().flatten()[:8],
                torch.tensor([float(k)]),
            ]
        )
        return tuple(round(float(value), 4) for value in values)

    def _validate_dims(self, clip_embedding: torch.Tensor, dino_embedding: torch.Tensor) -> None:
        if not self.has_database():
            raise RuntimeError("FontRetriever requires a non-empty font database.")
        if clip_embedding.shape[-1] != self.database.clip_embeddings.shape[-1]:
            raise ValueError(
                f"CLIP query dim {clip_embedding.shape[-1]} does not match database dim "
                f"{self.database.clip_embeddings.shape[-1]}."
            )
        if dino_embedding.shape[-1] != self.database.dino_embeddings.shape[-1]:
            raise ValueError(
                f"DINO query dim {dino_embedding.shape[-1]} does not match database dim "
                f"{self.database.dino_embeddings.shape[-1]}."
            )

    def retrieve_with_scores(
        self,
        clip_embedding: torch.Tensor,
        dino_embedding: torch.Tensor,
        k: int = 1,
        coarse_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return two-stage retrieval results.

        Shape:
            clip_embedding: ``[B, D_clip]``.
            dino_embedding: ``[B, D_dino]``.
        """

        self._validate_dims(clip_embedding, dino_embedding)
        database = self.database.to(clip_embedding.device)
        clip_query = F.normalize(clip_embedding.detach().float(), dim=-1)
        dino_query = F.normalize(dino_embedding.detach().float().to(clip_embedding.device), dim=-1)
        clip_keys = F.normalize(database.clip_embeddings.float(), dim=-1)
        dino_keys = F.normalize(database.dino_embeddings.float(), dim=-1)

        candidate_count = min(max(k, coarse_k or self.coarse_k), len(database.font_ids))
        clip_scores = clip_query @ clip_keys.T
        coarse_scores, coarse_indices = clip_scores.topk(candidate_count, dim=-1)

        results: list[RetrievalResult] = []
        for batch_idx in range(clip_query.shape[0]):
            candidate_indices = coarse_indices[batch_idx]
            candidate_dino = dino_keys[candidate_indices]
            rerank_scores = dino_query[batch_idx : batch_idx + 1] @ candidate_dino.T
            final_scores, final_order = rerank_scores.squeeze(0).topk(min(k, candidate_indices.numel()), dim=-1)
            final_indices = candidate_indices[final_order]
            results.append(
                RetrievalResult(
                    font_ids=[database.font_ids[int(index)] for index in final_indices],
                    scores=[float(score) for score in final_scores.detach().cpu()],
                    coarse_font_ids=[database.font_ids[int(index)] for index in candidate_indices],
                    coarse_scores=[float(score) for score in coarse_scores[batch_idx].detach().cpu()],
                )
            )
        return results

    def retrieve_fonts(self, style_embedding: torch.Tensor, k: int = 1) -> list[list[str]]:
        """Backward-compatible single-embedding retrieval.

        This path uses the same embedding for coarse and rerank stages. New code
        should call ``retrieve_with_scores`` with separate CLIP and DINO
        embeddings.
        """

        results = self.retrieve_with_scores(style_embedding, style_embedding, k=k)
        return [result.font_ids for result in results]

    def retrieve_from_image(self, style_image: torch.Tensor, k: int = 1) -> list[RetrievalResult]:
        """Encode style image, then run CLIP coarse retrieval and DINO rerank."""

        if self.clip_embedding_fn is None or self.dino_embedding_fn is None:
            raise RuntimeError("FontRetriever.retrieve_from_image requires CLIP and DINO embedding functions.")

        with torch.no_grad():
            clip_embedding = self.clip_embedding_fn(style_image)
            dino_embedding = self.dino_embedding_fn(style_image)

        if not self.cache_enabled:
            return self.retrieve_with_scores(clip_embedding, dino_embedding, k=k)

        results: list[RetrievalResult | None] = [None] * clip_embedding.shape[0]
        missed_clip: list[torch.Tensor] = []
        missed_dino: list[torch.Tensor] = []
        missed_positions: list[int] = []
        for position, (clip_item, dino_item) in enumerate(zip(clip_embedding, dino_embedding)):
            key = self._cache_key(clip_item, dino_item, k)
            if key in self._cache:
                cached = self._cache[key]
                results[position] = RetrievalResult(
                    font_ids=cached.font_ids[:k],
                    scores=cached.scores[:k],
                    coarse_font_ids=cached.coarse_font_ids,
                    coarse_scores=cached.coarse_scores,
                )
            else:
                missed_clip.append(clip_item)
                missed_dino.append(dino_item)
                missed_positions.append(position)

        if missed_clip:
            retrieved = self.retrieve_with_scores(torch.stack(missed_clip), torch.stack(missed_dino), k=k)
            for position, result in zip(missed_positions, retrieved):
                key = self._cache_key(clip_embedding[position], dino_embedding[position], k)
                self._cache[key] = result
                results[position] = result

        return [result for result in results if result is not None]

    def __call__(self, style_image: torch.Tensor, k: int = 1) -> list[list[str]]:
        return [result.font_ids for result in self.retrieve_from_image(style_image, k=k)]
