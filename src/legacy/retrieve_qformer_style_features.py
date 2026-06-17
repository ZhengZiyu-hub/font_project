from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve_similar_crops import (
    build_results,
    l2_normalize_matrix,
    numpy_search,
    read_jsonl,
    try_faiss_search,
    write_gallery,
    write_jsonl,
)


def resolve_feature_path(record: dict, feature_dir: Path) -> Path:
    value = record.get("style_pooled_feature_path")
    if value:
        path = Path(value)
        return path if path.is_absolute() else feature_dir / path
    image_stem = Path(str(record.get("image_path", ""))).stem
    return feature_dir / f"{image_stem}_qformer_style_pooled.npy"


def retrieve(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    metadata = read_jsonl(args.metadata)
    if len(metadata) < 2:
        raise ValueError("Need at least two metadata records for retrieval.")
    features = []
    aligned_metadata = []
    for record in metadata:
        feature_path = resolve_feature_path(record, feature_dir)
        feature = np.load(feature_path)
        if feature.ndim != 1:
            raise ValueError(f"Expected pooled feature [D], got {feature.shape}: {feature_path}")
        features.append(feature.astype(np.float32, copy=False))
        aligned_metadata.append(record)
    matrix = l2_normalize_matrix(np.stack(features))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", matrix)
    topk = min(args.topk, len(metadata) - 1)
    search = try_faiss_search(matrix, topk)
    scores, indices = search if search is not None else numpy_search(matrix, topk, args.batch_size)
    results = build_results(scores, indices, aligned_metadata, topk)
    results_path = output_dir / "retrieval_results.jsonl"
    gallery_path = output_dir / "retrieval_gallery.html"
    write_jsonl(results_path, results)
    write_gallery(
        results,
        gallery_path,
        args.max_queries,
        title="QFormer Font Style Retrieval Gallery",
    )
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in aligned_metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved feature bank: {output_dir / 'features.npy'} shape={matrix.shape}")
    print(f"saved retrieval results: {results_path}")
    print(f"saved retrieval gallery: {gallery_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve crops using QFormer style features.")
    parser.add_argument("--feature_dir", default=PROJECT_ROOT / "output/features_qformer_style")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "output/retrieval_qformer_style")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_queries", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    if args.metadata is None:
        args.metadata = Path(args.feature_dir) / "metadata.jsonl"
    return args


if __name__ == "__main__":
    retrieve(parse_args())
