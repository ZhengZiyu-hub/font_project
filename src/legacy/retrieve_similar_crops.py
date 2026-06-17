from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_BANK = PROJECT_ROOT / "output" / "feature_bank" / "features.npy"
DEFAULT_METADATA = PROJECT_ROOT / "output" / "feature_bank" / "metadata.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "retrieval"


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def l2_normalize_matrix(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def try_faiss_search(features: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    print("Using FAISS IndexFlatIP for cosine retrieval.")
    features = np.ascontiguousarray(features.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    scores, indices = index.search(features, topk + 1)
    return scores, indices


def numpy_search(features: np.ndarray, topk: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    print("Using NumPy matrix multiplication for cosine retrieval.")
    n = features.shape[0]
    all_scores = np.empty((n, topk + 1), dtype=np.float32)
    all_indices = np.empty((n, topk + 1), dtype=np.int64)
    features_t = features.T

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = features[start:end] @ features_t
        rows = np.arange(end - start)
        sims[rows, np.arange(start, end)] = -np.inf
        candidate_count = min(topk + 1, n)
        kth = candidate_count - 1
        candidate_idx = np.argpartition(-sims, kth=kth, axis=1)[:, :candidate_count]
        candidate_scores = np.take_along_axis(sims, candidate_idx, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        sorted_idx = np.take_along_axis(candidate_idx, order, axis=1)
        sorted_scores = np.take_along_axis(candidate_scores, order, axis=1)
        fill = topk + 1 - sorted_idx.shape[1]
        if fill > 0:
            sorted_idx = np.pad(sorted_idx, ((0, 0), (0, fill)), constant_values=-1)
            sorted_scores = np.pad(sorted_scores, ((0, 0), (0, fill)), constant_values=-np.inf)
        all_indices[start:end] = sorted_idx[:, : topk + 1]
        all_scores[start:end] = sorted_scores[:, : topk + 1]
        print(f"[{end}/{n}] searched")
    return all_scores, all_indices


def build_results(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict],
    topk: int,
) -> list[dict]:
    results = []
    n = len(metadata)
    for query_idx in range(n):
        query_results = []
        seen = {query_idx}
        for score, idx in zip(scores[query_idx], indices[query_idx]):
            idx = int(idx)
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            item = metadata[idx]
            query_results.append(
                {
                    "rank": len(query_results) + 1,
                    "image_path": item.get("image_path"),
                    "index": idx,
                    "score": round(float(score), 6),
                }
            )
            if len(query_results) >= topk:
                break
        results.append(
            {
                "query_image": metadata[query_idx].get("image_path"),
                "query_index": query_idx,
                "results": query_results,
            }
        )
    return results


def copy_gallery_image(image_path: str | None, assets_dir: Path, asset_name: str) -> str | None:
    if not image_path:
        return None
    src = Path(image_path)
    if not src.exists():
        return None
    dst = assets_dir / asset_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"assets/{dst.name}"


def image_cell(image_path: str | None, img_src: str | None, label: str, score: float | None = None) -> str:
    escaped_label = html.escape(label)
    original_path = html.escape(str(image_path))
    if img_src:
        img_html = f'<img src="{html.escape(img_src)}" loading="lazy" />'
    else:
        img_html = '<div class="missing">missing image</div>'
    score_html = "" if score is None else f'<div class="score">{score:.4f}</div>'
    return f'<div class="cell">{img_html}<div class="label">{escaped_label}</div>{score_html}<div class="path">{original_path}</div></div>'


def asset_filename(query_index: int, basename: str, rank: int | None = None) -> str:
    safe_basename = Path(basename).name
    if rank is None:
        return f"query_{query_index}_{safe_basename}"
    return f"query_{query_index}_rank_{rank}_{safe_basename}"


def write_gallery(
    results: list[dict],
    output_path: str | Path,
    max_queries: int,
    title: str = "Qwen-VL Crop Retrieval Gallery",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in results[:max_queries]:
        query_index = int(item["query_index"])
        query_image = item["query_image"]
        query_asset = copy_gallery_image(
            query_image,
            assets_dir,
            asset_filename(query_index, Path(str(query_image)).name),
        )
        cells = [image_cell(query_image, query_asset, f"query #{query_index}")]
        for result in item["results"]:
            rank = int(result["rank"])
            image_path = result["image_path"]
            result_asset = copy_gallery_image(
                image_path,
                assets_dir,
                asset_filename(query_index, Path(str(image_path)).name, rank=rank),
            )
            cells.append(image_cell(image_path, result_asset, f"rank {rank}", float(result["score"])))
        rows.append('<div class="row">' + "\n".join(cells) + "</div>")

    result_count = max((len(item.get("results", [])) for item in results), default=0)
    escaped_title = html.escape(title)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; background: #f7f7f7; color: #222; }}
    h1 {{ font-size: 22px; margin: 0 0 14px; }}
    .note {{ margin-bottom: 18px; color: #555; }}
    .row {{ display: flex; gap: 10px; align-items: flex-start; padding: 10px; margin-bottom: 12px; background: #fff; border: 1px solid #ddd; overflow-x: auto; }}
    .cell {{ width: 130px; flex: 0 0 130px; text-align: center; font-size: 12px; }}
    img {{ width: 96px; height: 96px; object-fit: contain; border: 1px solid #ccc; background: #fff; }}
    .missing {{ width: 96px; height: 96px; border: 1px solid #c55; background: #fee; color: #a00; display: flex; align-items: center; justify-content: center; margin: 0 auto; }}
    .label {{ margin-top: 5px; color: #333; }}
    .score {{ margin-top: 3px; font-weight: 700; }}
    .path {{ margin-top: 4px; color: #666; overflow-wrap: anywhere; text-align: left; line-height: 1.25; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="note">Showing {min(max_queries, len(results))} query crops. Each row: query + Top-{result_count} similar crops.</div>
  {''.join(rows)}
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")


def retrieve(feature_bank: str | Path, metadata_path: str | Path, output_dir: str | Path, topk: int, max_queries: int, batch_size: int) -> tuple[list[dict], Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_jsonl(metadata_path)
    features = np.load(feature_bank)
    if features.ndim != 2:
        raise ValueError(f"Expected feature bank shape [N, D], got {features.shape}: {feature_bank}")
    if features.shape[0] != len(metadata):
        raise ValueError(f"features rows ({features.shape[0]}) != metadata records ({len(metadata)})")
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 features for retrieval.")

    topk = min(topk, features.shape[0] - 1)
    features = l2_normalize_matrix(features)
    print(f"feature bank shape: {features.shape}")
    search = try_faiss_search(features, topk)
    if search is None:
        scores, indices = numpy_search(features, topk, batch_size)
    else:
        scores, indices = search

    results = build_results(scores, indices, metadata, topk)
    results_path = output_dir / "retrieval_results.jsonl"
    gallery_path = output_dir / "retrieval_gallery.html"
    write_jsonl(results_path, results)
    write_gallery(results, gallery_path, max_queries=max_queries)
    print(f"saved retrieval results: {results_path}")
    print(f"saved retrieval gallery: {gallery_path}")
    return results, gallery_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve similar crop images from a Qwen-VL pooled feature bank.")
    parser.add_argument("--feature_bank", default=DEFAULT_FEATURE_BANK)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_queries", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, gallery_path = retrieve(
        feature_bank=args.feature_bank,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        topk=args.topk,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
    )
    print(f"retrieval_gallery.html path: {gallery_path}")


if __name__ == "__main__":
    main()
