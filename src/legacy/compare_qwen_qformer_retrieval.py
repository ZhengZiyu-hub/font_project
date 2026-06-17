from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_results(path: str | Path, required: bool) -> list[dict]:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Retrieval result not found: {path}")
        print(f"[WARNING] Optional retrieval result not found: {path}")
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def result_map(records: list[dict]) -> dict[str, dict]:
    return {str(record.get("query_image")): record for record in records}


def copy_image(path: str | None, assets_dir: Path, name: str) -> str | None:
    if not path or not Path(path).exists():
        return None
    suffix = Path(path).suffix
    destination = assets_dir / f"{name}{suffix}"
    shutil.copy2(path, destination)
    return f"assets/{destination.name}"


def cell(path: str | None, src: str | None, label: str, score: float | None = None) -> str:
    image = (
        f'<img src="{html.escape(src)}" loading="lazy" />'
        if src
        else '<div class="missing">unavailable</div>'
    )
    score_text = "" if score is None else f'<div class="score">{score:.4f}</div>'
    return (
        f'<div class="cell">{image}<div class="label">{html.escape(label)}</div>'
        f"{score_text}<div class=\"path\">{html.escape(str(path or ''))}</div></div>"
    )


def method_row(
    method: str,
    record: dict | None,
    query_number: int,
    assets_dir: Path,
    topk: int,
) -> str:
    if record is None:
        return (
            f'<div class="method-row"><div class="method">{html.escape(method)}</div>'
            '<div class="unavailable">retrieval results unavailable</div></div>'
        )
    cells = []
    for result in record.get("results", [])[:topk]:
        rank = int(result.get("rank", len(cells) + 1))
        path = result.get("image_path")
        src = copy_image(path, assets_dir, f"q{query_number}_{method}_r{rank}")
        cells.append(cell(path, src, f"rank {rank}", float(result["score"])))
    return (
        f'<div class="method-row"><div class="method">{html.escape(method)}</div>'
        f'<div class="results">{"".join(cells)}</div></div>'
    )


def compare(args: argparse.Namespace) -> Path:
    pooled = read_results(args.qwen_pooled_results, required=True)
    tokens = read_results(args.qwen_token_results, required=False)
    qformer = read_results(args.qformer_results, required=True)
    token_map = result_map(tokens)
    qformer_map = result_map(qformer)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    groups = []

    for query_number, pooled_record in enumerate(pooled[: args.max_queries]):
        query_path = str(pooled_record.get("query_image"))
        query_src = copy_image(query_path, assets_dir, f"q{query_number}_query")
        groups.append(
            '<section class="query">'
            f'<div class="query-head">{cell(query_path, query_src, f"query #{query_number}")}</div>'
            '<div class="methods">'
            + method_row("Qwen pooled", pooled_record, query_number, assets_dir, args.topk)
            + method_row("Qwen token matching", token_map.get(query_path), query_number, assets_dir, args.topk)
            + method_row("QFormer style", qformer_map.get(query_path), query_number, assets_dir, args.topk)
            + "</div></section>"
        )

    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Qwen vs QFormer Retrieval</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; background: #f5f5f5; color: #222; }}
    h1 {{ font-size: 22px; }}
    .query {{ display: flex; gap: 12px; background: white; border: 1px solid #ddd; margin: 14px 0; padding: 10px; }}
    .query-head {{ flex: 0 0 125px; border-right: 2px solid #888; padding-right: 10px; }}
    .methods {{ min-width: 0; flex: 1; }}
    .method-row {{ display: flex; align-items: flex-start; min-height: 130px; border-bottom: 1px solid #eee; padding: 4px 0; }}
    .method {{ width: 145px; flex: 0 0 145px; font-weight: bold; padding-top: 36px; }}
    .results {{ display: flex; gap: 8px; overflow-x: auto; }}
    .cell {{ width: 112px; flex: 0 0 112px; text-align: center; font-size: 11px; }}
    img, .missing {{ width: 88px; height: 88px; object-fit: contain; border: 1px solid #bbb; background: white; }}
    .missing {{ display: flex; align-items: center; justify-content: center; margin: auto; color: #a33; }}
    .label {{ margin-top: 3px; }}
    .score {{ font-weight: bold; }}
    .path {{ color: #777; overflow-wrap: anywhere; text-align: left; }}
    .unavailable {{ color: #a33; padding-top: 36px; }}
  </style>
</head>
<body>
  <h1>Qwen pooled vs token matching vs QFormer style</h1>
  <p>Showing {min(args.max_queries, len(pooled))} queries, Top-{args.topk} per method.</p>
  {''.join(groups)}
</body>
</html>
"""
    output_path = output_dir / "compare_gallery.html"
    output_path.write_text(document, encoding="utf-8")
    print(f"saved comparison gallery: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen and QFormer retrieval galleries.")
    parser.add_argument("--qwen_pooled_results", required=True)
    parser.add_argument("--qwen_token_results", required=True)
    parser.add_argument("--qformer_results", required=True)
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "output/compare_qwen_qformer")
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
