from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(results_path: str | Path, annotation_file: str | Path, topk: int) -> dict:
    annotations = read_jsonl(annotation_file)
    labels = {
        Path(str(record["image"])).stem: PureWindowsPath(str(record["font"])).name
        for record in annotations
    }
    results = read_jsonl(results_path)
    top1_correct = 0
    hit_count = 0
    positive_count = 0
    reciprocal_rank_sum = 0.0

    for query in results:
        query_stem = Path(str(query["query_image"])).stem
        query_label = labels[query_stem]
        retrieved = query.get("results", [])[:topk]
        matches = [
            labels.get(Path(str(item["image_path"])).stem) == query_label for item in retrieved
        ]
        top1_correct += int(bool(matches) and matches[0])
        positive_count += sum(matches)
        if any(matches):
            hit_count += 1
            reciprocal_rank_sum += 1.0 / (matches.index(True) + 1)

    query_count = len(results)
    denominator = max(query_count, 1)
    return {
        "results_path": str(Path(results_path).resolve()),
        "queries": query_count,
        "topk": topk,
        "top1_accuracy": top1_correct / denominator,
        f"precision_at_{topk}": positive_count / max(query_count * topk, 1),
        f"hit_rate_at_{topk}": hit_count / denominator,
        f"mrr_at_{topk}": reciprocal_rank_sum / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using normalized font labels.")
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--annotation_file",
        default=PROJECT_ROOT / "input/dataset_3000/annotations.jsonl",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = evaluate(args.results, args.annotation_file, args.topk)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
