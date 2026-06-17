from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def prepare(args: argparse.Namespace) -> None:
    source_records = read_jsonl(args.metadata)
    annotations = read_jsonl(args.annotation_file)
    annotation_root = Path(args.annotation_file).resolve().parent
    annotations_by_stem = {
        Path(str(record["image"])).stem: record for record in annotations
    }
    output_records = []
    for record in source_records:
        token_path = Path(str(record["token_feature_path"]))
        stem = token_path.stem.removesuffix("_tokens")
        annotation = annotations_by_stem.get(stem)
        if annotation is None:
            raise KeyError(f"No annotation matched feature stem: {stem}")
        image_path = Path(str(annotation["image"]))
        if not image_path.is_absolute():
            image_path = annotation_root / image_path
        enriched = dict(record)
        enriched.update(
            {
                "image_path": str(image_path.resolve()),
                "font_name": PureWindowsPath(str(annotation["font"])).name,
                "font_path": annotation["font"],
                "font_group": annotation.get("font_group"),
                "text": annotation.get("text"),
                "annotation": annotation,
            }
        )
        output_records.append(enriched)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved labeled metadata: {output_path}")
    print(
        f"records={len(output_records)} "
        f"font_classes={len({record['font_name'] for record in output_records})}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Qwen feature metadata with font labels.")
    parser.add_argument(
        "--metadata",
        default=PROJECT_ROOT / "output/features_qwen_vl/metadata.jsonl",
    )
    parser.add_argument(
        "--annotation_file",
        default=PROJECT_ROOT / "input/dataset_3000/annotations.jsonl",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "output/features_qwen_vl/metadata_labeled.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
