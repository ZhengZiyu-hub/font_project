from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "output" / "features_qwen_vl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "feature_bank"


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


def metadata_by_pooled_path(records: list[dict]) -> dict[str, dict]:
    mapping = {}
    for record in records:
        pooled_path = record.get("pooled_feature_path")
        if not pooled_path:
            continue
        mapping[str(Path(pooled_path).resolve())] = record
    return mapping


def build_feature_bank(
    feature_dir: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path) if metadata_path else feature_dir / "metadata.jsonl"
    records = read_jsonl(metadata_path)
    pooled_paths = sorted(feature_dir.glob("*_pooled.npy"), key=lambda p: p.name)
    if not pooled_paths:
        raise RuntimeError(f"No *_pooled.npy files found under: {feature_dir}")

    record_map = metadata_by_pooled_path(records)
    features = []
    aligned_records = []
    print(f"Found pooled features: {len(pooled_paths)}")
    print(f"Found metadata records: {len(records)}")

    for idx, pooled_path in enumerate(pooled_paths, start=1):
        feature = np.load(pooled_path)
        if feature.ndim != 1:
            raise ValueError(f"Expected 1D pooled feature, got shape={feature.shape}: {pooled_path}")
        resolved = str(pooled_path.resolve())
        record = record_map.get(resolved)
        if record is None:
            raise KeyError(
                f"No metadata record matched pooled feature: {pooled_path}. "
                "Check pooled_feature_path values in metadata.jsonl."
            )
        record = dict(record)
        record["pooled_feature_path"] = str(pooled_path)
        record.setdefault("feature_shape", {"pooled": list(feature.shape)})
        features.append(feature.astype(np.float32, copy=False))
        aligned_records.append(record)
        if idx == 1 or idx % 500 == 0 or idx == len(pooled_paths):
            print(f"[{idx}/{len(pooled_paths)}] loaded {pooled_path.name} shape={feature.shape}")

    matrix = l2_normalize_matrix(np.stack(features, axis=0))
    output_dir.mkdir(parents=True, exist_ok=True)
    features_out = output_dir / "features.npy"
    metadata_out = output_dir / "metadata.jsonl"
    np.save(features_out, matrix)
    write_jsonl(metadata_out, aligned_records)
    print(f"saved features: {features_out}")
    print(f"saved metadata: {metadata_out}")
    print(f"features.npy shape: {matrix.shape}")
    return matrix, aligned_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized Qwen-VL pooled feature bank.")
    parser.add_argument("--feature_dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_feature_bank(args.feature_dir, args.output_dir, args.metadata)


if __name__ == "__main__":
    main()
