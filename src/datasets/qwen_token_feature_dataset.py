from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_FIELDS = ("font_name", "font_path", "font", "style_type")
MATCH_FIELDS = ("image_path", "file_name", "filename", "image", "path")


def read_jsonl(path: str | Path, required: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"JSONL file not found: {path}")
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def _record_keys(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in MATCH_FIELDS:
        value = record.get(field)
        if value:
            path = Path(str(value))
            keys.update((str(path), path.name, path.stem))
    return keys


def _annotation_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in _record_keys(record):
            index.setdefault(key, record)
    return index


def _find_annotation(record: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in _record_keys(record):
        if key in index:
            return index[key]
    return {}


def normalize_label(field: str, value: Any) -> str:
    label = str(value)
    if field in {"font_name", "font_path", "font"}:
        return PureWindowsPath(label).name
    return label


def resolve_image_path(
    metadata_record: dict[str, Any],
    annotation: dict[str, Any],
    annotation_file: str | Path | None,
) -> str:
    annotation_image = annotation.get("image")
    if annotation_image and annotation_file:
        image_path = Path(str(annotation_image))
        if not image_path.is_absolute():
            image_path = Path(annotation_file).parent / image_path
        return str(image_path.resolve())
    return str(metadata_record.get("image_path", ""))


class QwenTokenFeatureDataset(Dataset):
    def __init__(
        self,
        feature_dir: str | Path,
        metadata_path: str | Path,
        annotation_file: str | Path | None = None,
        expected_input_dim: int = 3584,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.records = read_jsonl(metadata_path)
        annotations = read_jsonl(annotation_file, required=False) if annotation_file else []
        annotation_index = _annotation_index(annotations)
        self.expected_input_dim = expected_input_dim
        self.samples: list[dict[str, Any]] = []

        label_values = []
        for record in self.records:
            annotation = _find_annotation(record, annotation_index)
            merged = dict(record)
            merged.update(annotation)
            token_path = merged.get("token_feature_path")
            if token_path:
                token_path = Path(str(token_path))
                if not token_path.is_absolute():
                    token_path = self.feature_dir / token_path
            else:
                image_path = Path(str(merged.get("image_path", "")))
                token_path = self.feature_dir / f"{image_path.stem}_tokens.npy"
            if not token_path.exists():
                raise FileNotFoundError(f"Token feature not found: {token_path}")

            label = None
            for field in LABEL_FIELDS:
                if merged.get(field) not in (None, ""):
                    label = normalize_label(field, merged[field])
                    break
            sample = {
                "tokens_path": str(token_path),
                "image_path": resolve_image_path(record, annotation, annotation_file),
                "label": label,
                "metadata": merged,
            }
            self.samples.append(sample)
            if label is not None:
                label_values.append(label)

        self.has_labels = bool(self.samples) and len(label_values) == len(self.samples)
        unique_labels = sorted(set(label_values)) if self.has_labels else []
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        tokens = np.load(sample["tokens_path"])
        if tokens.ndim != 2 or tokens.shape[-1] != self.expected_input_dim:
            raise ValueError(
                f"Expected token feature [T, {self.expected_input_dim}], "
                f"got {tokens.shape}: {sample['tokens_path']}"
            )
        label = sample["label"]
        return {
            "tokens": torch.from_numpy(tokens.astype(np.float32, copy=False)),
            "image_path": sample["image_path"],
            "label": label if label is not None else "",
            "label_id": self.label_to_id[label] if self.has_labels else -1,
            "index": index,
        }
