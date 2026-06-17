from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.qwen_token_feature_dataset import QwenTokenFeatureDataset
from models.encoders.qformer_style_encoder import QFormerStyleEncoder


def load_model(checkpoint_path: str | Path, device: torch.device) -> QFormerStyleEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config")
    if not config:
        raise KeyError("Checkpoint is missing model_config.")
    model = QFormerStyleEncoder(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    model = load_model(args.checkpoint, device)
    dataset = QwenTokenFeatureDataset(
        args.feature_dir,
        args.metadata,
        args.annotation_file,
        expected_input_dim=model.config.input_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_records = []

    with torch.inference_mode():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            style_tokens, style_pooled = model(tokens)
            style_tokens = style_tokens.float().cpu().numpy()
            style_pooled = style_pooled.float().cpu().numpy()
            for offset, dataset_index in enumerate(batch["index"].tolist()):
                sample = dataset.samples[dataset_index]
                image_path = Path(sample["image_path"])
                stem = image_path.stem or Path(sample["tokens_path"]).stem.removesuffix("_tokens")
                tokens_path = output_dir / f"{stem}_qformer_style_tokens.npy"
                pooled_path = output_dir / f"{stem}_qformer_style_pooled.npy"
                np.save(tokens_path, style_tokens[offset])
                np.save(pooled_path, style_pooled[offset])
                metadata_records.append(
                    {
                        "image_path": sample["image_path"],
                        "source_token_feature_path": sample["tokens_path"],
                        "style_token_feature_path": str(tokens_path),
                        "style_pooled_feature_path": str(pooled_path),
                        "label": sample["label"],
                        "feature_shape": {
                            "style_tokens": list(style_tokens[offset].shape),
                            "style_pooled": list(style_pooled[offset].shape),
                        },
                    }
                )
            print(f"[{len(metadata_records)}/{len(dataset)}] extracted")

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in metadata_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract QFormer font-style features.")
    parser.add_argument("--feature_dir", default=PROJECT_ROOT / "output/features_qwen_vl")
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--annotation_file", default=PROJECT_ROOT / "input/dataset_3000/annotations.jsonl"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "output/features_qformer_style")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.metadata is None:
        args.metadata = Path(args.feature_dir) / "metadata.jsonl"
    return args


if __name__ == "__main__":
    extract(parse_args())
