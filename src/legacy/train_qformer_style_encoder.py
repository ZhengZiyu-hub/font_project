from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.qwen_token_feature_dataset import QwenTokenFeatureDataset
from models.encoders.qformer_style_encoder import QFormerStyleEncoder


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B, views, D], got {tuple(features.shape)}")
        batch_size, num_views, _ = features.shape
        features = F.normalize(features, dim=-1)
        contrast = features.reshape(batch_size * num_views, -1)
        logits = contrast @ contrast.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        device = features.device
        identity = torch.eye(batch_size * num_views, device=device, dtype=torch.bool)
        if labels is None:
            positive_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        else:
            labels = labels.reshape(-1)
            positive_mask = labels[:, None].eq(labels[None, :])
        positive_mask = positive_mask.repeat_interleave(num_views, dim=0).repeat_interleave(
            num_views, dim=1
        )
        positive_mask = positive_mask & ~identity

        exp_logits = torch.exp(logits) * (~identity)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            raise RuntimeError("No positive pairs were available for contrastive loss.")
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1)
        return -mean_log_prob[valid].mean()


def augment_tokens(
    tokens: torch.Tensor,
    token_dropout: float,
    gaussian_noise: float,
    token_mask_prob: float,
) -> torch.Tensor:
    augmented = tokens.clone()
    if gaussian_noise > 0:
        augmented = augmented + torch.randn_like(augmented) * gaussian_noise
    if token_dropout > 0:
        keep = torch.rand_like(augmented).ge(token_dropout)
        augmented = augmented * keep / max(1.0 - token_dropout, 1e-6)
    if token_mask_prob > 0:
        token_keep = torch.rand(
            augmented.shape[:2], device=augmented.device, dtype=torch.float32
        ).ge(token_mask_prob)
        augmented = augmented * token_keep.unsqueeze(-1)
    return augmented


def save_checkpoint(
    path: Path,
    model: QFormerStyleEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_config": model.get_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    dataset = QwenTokenFeatureDataset(
        args.feature_dir, args.metadata, args.annotation_file, args.input_dim
    )
    if len(dataset) < 2:
        raise ValueError("At least two feature records are required for training.")
    if dataset.has_labels:
        print(f"Using supervised contrastive learning with {len(dataset.label_to_id)} labels.")
    else:
        print(
            "[WARNING] No complete font_name/font_path/style_type labels were found. "
            "Falling back to instance-level SimCLR contrastive learning."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
    )
    model = QFormerStyleEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        style_dim=args.style_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = SupConLoss(args.temperature)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for step, batch in enumerate(loader, start=1):
            tokens = batch["tokens"].to(device, non_blocking=True)
            view_a = augment_tokens(
                tokens, args.token_dropout, args.gaussian_noise, args.token_mask_prob
            )
            view_b = augment_tokens(
                tokens, args.token_dropout, args.gaussian_noise, args.token_mask_prob
            )
            both_views = torch.cat((view_a, view_b), dim=0)
            _, pooled = model(both_views)
            batch_size = tokens.shape[0]
            pooled = torch.stack((pooled[:batch_size], pooled[batch_size:]), dim=1)
            labels = batch["label_id"].to(device) if dataset.has_labels else None
            loss = criterion(pooled, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            if step == 1 or step % args.log_every == 0:
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(loader)} "
                    f"loss={loss.item():.6f}"
                )

        epoch_loss = loss_sum / max(sample_count, 1)
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mode": "supervised" if dataset.has_labels else "instance",
            "samples": sample_count,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        epoch_path = checkpoints_dir / f"qformer_style_encoder_epoch_{epoch}.pt"
        save_checkpoint(epoch_path, model, optimizer, epoch, epoch_loss, args)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, epoch_loss, args)
        print(f"epoch={epoch} mean_loss={epoch_loss:.6f} checkpoint={epoch_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QFormer + MLP on offline Qwen tokens.")
    parser.add_argument("--feature_dir", default=PROJECT_ROOT / "output/features_qwen_vl")
    parser.add_argument(
        "--metadata", default=PROJECT_ROOT / "output/features_qwen_vl/metadata_labeled.jsonl"
    )
    parser.add_argument(
        "--annotation_file", default=PROJECT_ROOT / "input/dataset_3000/annotations.jsonl"
    )
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "output/qformer_style_encoder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_dim", type=int, default=3584)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_queries", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--style_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_dropout", type=float, default=0.02)
    parser.add_argument("--gaussian_noise", type=float, default=0.01)
    parser.add_argument("--token_mask_prob", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
