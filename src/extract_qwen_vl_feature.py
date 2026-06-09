from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "input" / "crop"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "features_qwen_vl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image_dir does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def move_to_device(inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


class QwenVLFeatureExtractor:
    """Extract Qwen2.5-VL visual token and pooled features for crop images."""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is False: {device}")

        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        if self.device.type == "cpu":
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _build_inputs(self, image: Image.Image) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract visual font style features from this Chinese character crop."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        return move_to_device(inputs, self.device)

    @torch.inference_mode()
    def extract(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        image = load_image(image_path)
        inputs = self._build_inputs(image)

        tokens = self._extract_from_visual_encoder(inputs)
        if tokens is None:
            tokens = self._extract_from_lm_hidden_states(inputs)

        tokens = l2_normalize(tokens)
        pooled = l2_normalize(tokens.mean(dim=0))
        return pooled.cpu().numpy().astype(np.float32), tokens.cpu().numpy().astype(np.float32)

    def _extract_from_visual_encoder(self, inputs: dict) -> torch.Tensor | None:
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None or not hasattr(self.model, "visual"):
            return None
        try:
            try:
                visual_tokens = self.model.visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
            except TypeError:
                visual_tokens = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            if isinstance(visual_tokens, tuple):
                visual_tokens = visual_tokens[0]
            if visual_tokens.ndim == 3:
                visual_tokens = visual_tokens[0]
            if visual_tokens.ndim != 2:
                return None
            return visual_tokens.float()
        except Exception as exc:
            print(f"[WARN] direct visual encoder extraction failed, falling back to hidden states: {exc}")
            return None

    def _extract_from_lm_hidden_states(self, inputs: dict) -> torch.Tensor:
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Qwen2.5-VL did not return hidden_states.")
        last_hidden = hidden_states[-1][0].float()
        input_ids = inputs["input_ids"][0]
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_mask = input_ids == image_token_id
        if image_mask.any():
            return last_hidden[image_mask]

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            valid_mask = attention_mask[0].bool()
            return last_hidden[valid_mask]
        return last_hidden


def save_features(
    extractor: QwenVLFeatureExtractor,
    images: Iterable[Path],
    output_dir: str | Path,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}] extracting {image_path}")
        pooled, tokens = extractor.extract(image_path)
        stem = image_path.stem
        pooled_path = output_dir / f"{stem}_pooled.npy"
        token_path = output_dir / f"{stem}_tokens.npy"
        np.save(pooled_path, pooled)
        np.save(token_path, tokens)
        record = {
            "image_path": str(image_path),
            "pooled_feature_path": str(pooled_path),
            "token_feature_path": str(token_path),
            "feature_shape": {
                "pooled": list(pooled.shape),
                "tokens": list(tokens.shape),
            },
        }
        records.append(record)
        print(f"    pooled={pooled.shape} tokens={tokens.shape}")

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote metadata: {metadata_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-VL visual style features from crop images.")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No images found under {args.image_dir}")
    extractor = QwenVLFeatureExtractor(model_path=args.model_path, device=args.device)
    save_features(extractor, images, args.output_dir)


if __name__ == "__main__":
    main()
