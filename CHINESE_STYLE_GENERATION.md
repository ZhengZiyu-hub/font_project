# Chinese Style Generation

## Project Direction

The project now targets clean Chinese style text generation:

```text
reference_patch + target_text -> clean stylized Chinese text image
```

The output should be a standalone image on a clean background, not an edit
merged back into an existing scene.

## Difference From Calligrapher

Original Calligrapher is an inpainting/editing system:

```text
reference_patch + masked_image + mask + target_text -> edited image
```

This project uses Calligrapher components for style conditioning but changes the
main decoder task:

```text
reference_patch + target_text + noise -> target clean text image
```

FLUX.1-dev is the default generation backbone:

```text
/data/zhengziyu/models/FLUX.1-dev
```

FLUX-Fill remains optional for future inpainting experiments.

## Reused Components

The repository links Calligrapher source code through:

```text
third_party/Calligrapher -> /data/zhengziyu/Calligrapher
```

Large model weights stay outside the repo:

- `/data/zhengziyu/models/FLUX.1-dev`
- `/data/zhengziyu/Calligrapher/pretrained/siglip-so400m-patch14-384`
- `/data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin`
- `/data/pretrained_models` for any future shared local checkpoints

## Style Injection

The generation pipeline loads Calligrapher's SigLIP image encoder, QFormer/MLP
projection weights, and IP-Adapter-style attention processors. At runtime it
prints:

- `style injection enabled`
- style token shape
- loaded style checkpoint path

Current implementation uses adapter token injection. Exact paper-style K/V
replacement is still marked as a TODO in `src/models/style_injection.py`.

## Data Format

Generation annotations should use:

```json
{
  "target_image": "images/000043.png",
  "reference": "references/000043.png",
  "target_text": "生日快乐",
  "prompt": "The Chinese text is \"生日快乐\"."
}
```

The training dataset reader also accepts legacy-compatible fields:

- reference: `reference`, `reference_patch`, `ref_image`, `text_patch`, `style_patch`
- target image: `target_image`, `image`, `image_path`
- target text: `target_text`, `text`, `caption`
- prompt: `prompt`, `instruction`

## Build Generation Dataset

By default, the builder now creates:

- `images/`: original dataset images with background, copied from `input/dataset_3000/images`
- `references/`: plain black-on-white target text images with no style

```bash
python scripts/build_clean_generation_dataset.py \
  --annotation-file input/dataset_3000/annotations.jsonl \
  --data-root input/dataset_3000 \
  --output-root input/generation_dataset \
  --target-mode original \
  --reference-mode plain-text
```

For clean-crop experiments, add `--target-mode clean`; that path uses the
existing mask to extract the text region, crop by bbox, and place it on a white
or transparent background.

## Run Generation

```bash
python infer_calligrapher.py \
  --mode generation \
  --backend flux \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --reference input/dataset_3000/text_patches/000043.png \
  --text "生日快乐" \
  --output output/generation/result_000043.png \
  --steps 30 \
  --guidance-scale 3.5
```

Generation mode does not require `--image` or `--mask`, and it does not silently
fall back to smoke output.

## Dry-run Training

```bash
python train_calligrapher_decoder.py \
  --mode generation \
  --annotation-file input/generation_dataset/annotations.jsonl \
  --data-root input/generation_dataset \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --dry-run
```

Dry-run prints loaded model paths, trainable/frozen parameter counts, one batch
target tensor shape, and style token shape.

## Docker / Environment

No Dockerfile or compose file exists in this project. Reuse the active container
or existing environment and install:

```bash
pip install -r requirements.txt
```

The current default shell Python in this workspace does not provide
`torch`, `diffusers`, `transformers`, or `PIL`, so real FLUX inference and
dry-run training must be executed in the project GPU environment after
installing dependencies.

## Current Status

Implemented:

- `CalligrapherGenerationPipeline` generation-first API
- FLUX.1-dev default backbone path
- Calligrapher SigLIP/QFormer/MLP/attention adapter loading
- generation CLI without image/mask
- clean generation dataset builder
- generation training script with `--dry-run`
- optional experimental inpainting mode

Not yet proven in this shell:

- Real FLUX output image
- Runtime dry-run log

Reason: the visible Python environment lacks the required ML/image packages.
