# Qwen-VL Font Feature Retrieval

Extract visual features from character crop images with Qwen2.5-VL, build a
normalized feature bank, and retrieve visually similar crops.

## Pipeline

1. Extract pooled and token-level Qwen2.5-VL features.
2. Combine pooled features into a normalized NumPy feature bank.
3. Run cosine-similarity retrieval and generate JSONL results plus an HTML
   gallery.

## Setup

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/ZhengZiyu-hub/font_project.git
cd font_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place crop images under `input/crop/`. Images may also be stored in nested
directories.

## Usage

Extract features:

```bash
python src/extract_qwen_vl_feature.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0
```

`--model_path` can also point to a local Qwen2.5-VL checkpoint.

Build the pooled feature bank:

```bash
python src/build_feature_bank.py
```

Retrieve similar crops and create a gallery:

```bash
python src/retrieve_similar_crops.py --topk 10 --max_queries 100
```

The generated files are written under `output/`. The retrieval gallery is
available at `output/retrieval/retrieval_gallery.html`.

All paths and runtime options can be overridden with command-line arguments.
Run any script with `--help` for details.

## Data

Input images, model checkpoints, and generated outputs are intentionally not
tracked by Git because they can be large and may have separate usage rights.

