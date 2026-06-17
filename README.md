# Font Project: Chinese Stylized Glyph Generation

## 1. Project Overview

本项目是一个中文艺术字 / 字形风格生成工程，核心任务是：

```text
reference image + target text -> stylized Chinese glyph image
```

最终希望得到的是 **无背景或透明背景的中文风格字形图**，而不是把文字嵌回某张原始场景图中。

### Task Definition

给定：

- `reference image`：参考图，可为无样式目标文字图、风格参考 patch、或后续扩展的多参考图。
- `target text`：目标中文文本，例如 `生日快乐`。

输出：

- `stylized Chinese glyph image`：包含目标中文文本的风格化字形图，推理阶段默认会做近白背景透明化后处理。

### Difference vs Calligrapher

原始 Calligrapher 更偏向 inpainting / image editing：

```text
reference_patch + input_image + mask + target_text -> edited image
```

本项目主线是 standalone glyph generation：

```text
reference image + target_text -> transparent stylized glyph image
```

因此：

- 主路径不依赖 `input_image + mask`。
- 主路径优先使用 `/data/zhengziyu/models/FLUX.1-dev`。
- FLUX-Fill 仅作为未来 inpainting 实验路径保留。
- Qwen-VL retrieval / feature bank 已废弃为 legacy，不参与主 pipeline。

## 2. System Architecture

整理后的工程结构：

```text
font_project/
  configs/                  # 实验配置，占位；当前 CLI 参数为主
  data_engine/              # 数据构建与预处理
    build_clean_generation_dataset.py
  datasets/                 # 数据集入口，使用软链接指向已有数据
    dataset_3000 -> ../input/dataset_3000
    generation_dataset_smoke -> ../input/generation_dataset_smoke
  models/                   # 模型定义
    encoders/
      qformer_style_encoder.py
    fusion/
      style_injection.py
    diffusion/
      flux_style_adapter_placeholder.py
  training/                 # 训练入口
    train_calligrapher_decoder.py
  inference/                # 推理入口
    infer_calligrapher.py
  scripts/                  # 兼容脚本 / shell 入口
    build_clean_generation_dataset.py
  src/                      # 核心 pipeline 与兼容层
    pipelines/
      calligrapher_pipeline.py
    legacy/                 # deprecated retrieval / feature-bank 代码
  third_party/
    Calligrapher -> /data/zhengziyu/Calligrapher
  outputs -> output         # 输出目录兼容链接
  pretrained/               # 只记录外部权重路径，不存模型
  README.md
```

兼容入口仍保留：

```text
infer_calligrapher.py -> inference/infer_calligrapher.py
train_calligrapher_decoder.py -> training/train_calligrapher_decoder.py
scripts/build_clean_generation_dataset.py -> data_engine/build_clean_generation_dataset.py
```

### Data Engine

负责把已有 `dataset_3000` 转成 generation 训练格式。

入口：

```text
data_engine/build_clean_generation_dataset.py
```

输入：

```text
datasets/dataset_3000/annotations.jsonl
datasets/dataset_3000/images/
datasets/dataset_3000/masks/
datasets/dataset_3000/text_patches/
```

输出：

```text
datasets/generation_dataset/
  annotations.jsonl
  images/
  references/
```

### Training Pipeline

入口：

```text
training/train_calligrapher_decoder.py
```

训练目标：

```text
reference + target_text + noise -> target_image
```

冻结：

- FLUX transformer backbone
- VAE
- text encoders
- SigLIP image encoder

训练：

- style projection MLP
- QFormer projection
- attention adapter / style injection processor

### Inference Pipeline

入口：

```text
inference/infer_calligrapher.py
```

推理数据流：

```text
reference image
+ target text
-> style encoder + text encoder
-> diffusion generator
-> background removal
-> transparent glyph PNG
```

## 3. Dataset Construction

### dataset_3000

原始数据位于：

```text
datasets/dataset_3000/
  annotations.jsonl
  images/
  masks/
  text_patches/
```

原始 annotation 示例：

```json
{
  "text": "姬松茸",
  "image": "images/000043.png",
  "mask": "masks/000043.png",
  "text_patch": "text_patches/000043.png",
  "bbox": [100, 24, 316, 103]
}
```

字段说明：

- `text`：目标中文文本。
- `image`：原始艺术字图。
- `mask`：文字区域 mask。
- `text_patch`：原始 stylized text patch。
- `bbox`：文字区域框。

### generation_dataset

推荐构建无背景训练数据：

```bash
python data_engine/build_clean_generation_dataset.py \
  --annotation-file datasets/dataset_3000/annotations.jsonl \
  --data-root datasets/dataset_3000 \
  --output-root datasets/generation_dataset \
  --target-mode clean \
  --reference-mode plain-text \
  --size 512 \
  --background transparent
```

调试 / smoke 数据可以保留原图背景：

```bash
python data_engine/build_clean_generation_dataset.py \
  --annotation-file datasets/dataset_3000/annotations.jsonl \
  --data-root datasets/dataset_3000 \
  --output-root datasets/generation_dataset_smoke \
  --target-mode original \
  --reference-mode plain-text
```

### Annotation Format

生成后的 annotation 示例：

```json
{
  "target_image": "images/000043.png",
  "reference": "references/000043.png",
  "target_text": "姬松茸",
  "prompt": "The Chinese text is \"姬松茸\".",
  "source_image": "images/000043.png",
  "source_mask": "masks/000043.png",
  "source_reference": "text_patches/000043.png",
  "bbox": [100, 24, 316, 103],
  "target_mode": "clean",
  "reference_mode": "plain-text"
}
```

训练 dataloader 兼容字段：

- reference: `reference`, `reference_patch`, `ref_image`, `text_patch`, `style_patch`
- target image: `target_image`, `image`, `image_path`
- text: `target_text`, `text`, `caption`
- prompt: `prompt`, `instruction`

## 4. Model Architecture

### Style Encoder

主路径复用 Calligrapher 的 style encoder 组件：

```text
reference image
-> SigLIP vision encoder
-> QFormer projection
-> MLP projection
-> style tokens
```

相关文件：

```text
models/fusion/style_injection.py
models/encoders/qformer_style_encoder.py
src/pipelines/calligrapher_pipeline.py
```

当前 style token shape：

```text
(B, 128, 4096)
```

### Content Encoder

主 diffusion pipeline 使用 FLUX 自带文本编码器：

- CLIP text encoder
- T5 text encoder

Qwen / Qwen-VL 相关代码目前为 legacy，主要用于早期字体检索与特征实验，不参与主 generation pipeline。

### Diffusion Backbone

主干：

```text
/data/zhengziyu/models/FLUX.1-dev
```

代码加载路径：

```text
src/pipelines/calligrapher_pipeline.py
```

本项目没有修改 FLUX backbone 的核心接口，只在 pipeline 调用 transformer 时额外传入 `image_emb` style tokens。

### Style Injection

当前实现：

```text
style tokens -> IP-Adapter-style attention processor -> FLUX attention blocks
```

复用：

```text
/data/zhengziyu/Calligrapher/models/attention_processor.py
/data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin
```

运行时日志：

```text
style injection enabled
style token shape: (1, 128, 4096)
loaded style checkpoint: /data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin
```

后续可扩展为更严格的 attention K/V replacement。

## 5. Training Pipeline

### Data Loading

入口：

```bash
python training/train_calligrapher_decoder.py --help
```

默认读取：

```text
datasets/generation_dataset/annotations.jsonl
datasets/generation_dataset/images/
datasets/generation_dataset/references/
```

每个 batch 包含：

```text
reference image
target_text
prompt
target_image
```

### Forward Pass

训练前向过程：

```text
1. reference image -> SigLIP -> QFormer/MLP -> style tokens
2. target_text -> FLUX text encoders -> prompt embeddings
3. target_image -> VAE -> target latents
4. target latents + noise -> noisy latents
5. noisy latents + prompt embeddings + style tokens -> FLUX transformer
6. predict velocity / noise residual
```

### Loss Functions

当前已实现：

- diffusion flow-matching MSE loss

代码位置：

```text
training/train_calligrapher_decoder.py
```

后续建议扩展：

- style consistency loss：约束生成图与 reference 的风格编码距离。
- content consistency loss：约束 OCR / glyph recognizer 能读出 `target_text`。
- mask / alpha loss：针对透明背景输出约束 glyph 区域。

### Self-distillation Pipeline

可扩展流程：

```text
teacher generator / original target image
-> generated pseudo target
-> student adapter training
```

建议用途：

- 用 FLUX 或已有 Calligrapher adapter 生成 pseudo pairs。
- 对 style encoder / adapter 做自蒸馏。
- 用 content consistency loss 过滤错误文字。

当前仓库已经具备 adapter 训练入口，但 self-distillation 采样与过滤脚本尚未独立拆出。

### Frozen vs Trainable Components

冻结：

```text
FLUX transformer backbone
VAE
CLIP/T5 text encoders
SigLIP image encoder
```

训练：

```text
Calligrapher image projection MLP
Calligrapher QFormer projection
attention adapter processors
```

Dry-run：

```bash
python training/train_calligrapher_decoder.py \
  --mode generation \
  --annotation-file datasets/generation_dataset/annotations.jsonl \
  --data-root datasets/generation_dataset \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --dry-run \
  --batch-size 1 \
  --device cuda
```

正式训练：

```bash
python training/train_calligrapher_decoder.py \
  --mode generation \
  --annotation-file datasets/generation_dataset/annotations.jsonl \
  --data-root datasets/generation_dataset \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --output-dir outputs/calligrapher_generation \
  --batch-size 1 \
  --epochs 1 \
  --lr 1e-5 \
  --device cuda
```

## 6. Inference Pipeline

### Input Format

最小输入：

```text
reference image: PNG/JPG
target text: 中文字符串
```

示例：

```text
reference = datasets/generation_dataset_smoke/references/000043.png
target_text = "生日快乐"
```

### Step-by-step Generation

```text
1. 读取 reference image
2. 根据 target_text 构造 prompt:
   The Chinese text is "{target_text}".
3. reference image -> style encoder -> style tokens
4. target_text prompt -> FLUX text encoder
5. FLUX denoising with style injection
6. VAE decode
7. background removal
8. 保存 transparent PNG + metadata JSON
```

### Post-processing: Background Removal

`inference/infer_calligrapher.py` 默认对 generation 输出做近白背景透明化：

```text
RGB image -> RGBA image
near-white pixels alpha = 0
glyph pixels alpha = 255
```

可关闭：

```bash
--no-remove-background
```

可调阈值：

```bash
--background-threshold 245
```

### Output Format

输出：

```text
outputs/generation/result.png
outputs/generation/result.png.json
```

metadata 包含：

- mode
- backend
- style injection status
- style token shape
- prompt
- target text
- reference path
- external resource paths

### Generate Final Result

CPU smoke 级别生成：

```bash
python inference/infer_calligrapher.py \
  --mode generation \
  --backend flux \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --reference datasets/generation_dataset_smoke/references/000043.png \
  --text "生日快乐" \
  --output outputs/generation/cpu_result.png \
  --steps 4 \
  --guidance-scale 3.5 \
  --device cpu \
  --dtype float32 \
  --width 256 \
  --height 256
```

GPU 正式生成：

```bash
python inference/infer_calligrapher.py \
  --mode generation \
  --backend flux \
  --base-model-path /data/zhengziyu/models/FLUX.1-dev \
  --reference datasets/generation_dataset/references/000043.png \
  --text "生日快乐" \
  --output outputs/generation/result_000043.png \
  --steps 30 \
  --guidance-scale 3.5 \
  --device cuda \
  --dtype bfloat16 \
  --width 512 \
  --height 512
```

兼容旧入口：

```bash
python infer_calligrapher.py ...
```

## 7. Legacy Modules

以下模块已 deprecated，不参与主 generation pipeline：

```text
src/legacy/build_feature_bank.py
src/legacy/extract_qwen_vl_feature.py
src/legacy/retrieve_similar_crops.py
src/legacy/retrieve_qformer_style_features.py
src/legacy/compare_qwen_qformer_retrieval.py
src/legacy/evaluate_labeled_retrieval.py
src/legacy/extract_qformer_style_features.py
src/legacy/prepare_labeled_qwen_metadata.py
src/legacy/train_qformer_style_encoder.py
src/legacy/CALLIGRAPHER_DECODER.md
```

说明：

- retrieval is deprecated。
- feature bank is unused in main pipeline。
- Qwen-VL features are legacy retrieval assets。
- 旧路径 `src/*.py` 保留 wrapper，便于旧命令转发到 `src/legacy/*`。

## 8. External Dependencies

### Code

```text
third_party/Calligrapher -> /data/zhengziyu/Calligrapher
```

### Model Paths

FLUX generation backbone：

```text
/data/zhengziyu/models/FLUX.1-dev
```

SigLIP image encoder：

```text
/data/zhengziyu/Calligrapher/pretrained/siglip-so400m-patch14-384
```

Calligrapher adapter：

```text
/data/zhengziyu/Calligrapher/pretrained/Calligrapher/calligrapher.bin
```

Optional FLUX-Fill inpainting backbone：

```text
/data/zhengziyu/Calligrapher/pretrained/FLUX.1-Fill-dev
```

Shared local model pool：

```text
/data/pretrained_models
```

`pretrained/README.md` 仅记录路径，不存放权重。

## Current Verified Status

已验证：

- CPU generation 可生成真实 FLUX diffusion 输出。
- `style injection enabled`
- `style token shape: (1, 128, 4096)`
- `outputs -> output` 兼容链接可写入已有输出目录。

示例产物：

```text
outputs/generation/cpu_result_000043.png
outputs/generation/cpu_result_000043_steps4.png
```

后续建议：

- 使用 CUDA 环境跑 `512x512 + 30 steps`。
- 训练 adapter checkpoint 后重新推理对比。
- 增加 content recognizer / OCR consistency loss。
- 增加更稳健的 alpha matting 或 background removal。
