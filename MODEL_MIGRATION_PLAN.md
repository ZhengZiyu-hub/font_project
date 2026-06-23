# Model Migration Plan

## 1. Reference Repository Analysis

### Path Check

- Requested path `/data/zhengziyu/font_project/Calligrapher`: not found as a direct directory.
- Actual readable path: `/data/zhengziyu/font_project/third_party/Calligrapher`, a symlink to `/data/zhengziyu/Calligrapher`.
- Requested path `/data/zhengziyu/font_project/minit2i-jax`: not found as a direct directory.
- Actual readable path: `/data/zhengziyu/font_project/third_party/minit2i-jax`.

These repositories should be treated only as references. The target project should not import them as runtime modules.

### Calligrapher Files

| File | Responsibility | Category | External Dependencies |
| --- | --- | --- | --- |
| `models/calligrapher.py` | Wraps an image-conditioned generation pipeline. Builds a pretrained visual encoder, projection layers, attention processors, loads a checkpoint, and calls a diffusion pipeline. | encoder wrapper, projection setup, pipeline orchestration, inference | PyTorch, Transformers `AutoProcessor`, `SiglipVisionModel`, Diffusers-style pipeline, specific checkpoint file |
| `models/projection_models.py` | Converts pooled image embeddings into condition token sequences. `MLPProjModel` expands one embedding to N tokens; `QFormerProjModel` uses learned query tokens and multi-head attention over projected image features. | projection, lightweight style token adapter | PyTorch only |
| `models/attention_processor.py` | Adds extra image-condition K/V projections into existing attention layers. It computes attention from hidden-state queries to image-condition keys/values and adds the result back with a scale factor. | attention, condition injection | PyTorch, `torch.nn.functional.scaled_dot_product_attention`, Diffusers `RMSNorm`, assumes Diffusers attention module fields |
| `models/transformer_flux_inpainting.py` | Diffusers-compatible conditional transformer with dual-stream blocks, single-stream blocks, timestep/text embeddings, rotary position embeddings, and final projection to latent patches. | decoder / transformer denoiser | PyTorch, Diffusers model mixins, Diffusers attention, Diffusers normalization/embedding classes, Flux checkpoint conventions |
| `pipeline_calligrapher.py` | Full inpainting pipeline: prompt encoding, mask/latent packing, scheduler timesteps, denoising loop, VAE encode/decode. | pipeline, inference, scheduler orchestration | PyTorch, Diffusers, Transformers CLIP/T5, VAE, scheduler, image processors |
| `requirements.txt` | Pins PyTorch 2.5.0, Diffusers 0.33.0, Transformers 4.49.0, Gradio, OpenCV, etc. | environment | CUDA-oriented PyTorch and UI dependencies |
| `README.md` | Describes the full pretrained model, environment, demos, model downloads, path configuration, and usage. | documentation | Requires pretrained base model, SigLIP, project checkpoint, Hugging Face assets |
| `LICENSE` | Not found at `/data/zhengziyu/Calligrapher/LICENSE`. | license | Must clarify license before copying substantial code |

Key model ideas worth retaining:

- Style/reference image is encoded by a pretrained vision encoder in the original code, then projected into a small sequence of condition tokens.
- A simple MLP projection and a query-attention projection are both used, and their outputs are summed.
- Style condition is injected into decoder attention as extra key/value information.
- The full Diffusers pipeline is too coupled to pretrained Flux/VAE/mask logic for direct migration.

### minit2i-jax Files

| File | Responsibility | Category | External Dependencies |
| --- | --- | --- | --- |
| `models/base.py` | Defines initialization schemes and `PartialModel` for JAX sharded parameter shape/init. | model utility, distributed init | JAX, Flax, pjit |
| `models/torch_models.py` | Flax modules that mimic PyTorch layer behavior: linear, layer norm, RMSNorm, sequential, param wrapper. | model utility | JAX, Flax |
| `models/dit_blocks.py` | Patch embedding, final patch projection, 2D sin/cos position embedding, 1D/2D rotary embeddings, multimodal RoPE, SwiGLU MLP. | transformer blocks, positional embeddings, patch IO | JAX, Flax |
| `models/mmjit.py` | Main MM-JiT image/text transformer. Uses image patch tokens plus text tokens, optional text preamble blocks, multimodal attention, final unpatchify. | transformer denoiser / decoder | JAX, Flax, project JAX utilities |
| `models/t5_encoder.py` | JAX/Flax T5 encoder implementation and PyTorch weight loading helpers. | text encoder | JAX, Flax, Transformers-compatible T5 weights |
| `diffusion.py` | Flow-matching/DDPM wrapper: timestep schedules, velocity prediction objective, Euler/Heun/SDE samplers, CFG wrapper, sampling loop. | diffusion, training objective, sampler | JAX, Flax, pjit/DDP utilities |
| `train.py` | Distributed training/eval loop: data loading, frozen text encoder, pjit mesh, model init, checkpoint restore/save, sampling/evaluation. | training orchestration | JAX distributed runtime, Flax, ml_collections, WebDataset, evaluators, W&B/TensorBoard |
| `settings.py` | Local machine paths for datasets, checkpoints, eval assets, and logging. | path configuration | local filesystem, benchmark assets |
| `requirements.txt` | JAX/Flax/Optax/Orbax/WebDataset/Transformers/Torch dependencies. | environment | TPU/JAX stack |
| `README.md` | Documents direct-RGB text-to-image training recipe, model zoo, datasets, training/evaluation commands, checkpoint download. | documentation | Hugging Face checkpoints, TPU/JAX workflow |
| `LICENSE` | MIT License. | license | Compatible with permissive reuse if notices are preserved |

Key model ideas worth retaining:

- Pixel-space / direct RGB denoising avoids requiring a VAE in the first custom implementation.
- Image is patchified, processed by transformer blocks, then unpatchified back to pixels.
- Text/condition tokens and image tokens interact in multimodal transformer blocks.
- Flow matching with velocity prediction is a compact objective and sampler family.
- Most code is JAX/Flax and must be rewritten in PyTorch for the current project.

## 2. Transferable Components

| Source Repository | Source File | Original Component | Responsibility | Target File | New Component Name | Required Changes |
| --- | --- | --- | --- | --- | --- | --- |
| Calligrapher | `models/projection_models.py` | `MLPProjModel` | Expand one pooled visual embedding into N condition tokens. | `src/models/projection.py` | `TokenMLPProjection` | Keep design, rewrite names/configs, remove checkpoint assumptions, adapt dims to project config. |
| Calligrapher | `models/projection_models.py` | `QFormerProjModel` | Learned query tokens attend to projected image features to produce condition tokens. | `src/models/projection.py` or `src/models/style_encoder.py` | `QueryTokenProjection` | Keep PyTorch, simplify input contract to `[B, D]` or `[B, N, D]`, remove original fixed dims. |
| Calligrapher | `models/calligrapher.py` | `get_image_embeds` concept | Encode reference image, project embedding to condition tokens. | `src/models/style_encoder.py` | `StyleEncoder` | Do not migrate wrapper or checkpoint loading. Keep interface; initially use lightweight Conv encoder, later optional pretrained vision backbone. |
| Calligrapher | `models/attention_processor.py` | `FluxAttnProcessor` condition branch | Project condition tokens to K/V and add condition attention to hidden states. | `src/models/attention.py` | `ConditionCrossAttention` | Reimplement independent of Diffusers attention object; use `nn.MultiheadAttention` or `scaled_dot_product_attention`; expose scale. |
| Calligrapher | `models/transformer_flux_inpainting.py` | `FluxTransformerBlock` high-level pattern | Dual stream condition/image attention with MLP residual blocks. | `src/models/decoder.py` | `ConditionedTransformerBlock` | Do not copy Diffusers mixins or AdaLayerNorm classes directly; implement project-native PyTorch block. |
| Calligrapher | `models/transformer_flux_inpainting.py` | `FluxSingleTransformerBlock` high-level pattern | Single-stream transformer block after condition/image fusion. | `src/models/decoder.py` | `ImageTransformerBlock` | Reimplement as simple norm-attention-MLP block; remove Diffusers-specific processor APIs. |
| Calligrapher | `pipeline_calligrapher.py` | `_pack_latents`, `_unpack_latents` idea | Convert latent/image grids to patch token sequences and back. | `src/models/decoder.py` or `src/models/patch.py` | `patchify`, `unpatchify` | Only migrate concept; remove VAE/mask/pipeline coupling. |
| minit2i-jax | `models/dit_blocks.py` | `BottleneckPatchEmbed` | Convert image tensor to patch tokens. | `src/models/image_transformer.py` or `src/models/decoder.py` | `PatchEmbed` | Must rewrite JAX/Flax to PyTorch; use `[B, C, H, W]` layout. |
| minit2i-jax | `models/dit_blocks.py` | `FinalLayer` | Normalize tokens and project to pixel patches. | `src/models/decoder.py` | `PatchOutputProjection` | Must rewrite to PyTorch; keep zero-init option optional. |
| minit2i-jax | `models/dit_blocks.py` | `get_2d_sincos_pos_embed` family | Fixed 2D image patch positional embedding. | `src/models/position.py` or `src/models/decoder.py` | `build_2d_sincos_pos_embed` | Rewrite to PyTorch/NumPy; no checkpoint dependency. |
| minit2i-jax | `models/dit_blocks.py` | `TextRotaryEmbedding1D`, `VisionRotaryEmbeddingFast`, `MultiModalRotaryEmbeddingFast` | RoPE for text and image token attention. | `src/models/position.py` or `src/models/attention.py` | `TextRoPE`, `ImageRoPE`, `MultiModalRoPE` | Rewrite JAX to PyTorch; can defer until baseline block is stable. |
| minit2i-jax | `models/dit_blocks.py` | `SwiGLUMlp` | Gated MLP used in transformer blocks. | `src/models/attention.py` or `src/models/blocks.py` | `SwiGLUFeedForward` | Rewrite to PyTorch; straightforward. |
| minit2i-jax | `models/mmjit.py` | `MMJiTBlock` | Double-stream attention between image tokens and text tokens. | `src/models/decoder.py` | `DualStreamConditionBlock` | Must rewrite JAX/Flax to PyTorch; generalize text stream to content/style/text condition streams. |
| minit2i-jax | `models/mmjit.py` | `TextPreambleBlock` | Preprocess text tokens before joint attention. | `src/models/text_encoder.py` or `src/models/decoder.py` | `ConditionPreambleBlock` | Optional; rewrite to PyTorch; can process text/style/content tokens before fusion. |
| minit2i-jax | `models/mmjit.py` | `MMJiT` architecture | Patchify image, embed condition tokens, multimodal transformer, unpatchify. | `src/models/decoder.py` | `ImageTransformerDenoiser` | Must rewrite JAX/Flax to PyTorch; remove registry and preset checkpoint names. |
| minit2i-jax | `diffusion.py` | `LinearSchedule`, `LognormSchedule` | Sample training timesteps. | `src/models/diffusion.py` | `LinearTimestepSchedule`, `LognormalTimestepSchedule` | Rewrite to PyTorch; independent from JAX RNG. |
| minit2i-jax | `diffusion.py` | `SimDDPM.forward` | Flow-matching training objective with velocity target. | `src/models/diffusion.py` | `FlowMatchingObjective` | Rewrite to PyTorch; remove Flax module wrapper, DDP utilities, visualization side effects. |
| minit2i-jax | `diffusion.py` | `sample_one_step_euler`, `sample_one_step_heun`, `sample_one_step_sde` | Iterative sampling update rules. | `src/models/diffusion.py` | `EulerSampler`, `HeunSampler`, `SDESampler` | Rewrite to PyTorch; start with Euler only, add Heun/SDE later. |
| minit2i-jax | `models/t5_encoder.py` | T5 encoder design | Text token encoder and attention mask interface. | `src/models/text_encoder.py` | `TextConditionEncoder` | Do not port full JAX T5 initially; keep interface and optionally wrap Transformers PyTorch T5 later. |

## 3. Components Not to Transfer

### Calligrapher

- `models/calligrapher.py` as a whole should not transfer. It is a runtime wrapper around a Diffusers pipeline, SigLIP, checkpoint loading, path/device management, and generation calls. Only its style-embedding/projection idea should be reused.
- `pipeline_calligrapher.py` should not transfer as a full file. It is tightly coupled to Flux Fill, VAE latent packing, mask preprocessing, CLIP/T5 prompt encoding, scheduler calls, and inference UX. Only the patch/latent packing idea and denoising-loop shape contracts are useful.
- Gradio demos and CLI inference scripts should not transfer. They are UI or entry-point code, not reusable model internals.
- Path configuration such as `path_dict.json` should not transfer. It binds the original project to local checkpoint and output paths.
- Checkpoint loading logic for `calligrapher.bin` should not transfer now. The current model must initialize and run dummy forward without pretrained weights.
- Direct Diffusers model mixins from `transformer_flux_inpainting.py` should not transfer. They add a large API surface tied to Diffusers checkpoint compatibility.
- `LICENSE` was not found in the readable repository path. Substantial code copying should wait until license terms are clarified.

### minit2i-jax

- `train.py` should not transfer as code. It is distributed JAX training orchestration with pjit mesh setup, input pipeline, checkpointing, sampling, logging, and evaluation.
- `settings.py` should not transfer. It contains local paths for datasets, checkpoints, eval assets, and logging.
- `models/base.py` `PartialModel` should not transfer. It is specific to Flax/JAX sharded initialization.
- `models/torch_models.py` should not transfer directly. Despite the name, these are Flax modules emulating PyTorch initialization behavior; for this project, use native PyTorch layers.
- `models/t5_encoder.py` should not transfer directly. It is a large JAX/Flax T5 implementation and PyTorch-to-JAX weight conversion layer. Prefer a simple interface now and optional PyTorch Transformers wrapper later.
- Evaluators, external benchmark ports, WebDataset data pipeline, scripts, and checkpoint download/eval helpers should not transfer. They are outside current model-layer scope.
- Preset model registry names and checkpoint-specific constants should not transfer. They assume original architecture scales and text encoder identifiers.

## 4. Proposed Model Architecture

The target model should combine the style-token projection and condition-attention idea from Calligrapher with the direct-RGB patch transformer and flow-matching idea from minit2i-jax.

```text
content_image [B, 3, H, W]
    -> content encoder
    -> content tokens [B, N_content, D]
    -> content projection
    -> decoder condition stream

style_image [B, 3, H, W]
    -> style encoder
    -> style tokens [B, N_style, D]
    -> style projection
    -> decoder condition stream

optional text prompt
    -> text encoder interface
    -> text tokens [B, N_text, D]
    -> decoder condition stream

target_image [B, 3, H, W] + timestep t [B]
    -> add noise using diffusion schedule
    -> noisy image [B, 3, H, W]
    -> patch embedding
    -> image tokens [B, N_image, D]
    -> transformer denoiser / decoder
    -> predicted noise, velocity, or predicted image [B, 3, H, W]
```

Condition injection points:

- `content tokens`: inject into decoder blocks through cross-attention or a dual-stream block. They should preserve glyph layout and structure.
- `style tokens`: inject into decoder blocks through a separate cross-attention branch with a scale parameter, similar to image-condition K/V injection.
- `timestep embedding`: add to image tokens before transformer blocks, or use it to modulate normalization/MLP layers. The first implementation can use additive sinusoidal timestep embeddings; later versions can adopt gated/adaptive normalization.
- `text tokens`: keep an optional interface in `src/models/text_encoder.py`, but do not require it for content/style-only dummy forward. If enabled, concatenate text tokens with style/content condition tokens or give text its own condition branch.

Recommended division of source ideas:

- Style encoder: Calligrapher is more suitable. Its original style path explicitly encodes a reference image and projects it into condition tokens.
- Content encoder: neither project has a dedicated glyph content encoder. Use a project-native Conv/Patch encoder, borrowing minit2i-jax patch tokenization for layout tokens.
- Diffusion scheduler and timestep handling: minit2i-jax is more suitable. Its `diffusion.py` has compact flow-matching schedules and samplers.
- Transformer denoiser / decoder: minit2i-jax is more suitable for a standalone project because it is direct-RGB and not tied to a VAE. Calligrapher's transformer file is useful for condition injection and block layout ideas but is heavily Diffusers-coupled.
- Style condition injection: use Calligrapher's extra K/V attention idea, implemented as project-native PyTorch cross-attention.
- Content condition injection: inject as a structural condition stream, either through cross-attention after image self-attention or through a dual-stream block adapted from MM-JiT.
- Text condition interface: keep optional, initially as a shape-compatible placeholder or lightweight PyTorch/Transformers wrapper.

PyTorch rewrite requirements:

- All minit2i-jax model/diffusion components must be rewritten from JAX/Flax to PyTorch.
- Calligrapher projection modules are already PyTorch and can be migrated with light renaming/generalization.
- Calligrapher attention processor must be rewritten to remove Diffusers `Attention` object assumptions.
- Calligrapher transformer blocks should be reimplemented rather than copied, because they depend on Diffusers mixins and normalization classes.

## 5. Final src/models Structure

Suggested final structure after migration:

```text
src/models/
├── __init__.py
├── attention.py          # self-attention, cross-attention, condition K/V injection
├── blocks.py             # RMSNorm, SwiGLU, transformer blocks
├── content_encoder.py    # content image to structure tokens
├── decoder.py            # image transformer denoiser / decoder
├── diffusion.py          # timestep schedules, objective, samplers
├── font_model.py         # top-level composition
├── patch.py              # patchify, unpatchify, patch embeddings
├── position.py           # sin/cos position embeddings and optional RoPE
├── projection.py         # MLP and query-token projections
├── style_encoder.py      # style image to style tokens
└── text_encoder.py       # optional text condition interface
```

Current-stage implementations may keep fewer files, but this is the clean target structure for separating responsibilities.

## 6. Implementation Order

1. `projection.py`
   - Implement `TokenMLPProjection`.
   - Implement `QueryTokenProjection`.
   - Add shape tests with dummy `[B, D]` and `[B, N, D]`.

2. `style_encoder.py`
   - Keep a lightweight Conv encoder first.
   - Output `[B, N_style, D]`.
   - Add an optional future hook for a pretrained vision backbone without loading it by default.

3. `content_encoder.py`
   - Implement a Conv/Patch encoder for `content_image`.
   - Output `[B, N_content, D]`.
   - Preserve spatial token order.

4. `patch.py` and `position.py`
   - Add PyTorch `patchify`, `unpatchify`, `PatchEmbed`.
   - Add fixed 2D sin/cos positional embeddings.

5. `attention.py` and `blocks.py`
   - Implement project-native self-attention and condition cross-attention.
   - Add optional `scale` for style injection.
   - Add SwiGLU feed-forward block.

6. `decoder.py`
   - Implement patch image tokens from noisy image.
   - Add timestep embedding to image tokens.
   - Inject content tokens and style tokens in separate cross-attention branches.
   - Project tokens back to `[B, 3, H, W]`.

7. `diffusion.py`
   - Implement PyTorch linear/lognormal timestep schedules.
   - Implement velocity-prediction flow-matching objective.
   - Implement Euler sampler first.

8. `font_model.py`
   - Compose encoders, projections, decoder, and optional diffusion wrapper.
   - Provide dummy forward with content/style images.

9. Smoke tests
   - Keep `python -m compileall src`.
   - Add/keep a forward test that asserts `[B, 3, H, W]`.
   - Do not add dataset, dataloader, or trainer in this phase.

## 7. Risks and Dependencies

- JAX to PyTorch rewrite risk:
  - minit2i-jax uses NHWC tensors, Flax modules, JAX RNG, pjit sharding, and JAX einsum semantics. The target code should standardize on PyTorch NCHW for images and carefully check shape conversions.
- Pretrained weight compatibility:
  - Directly loading minit2i-jax checkpoints into PyTorch is not in scope.
  - Calligrapher projection and attention weights are tied to original hidden sizes and Diffusers transformer internals.
  - The first target should initialize from scratch and run dummy forward.
- Image size, latent size, and hidden dimension alignment:
  - Patch size must divide image size.
  - Content/style/text token dimensions must match decoder hidden dimension after projection.
  - Direct-RGB decoder outputs `[B, 3, H, W]`; a future latent decoder would need a VAE contract.
- Attention mask and token length alignment:
  - Text tokens may need masks.
  - Style/content tokens usually do not need masks but must have consistent batch size and hidden dimension.
  - Multimodal RoPE assumes square image token grids; this must be validated if adopted.
- Dependency risk:
  - Full Calligrapher pipeline depends on Diffusers, Transformers, SigLIP, VAE, Flux, and specific checkpoints.
  - minit2i-jax depends on JAX/Flax/Optax/Orbax/WebDataset and TPU-oriented distributed runtime.
  - The target implementation should keep core `src/models` PyTorch-only at first.
- License requirements:
  - minit2i-jax has an MIT License; migrated code or substantial derivatives should preserve license notices in documentation.
  - Calligrapher `LICENSE` was not found in the readable path. Do not copy substantial code verbatim until the license is confirmed. Use architectural ideas or small reimplementations.
