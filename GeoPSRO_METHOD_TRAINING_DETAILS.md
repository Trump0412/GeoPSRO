# GeoPSRO Method and Training Details

Last updated: 2026-07-07

This document records the current method, training objectives, data plan, losses,
prompt format, and pre-experiment reasonableness checks for GeoPSRO-4D.

## 1. Current Method Summary

GeoPSRO-4D is a geometry-augmented post-training pipeline for dynamic 4D spatial
reasoning. The model combines:

- Qwen3-VL-2B-Instruct as the main multimodal language backbone for the first
  experiment.
- VGGT-1B as a frozen RGB-to-geometry branch.
- A trainable geometry adapter stack:
  GeoTokenizer, GeoResampler, GeoProjector, and GeometryGate.
- Three training stages:
  Stage 1 VGGT geometry alignment, Stage 2 geometry-augmented SFT, and Stage 3
  PSRO/GSPO reinforcement fine-tuning.

The core claim is not trace verification. The claim is that explicit geometry
tokens plus process-oriented RFT improve multi-frame and video spatial reasoning.

## 2. Prompt Format

The Stage 3 prompt has been restored to the old GeoBridge-compatible format:

```text
<think>
Spatial Observation: write one concise sentence describing the relevant visual-spatial evidence.
Spatial Transition: write one concise sentence describing the key spatial change, state continuity, or multi-frame relation.
Answer Derivation: write one concise sentence explaining how the previous two parts determine the final answer.
</think>
<answer>
Write only the final answer. For multiple-choice questions, write only the option letter.
</answer>
```

The previous lightweight GeoPSRO prompt was:

```text
Observation:
Transition:
Derivation:
Answer:
```

The restored prompt is preferred for the main experiment because it cleanly
separates reasoning from final answer extraction. The reward can score the
presence of `<think>`, `<answer>`, `Spatial Observation`, `Spatial Transition`,
and `Answer Derivation` independently. It also prevents the answer parser from
accidentally reading intermediate reasoning as the final answer.

The plain prompt is still parseable as a fallback for old smoke tests or legacy
data, but it is not the formal Stage 3 training prompt.

## 3. Model Inputs and Geometry Tokens

For each sample, the RGB/VLM branch receives sampled frames or images. The VGGT
branch receives the same visual sequence and produces cached geometry features:

- camera intrinsics and extrinsics
- relative camera pose features
- depth maps
- point maps
- track or motion cues, when available
- visibility and confidence statistics

These outputs are converted into compact geometry tokens:

- camera tokens
- frame-level depth/point/confidence tokens
- track-tube tokens

The current default is:

```text
num_frames: 8 for first real runs, 16 for formal scaled runs
num_geo_tokens: 64
VGGT: frozen
geometry adapter: trainable
geometry gate init: 0.2
```

## 4. Stage 0: VGGT Cache Preparation

Stage 0 is not a learning stage.

Task:

- Sample frames from each image sequence or video.
- Run VGGT once.
- Save reusable cached geometry features.

Cache policy:

- Do not cache full-resolution VGGT depth maps or point maps by default.
- Use compact pooled cache only:
  `cache_profile=compact_pool16_float16`.
- Cache only data that will be consumed by the next pilot or formal run.
- Prefer pilot caches first, then expand only after cache hit rate, disk usage,
  and dataloader speed are acceptable.
- Avoid blindly caching all 234K SPAR, 64K LLaVA-Hound, or 50K Stage3 samples
  before the training/eval subset is finalized.

Data:

- SPAR / SPAR-7M-RGBD or SPAR-234K subset for Stage 1 and Stage 2.
- LLaVA-Hound or compatible spatial/video instruction data for Stage 2.
- 4DThinker / s3_50K media plus 4DThinker annotations for Stage 3.
- SpatialLadder multi-image and video samples for Stage 3.
- DSR-Bench, VSI-Bench, ReVSI, MMSI, and related evaluation data as eval-only.

Checks:

- cache hit rate should be above 98 percent for train samples
- frame count should match the configured sampler
- no ground-truth depth, pose, trajectory, or benchmark metadata is used as
  inference input
- compact cache file size should remain small enough that dataloader IO is not
  the bottleneck

## 5. Stage 1: VGGT Geometry Alignment

Goal:

Make frozen VGGT geometry features readable by the Qwen3-VL token space.

Trainable modules:

- GeoTokenizer
- GeoResampler
- GeoProjector
- GeometryGate

Frozen modules:

- VGGT
- Qwen3-VL language model
- Qwen3-VL visual encoder

Data:

- Primary: SPAR-7M-RGBD / SPAR-234K geometry-heavy samples.
- Use spatial QA, relation, depth, viewpoint, and layout examples.

Loss:

```text
L_stage1 = masked causal language modeling cross entropy
```

Only answer/assistant tokens should contribute to the loss. The model sees
RGB/text plus inserted geometry tokens, but only the geometry adapter is updated.

Default config:

```text
global_batch_size: 64
num_frames: 8
num_geo_tokens: 64
lr_geo_adapter: 1e-4
warmup_ratio: 0.03
scheduler: cosine
epochs: 1
bf16: true
```

Metrics to log:

- train loss
- validation loss
- geometry gate value
- normal-vs-zero geometry gap
- normal-vs-shuffle geometry gap
- cache missing rate

Reasonableness criteria before Stage 2:

- loss decreases smoothly during the first few hundred steps
- geometry gate does not collapse to 0
- normal geometry performs better than zero or shuffled geometry on a small eval
- no severe cache fallback to zero geometry

## 6. Stage 2: Geometry-Augmented SFT

Goal:

Teach the model to solve spatial QA and video/multi-image reasoning tasks using
RGB tokens plus VGGT geometry tokens.

Trainable modules:

- Qwen3-VL LoRA adapters
- GeoProjector / GeoResampler / GeometryGate
- optional small geometry adapter layers

Frozen modules:

- VGGT
- base Qwen3-VL weights outside LoRA for the first main run

Data mixture:

```text
SPAR / SPAR-RGBD geometry data: 70 percent
LLaVA-Hound or compatible video/spatial instruction data: 30 percent
```

Loss:

```text
L_stage2 = masked causal language modeling cross entropy
```

Only target answer/assistant tokens should be supervised. Prompts and visual
context should be masked out.

Geometry regularization by data policy:

```text
geometry_on_ratio: 0.7
geometry_drop_ratio: 0.3
```

This prevents the model from becoming brittle when geometry is noisy or missing.
The drop/zero branch uses a small learned null geometry prefix rather than
literal all-zero embeddings. It does not read VGGT cache or carry sample geometry,
but it avoids unstable Qwen3-VL LoRA gradients observed with all-zero prefixes.

Default config:

```text
use_lora: true
lora_rank: 64
lora_alpha: 128
lr_lora: 1e-5
lr_projector: 5e-5
lr_geo_adapter: 5e-5
global_batch_size: 64
max_seq_len: 4096
num_frames: 8
warmup_ratio: 0.03
scheduler: cosine
epochs: 1
```

Recommended GPU batch start:

```text
2 A100-80G: micro batch 1, grad accumulation 32, global batch 64
6 A100-80G: micro batch 1, grad accumulation 10 or 11, global batch about 60 to 66
```

Metrics to log:

- train loss
- validation loss
- answer accuracy proxy on held-out spatial QA
- geometry_on observed ratio
- geometry gate value
- normal / zero / shuffle / depth-only ablation gap

Reasonableness criteria before Stage 3:

- SFT loss decreases and does not diverge after geometry is enabled
- normal geometry is better than zero or shuffled geometry
- depth-only and depth-camera ablations give interpretable intermediate results
- LoRA training is stable before considering full fine-tuning

## 7. Stage 3: PSRO/GSPO Reinforcement Fine-Tuning

Goal:

Refine the Stage 2 checkpoint for structured dynamic spatial reasoning and
answer calibration.

Initial model:

```text
outputs/stage2_vggt_sft
```

Reference model:

```text
outputs/stage2_vggt_sft
```

Data mixture:

```text
4DThinker / s3_50K media + 4DThinker annotations: 70 percent
SpatialLadder multi-image/video samples: 30 percent
```

Important filtering:

- remove single-frame and single-image SpatialLadder samples
- do not preserve SpatialLadder's original SFT/train split for Stage 3
- use all qualified multi-image/video samples for Stage 3 train unless a
  scene-disjoint validation split is explicitly created

Prompt:

Use the restored GeoBridge-compatible `<think>` and `<answer>` format from
Section 2.

Reward:

```text
R_total = R_answer + 0.5 * R_format + 0.05 * R_words
```

Components:

```text
R_answer:
  1 if final answer is correct, else 0.
  Supports multiple choice letters, choice text, yes/no, direction aliases, and
  numeric tolerance where applicable.

R_format:
  score in [0, 1].
  +0.20 for single <think> block
  +0.20 for single <answer> block
  +0.20 for Spatial Observation
  +0.20 for Spatial Transition
  +0.20 for Answer Derivation

R_words:
  score in [0, 1].
  Answer-gated. If R_answer < 1, R_words = 0.
  Uses observation, transition, derivation, spatial relation, and evidence
  geometry vocabularies.
```

Anti-gaming rules:

- overlong answers cap format score
- overlong reasoning caps format score
- multiple conflicting options cap format score
- word-list-like outputs receive zero or reduced word reward
- repeated keyword stuffing does not increase R_words

GSPO/RFT default:

```text
algorithm: gspo
group_size: 8
temperature: 0.7
top_p: 0.9
max_new_tokens: 512
kl_coef: 0.02
reward_normalization: group
```

Recommended batch start:

```text
2 A100-80G:
  train_batch_size: 16 prompts/update
  ppo_mini_batch_size: 8
  ppo_micro_batch_size_per_gpu: 1
  group_size: 8

6 A100-80G:
  train_batch_size: 48 prompts/update
  ppo_mini_batch_size: 24
  ppo_micro_batch_size_per_gpu: 1
  group_size: 8
```

Scale-up after stability:

```text
6 A100-80G:
  train_batch_size: 72 prompts/update
  keep group_size at 8 first
```

Loss and optimization terms:

```text
L_stage3 = GSPO policy loss
         + KL regularization to the Stage 2 reference model
         + optional entropy/clip terms from the VERL implementation
```

Metrics to log:

- reward_total
- R_answer
- R_format
- R_words
- format_pass_rate
- answer_acc_proxy
- word_score_mean
- word_score_nonzero_rate
- KL
- policy loss
- entropy
- response length
- rollout tokens/sec
- old_log_prob time and actor update time

Reasonableness criteria:

- R_format should rise quickly and stabilize near 1
- R_answer should improve more slowly and is the main success signal
- R_words should remain a small auxiliary signal, not dominate training
- KL should not collapse or explode
- response length should not grow without answer gains
- eval should improve or at least remain stable on DSR/VSI-style held-out tasks

## 8. Evaluation and Ablations

Main eval:

- DSR-Bench
- VSI-Bench
- ReVSI
- MMSI
- held-out SpatialLadder multi-image/video validation
- held-out 4DThinker/4DRL validation if available

Geometry ablations:

```text
normal:
  full geometry tokens

zero:
  learned null geometry prefix, no sample-specific VGGT geometry

shuffle:
  geometry tokens are loaded from another sample

depth_only:
  only depth-like features are used

depth_camera:
  depth plus camera tokens, no track tokens

full:
  camera, depth, point, and track tokens
```

Reward ablations:

```text
A. R_answer only
B. R_answer + 0.5 * R_format
C. R_answer + 0.5 * R_format + 0.05 * R_words
```

The main result should use C. A and B are required to show whether the word
prior helps or merely changes style.

## 9. Pre-Experiment Reasonableness Audit

The current design is reasonable for a first serious experiment because:

- Stage 1 isolates geometry-token alignment and keeps the LLM frozen.
- Stage 2 uses LoRA, reducing memory and avoiding premature full-model drift.
- Stage 2 includes geometry dropout, so the model cannot rely blindly on VGGT.
- Stage 3 starts from the Stage 2 checkpoint and uses a Stage 2 reference model.
- Stage 3 reward is answer-dominant, with format as strong structure reward and
  word reward as a very weak style prior.
- The restored prompt and reward match the previous GeoBridge Stage 3 design.

Main risks:

- If Stage 1 does not create a normal-vs-zero gap, Stage 2/3 geometry claims are
  weak.
- If Stage 2 LoRA underfits, full or partial fine-tuning may be needed as an
  ablation.
- If R_format saturates but R_answer does not improve, the RL stage is only
  learning formatting.
- If R_words rises without R_answer gains, the vocabulary reward should be
  removed from the main run.
- If SpatialLadder filtering leaves too few true video/multi-image samples, the
  Stage 3 mix should be rebalanced toward 4DThinker.

Required go/no-go checks before the first long run:

```text
1. VGGT cache hit rate >= 98 percent.
2. SpatialLadder single-image/single-frame samples removed.
3. Stage 1 smoke and short real run pass.
4. Stage 2 smoke and short real run pass.
5. Stage 3 reward offline check on 100 to 300 samples prints sensible breakdown.
6. Stage 3 50-step GSPO smoke logs reward_total, R_answer, R_format, R_words, KL.
7. Normal geometry beats zero/shuffle on at least one held-out probe.
```

Current implementation validation, 2026-07-07:

- Qwen backbone path:
  `/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct`.
- Runtime environment:
  `/mnt/guojh/lq/new/conda/envs/geobridge-verl`.
- Stage 2 entrypoint is active:
  `scripts/run_stage2_sft.sh`.
- Stage 2 loads Stage 1 geometry adapter from
  `outputs/stage1_qwen2b_smoke_50/geo_adapter.pt` by default.
- Stage 2 uses compact VGGT pilot cache by default:
  `cache/vggt_pilot/spar_1k` and `cache/vggt_pilot/llava_hound_1k`.
- Available pilot cache coverage is 1024 SPAR samples plus 1024 LLaVA-Hound
  samples. This is enough for smoke and short pilot runs, not a full epoch over
  the 297806-sample manifest.
- The all-zero geometry dropout prefix was replaced by a learned null geometry
  prefix. A diagnostic zero/null sample now has finite loss and `bad_count=0`
  non-finite gradients.
- Stage 2 bf16 smoke passed after the null-prefix change.
- Batch sweep on GPU6 with bf16:
  batch4, batch8, and batch16 passed; batch32 OOMed. Because sequence length can
  vary, batch8 is the conservative default for immediate pilots, while batch16 is
  usable for shorter or length-capped runs.
- Stage 2 batch8 100-step pilot passed with zero skipped steps:
  output `outputs/stage2_sft_pilot_b8_100`, mean loss 2.1108, median loss 2.1342,
  max sequence length 476, observed geometry-on ratio 0.69625, and gate drift
  from 0.199962 to 0.199986.

## 10. First Run Recommendation

Recommended first experimental sequence:

```text
Stage 0:
  Build VGGT caches with num_frames=8.

Stage 1:
  2 GPUs if available.
  global_batch_size=64.
  Run 500-step check, then 3000 to 5000 steps if stable.

Stage 2:
  Start with LoRA.
  Start with bf16 batch_size=8 on one 80GB GPU.
  Use batch_size=16 only after checking max sequence length and free memory.
  Run 500-step check, then 1 epoch.

Stage 3:
  Start with group_size=8.
  2-GPU smoke: train_batch_size=16 prompts/update.
  6-GPU formal: train_batch_size=48 prompts/update.
  Save every 200 steps and run eval every 200 to 500 steps.
```

Decision after first eval:

- Continue if R_answer and held-out accuracy improve.
- Reduce format weight if format saturates but answer does not move.
- Remove R_words if keyword style increases without answer gains.
- Increase frames from 8 to 16 only after the 8-frame pipeline is stable.
