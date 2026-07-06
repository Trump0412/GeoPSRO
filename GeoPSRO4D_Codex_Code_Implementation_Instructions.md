# Codex Instructions: Implement GeoPSRO-4D Code

Implement the current GeoPSRO-4D experimental pipeline:

```text
Stage 1: VGGT Geometry Alignment
Stage 2: Geometry-Augmented Spatial SFT
Stage 3: PSRO-RFT
```

Do not implement trace verifier, evidence graph, or trace supervision. The RFT reasoning format is PSRO:

```text
Observation + Transition + Derivation + Answer
```

There is no separate PSRO cold-start SFT stage.

---

## 0. Expected Repository Structure

Create or adapt the repository to this structure:

```text
GeoPSRO4D/
  configs/
    vggt_cache.yaml
    stage1_vggt_align.yaml
    stage2_vggt_sft.yaml
    stage3_psro_rft.yaml
    eval_dsr.yaml
    ablation_geometry.yaml
    ablation_reward.yaml

  geopsro4d/
    data/
      __init__.py
      dataset_sft.py
      dataset_rft.py
      frame_sampler.py
      collator.py
      formatters.py

    geometry/
      __init__.py
      vggt_extractor.py
      vggt_cache.py
      geo_tokenizer.py
      geo_resampler.py
      geo_projector.py
      geometry_gate.py

    model/
      __init__.py
      qwen_vggt_wrapper.py
      input_builder.py
      lora_utils.py

    reward/
      __init__.py
      answer_parser.py
      psro_parser.py
      reward_fn.py
      process_rules.py

    train/
      train_stage1_align.py
      train_stage2_sft.py
      train_stage3_rft.py

    eval/
      eval_dsr.py
      eval_aux.py
      eval_geometry_ablation.py
      eval_reward_ablation.py
      error_analysis.py

    utils/
      io.py
      logging.py
      seed.py
      distributed.py
      metrics.py

  scripts/
    run_vggt_cache.sh
    run_stage1_align.sh
    run_stage2_sft.sh
    run_stage3_rft.sh
    run_eval_dsr.sh
    run_ablation_geometry.sh
    run_ablation_reward.sh

  results/
    baseline/
    stage1/
    stage2/
    stage3/
    ablations/
```

If the existing repository already has a different structure, preserve existing conventions where practical, but implement equivalent modules.

---

## 1. Dataset and Sample Schema

All dataset loaders should normalize examples to a shared schema.

### 1.1 SFT Sample Schema

```python
{
    "sample_id": str,
    "dataset": str,
    "media_paths": list[str],
    "question": str,
    "answer": str,
    "choices": list[str] | None,
    "task_type": str | None,
    "metadata": dict,
}
```

Datasets used in SFT:

```text
LLaVA-Hound-64K
SPAR-234K
```

### 1.2 RFT Sample Schema

```python
{
    "sample_id": str,
    "dataset": str,
    "media_paths": list[str],
    "question": str,
    "answer": str,
    "choices": list[str] | None,
    "task_type": str | None,
    "metadata": dict,
}
```

Datasets used in RFT:

```text
SpatialLadder multi-image 15K
4DThinker-4DRL 35K
```

No trace labels are required.

### 1.3 Evaluation Sample Schema

```python
{
    "sample_id": str,
    "dataset": str,
    "media_paths": list[str],
    "question": str,
    "answer": str,
    "choices": list[str] | None,
    "task_type": str | None,
    "metadata": dict,
}
```

Main evaluation:

```text
DSR-Bench
```

Auxiliary evaluation:

```text
VSI-Bench
Video-MME
SPAR validation
optional VLM4D / STI-Bench
```

---

## 2. Frame Sampling

Implement `frame_sampler.py`.

Required behavior:

```python
def sample_frames(media_paths, num_frames: int, mode: str = "uniform") -> list:
    """
    Return exactly num_frames visual inputs whenever possible.
    For multi-image inputs, preserve the original image order.
    For videos, use uniform sampling by default.
    """
```

Important rule:

```text
The frames sent to Qwen and the frames sent to VGGT must be identical.
```

Store sampled frame indices in every cache and evaluation log.

---

## 3. VGGT Cache Extraction

Implement:

```text
geopsro4d/geometry/vggt_extractor.py
geopsro4d/geometry/vggt_cache.py
```

### 3.1 Cache Interface

```python
class VGGTCache:
    def __init__(self, cache_root: str): ...
    def path_for(self, sample_id: str) -> str: ...
    def exists(self, sample_id: str) -> bool: ...
    def load(self, sample_id: str) -> dict: ...
    def save(self, sample_id: str, data: dict) -> None: ...
```

### 3.2 Cache Content

Each cache file should contain:

```python
{
    "sample_id": str,
    "frame_indices": list[int],
    "image_size": tuple[int, int],
    "camera_intrinsics": Tensor | None,
    "camera_extrinsics": Tensor | None,
    "depth": Tensor | None,
    "point_map": Tensor | None,
    "tracks": Tensor | None,
    "visibility": Tensor | None,
    "confidence": Tensor | None,
    "features": Tensor | None,
}
```

Use `.pt` or `.safetensors` depending on existing code conventions. `.pt` is acceptable for the first implementation.

### 3.3 Extraction Script

Create:

```text
scripts/run_vggt_cache.sh
```

It should run extraction for:

```text
SPAR-234K
LLaVA-Hound-64K
SpatialLadder multi-image
4DThinker-4DRL
DSR-Bench
```

Example command:

```bash
python -m geopsro4d.geometry.vggt_extractor \
  --dataset_json data/spar/train.jsonl \
  --cache_root cache/vggt/spar \
  --num_frames 8 \
  --model_name_or_path /path/to/vggt \
  --overwrite false
```

### 3.4 Safety Checks

Log and assert:

```text
sample_id
frame_indices
VGGT output shapes
number of NaNs
confidence statistics
cache path
```

If VGGT fails for a sample, mark the sample with `geometry_valid=False` and allow the collator to use zero geometry tokens.

---

## 4. Geometry Tokenizer

Implement:

```text
geopsro4d/geometry/geo_tokenizer.py
```

The tokenizer converts dense VGGT outputs into a variable-length set of raw geometry features.

### 4.1 Camera Tokens

For each frame, build one camera token from available camera data:

```python
camera_token_t = MLP([
    flattened_intrinsics_t,
    flattened_extrinsics_t,
    relative_pose_to_first_frame_t,
])
```

If camera data is unavailable, use zeros and a validity mask.

### 4.2 Frame Geometry Tokens

From depth and point map, compute patch statistics:

```python
patch_feature = [
    mean_depth,
    std_depth,
    mean_xyz,
    std_xyz,
    mean_confidence,
]
```

Use coarse pooling such as 8x8 or 16x16 patches. Keep the number of tokens manageable.

### 4.3 Track-Tube Tokens

From point tracks, select high-confidence tracks:

```python
track_feature = [
    xyz_start,
    xyz_end,
    xyz_mean,
    displacement,
    velocity,
    visibility_ratio,
    confidence_mean,
]
```

Default:

```text
top_k_tracks = 128
min_visibility = 0.5
min_confidence = 0.3
```

### 4.4 Output

```python
class GeoTokenizer(nn.Module):
    def forward(self, vggt_cache: dict) -> tuple[Tensor, Tensor]:
        """
        Returns:
            geo_features: FloatTensor [N_geo_raw, d_geo]
            geo_mask: BoolTensor [N_geo_raw]
        """
```

---

## 5. Geo Resampler, Projector, and Gate

Implement:

```text
geopsro4d/geometry/geo_resampler.py
geopsro4d/geometry/geo_projector.py
geopsro4d/geometry/geometry_gate.py
```

### 5.1 Geo Resampler

Compress raw geometry features into fixed-length geometry tokens.

Default:

```text
num_geo_tokens = 64
```

Use a Perceiver-style learned-query cross-attention resampler or a simpler attention pooling module.

Interface:

```python
class GeoResampler(nn.Module):
    def forward(self, geo_features: Tensor, geo_mask: Tensor) -> Tensor:
        """
        Args:
            geo_features: [B, N_geo_raw, d_geo]
            geo_mask: [B, N_geo_raw]
        Returns:
            geo_tokens: [B, K, d_model]
        """
```

### 5.2 Geo Projector

Map geometry tokens to the LLM hidden dimension:

```python
class GeoProjector(nn.Module):
    def forward(self, geo_tokens: Tensor) -> Tensor:
        """
        Returns [B, K, d_llm]
        """
```

Use a 2-layer MLP with GELU and LayerNorm.

### 5.3 Geometry Gate

Implement a learnable scalar or vector gate:

```python
geo_tokens = gate * geo_tokens
```

Start with a conservative initialization:

```text
gate ≈ 0.1 to 0.3
```

Log gate values during training.

---

## 6. Qwen-VGGT Wrapper

Implement:

```text
geopsro4d/model/qwen_vggt_wrapper.py
geopsro4d/model/input_builder.py
```

### 6.1 Required Behavior

The wrapper should:

```text
1. Run the original Qwen visual processing path for RGB frames.
2. Load or compute VGGT geometry cache for the same frames.
3. Convert VGGT outputs into geometry tokens.
4. Insert geometry tokens into the model input sequence.
5. Support geometry-on, geometry-drop, zero-geometry, and shuffled-geometry modes.
```

### 6.2 Geometry Modes

Implement these modes:

```python
GEOMETRY_NORMAL = "normal"
GEOMETRY_ZERO = "zero"
GEOMETRY_DROP = "drop"
GEOMETRY_SHUFFLE = "shuffle"
GEOMETRY_DEPTH_ONLY = "depth_only"
GEOMETRY_DEPTH_CAMERA = "depth_camera"
GEOMETRY_FULL = "full"
```

Use them for ablations.

### 6.3 Input Concatenation

The final LLM input should contain:

```text
text prompt tokens
RGB visual tokens handled by native Qwen pipeline
special marker <geo>
VGGT geometry embeddings
special marker </geo>
answer tokens during training
```

If the existing Qwen pipeline cannot easily insert embeddings after tokenization, implement using `inputs_embeds` and an attention mask.

### 6.4 Special Tokens

Add special tokens if needed:

```text
<geo>
</geo>
```

Ensure tokenizer resizing and embedding initialization are handled.

---

## 7. Stage 1 Training: VGGT Geometry Alignment

Implement:

```text
geopsro4d/train/train_stage1_align.py
configs/stage1_vggt_align.yaml
scripts/run_stage1_align.sh
```

### 7.1 Training Setup

```text
Data: SPAR-234K or geometry-heavy SPAR subset
Input: RGB + VGGT geometry
Output: original answer
Loss: next-token cross-entropy
```

Frozen:

```text
VGGT
Qwen visual encoder
Qwen LLM
original multimodal projector
```

Trainable:

```text
GeoTokenizer trainable MLPs
GeoResampler
GeoProjector
GeometryGate
```

### 7.2 Suggested Config

```yaml
model:
  base_model: Qwen3-VL-4B-Instruct
  vggt_model: /path/to/vggt
  num_geo_tokens: 64
  geometry_mode: normal

train:
  epochs: 1
  learning_rate: 1.0e-4
  warmup_ratio: 0.03
  scheduler: cosine
  global_batch_size: 64
  max_seq_len: 4096
  num_frames: 8
  bf16: true
  freeze_vggt: true
  freeze_visual_encoder: true
  freeze_llm: true
  train_geo_adapter: true

data:
  train_jsonl: data/spar/train.jsonl
  val_jsonl: data/spar/val.jsonl
  vggt_cache_root: cache/vggt/spar
```

### 7.3 Stage 1 Evaluation

After training, run:

```text
normal VGGT
zero VGGT
shuffled VGGT
RGB-only baseline
```

Output:

```text
results/stage1/geometry_sanity.csv
```

Required columns:

```text
model, geometry_mode, dataset, accuracy, answer_parse_rate, avg_length
```

---

## 8. Stage 2 Training: Geometry-Augmented SFT

Implement:

```text
geopsro4d/train/train_stage2_sft.py
configs/stage2_vggt_sft.yaml
scripts/run_stage2_sft.sh
```

### 8.1 Training Setup

Initialization:

```text
ckpt_stage1_vggt_align
```

Data:

```text
LLaVA-Hound-64K
SPAR-234K
```

No PSRO formatting in SFT.

Use original QA / instruction-output format.

### 8.2 Batch Mixing

Implement a weighted dataset sampler:

```text
SPAR: 70%
LLaVA-Hound: 30%
```

### 8.3 Geometry Drop

For each training sample:

```python
if random.random() < 0.7:
    geometry_mode = "normal"
else:
    geometry_mode = "zero"
```

Log the actual geometry-on/drop ratio.

### 8.4 Trainable Modules

Recommended:

```text
Frozen:
  VGGT
  optional Qwen visual encoder

Trainable:
  LLM LoRA
  multimodal projector
  GeoResampler
  GeoProjector
  GeometryGate
```

### 8.5 Suggested Config

```yaml
model:
  base_model: Qwen3-VL-4B-Instruct
  init_checkpoint: outputs/stage1_vggt_align
  num_geo_tokens: 64
  use_lora: true
  lora_rank: 64
  lora_alpha: 128

train:
  epochs: 1
  lr_lora: 1.0e-5
  lr_projector: 5.0e-5
  lr_geo_adapter: 5.0e-5
  warmup_ratio: 0.03
  scheduler: cosine
  global_batch_size: 64
  max_seq_len: 4096
  num_frames: 8
  bf16: true
  geometry_on_ratio: 0.7
  geometry_drop_ratio: 0.3

data:
  datasets:
    - name: spar
      jsonl: data/spar/train.jsonl
      weight: 0.7
      vggt_cache_root: cache/vggt/spar
    - name: llava_hound
      jsonl: data/llava_hound/train.jsonl
      weight: 0.3
      vggt_cache_root: cache/vggt/llava_hound
```

### 8.6 Stage 2 Evaluation

Evaluate:

```text
DSR-Bench dev
SPAR-val
VSI-Bench
Video-MME if available
```

Output:

```text
results/stage2/sft_eval.csv
```

---

## 9. PSRO Parsing and Reward

Implement:

```text
geopsro4d/reward/answer_parser.py
geopsro4d/reward/psro_parser.py
geopsro4d/reward/process_rules.py
geopsro4d/reward/reward_fn.py
```

### 9.1 Answer Parser

Required behavior:

```python
def parse_answer(text: str, choices: list[str] | None = None) -> str | None:
    """
    Extract the final answer from the Answer section.
    For multiple choice, return A/B/C/D.
    For short answer, return normalized text.
    """
```

Rules:

```text
Prefer text after the last 'Answer:' field.
For multiple-choice tasks, extract a single option letter.
If multiple conflicting answers appear, return None.
Normalize full-width letters, parentheses, and punctuation.
```

### 9.2 PSRO Parser

```python
def parse_psro(text: str) -> dict:
    """
    Return sections:
      observation
      transition
      derivation
      answer
    Also return flags for missing or empty sections.
    """
```

Accept case-insensitive section headers:

```text
Observation:
Transition:
Derivation:
Answer:
```

### 9.3 Process Rules

Implement weak rule-based process consistency.

Spatial predicates:

```text
left, right, above, below, front, behind, near, far, closer, farther,
between, inside, outside, overlap, occlude, visible, hidden, distance,
direction, orientation, viewpoint, camera, object, region
```

Temporal/change predicates:

```text
moves, moved, moving, changes, becomes, remains, approaches, recedes,
appears, disappears, before, after, first, then, later, earlier,
transition, across frames, from frame, to frame
```

Process score:

```python
def process_score(parsed: dict, task_type: str | None) -> float:
    score = 0.0
    score += observation_has_entities_and_spatial_terms
    score += transition_has_change_terms_for_multi_image_or_video
    score += derivation_mentions_observation_or_transition
    score += answer_is_not_repeated_in_every_section
    return clipped_score_in_0_1
```

Do not implement trace verification.

### 9.4 Reward Function

Implement:

```python
def compute_reward(response: str, gold_answer: str, choices: list[str] | None, task_type: str | None) -> dict:
    parsed = parse_psro(response)
    pred = parse_answer(response, choices)

    r_acc = 1.0 if pred == normalize_answer(gold_answer) else 0.0
    r_out_fmt = output_format_reward(pred, response)
    r_psro_fmt = psro_format_reward(parsed)
    r_proc = process_score(parsed, task_type)
    r_len = length_penalty(response)

    total = (
        r_acc
        + 0.20 * r_out_fmt
        + 0.15 * r_psro_fmt
        + 0.10 * r_acc * r_proc
        - 0.05 * r_len
    )

    return {
        "reward": total,
        "r_acc": r_acc,
        "r_out_fmt": r_out_fmt,
        "r_psro_fmt": r_psro_fmt,
        "r_proc": r_proc,
        "r_len": r_len,
        "pred_answer": pred,
    }
```

### 9.5 Reward Unit Tests

Create tests for:

```text
correct answer with valid PSRO
correct answer without PSRO
wrong answer with valid PSRO
multiple conflicting answers
empty Transition
very long repetitive output
lowercase section headers
Chinese punctuation around answer options
```

---

## 10. Stage 3 Training: PSRO-RFT

Implement:

```text
geopsro4d/train/train_stage3_rft.py
configs/stage3_psro_rft.yaml
scripts/run_stage3_rft.sh
```

### 10.1 Initialization

```text
init = ckpt_stage2_vggt_sft
reference_model = ckpt_stage2_vggt_sft
```

There is no PSRO cold-start SFT.

### 10.2 Data

Use weighted sampling:

```text
4DThinker-4DRL: 70%
SpatialLadder multi-image: 30%
```

### 10.3 Prompt Template

Use this prompt during RFT:

```text
You should solve the problem using the following format:

Observation: describe relevant entities, spatial relations, views, and visible states.
Transition: describe changes across frames/images, including motion, relation changes, occlusion, or viewpoint changes.
Derivation: reason from the observation and transition to the answer.
Answer: provide the final answer only.

Question: {question}
Choices: {choices_if_any}
```

### 10.4 Rollout Settings

```yaml
rft:
  algorithm: gspo  # or grpo if existing implementation uses grpo
  group_size: 8
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 512
  kl_coef: 0.02
  reward_normalization: group
  reference_model: outputs/stage2_vggt_sft
```

Do not set temperature to near-zero values. RFT needs diverse rollouts.

### 10.5 Trainable Modules

```text
Trainable:
  LLM LoRA
  GeoResampler
  GeoProjector
  GeometryGate

Frozen:
  VGGT
```

### 10.6 RFT Logs

Log every training step or interval:

```text
reward_total
r_acc
r_out_fmt
r_psro_fmt
r_proc
r_len
answer_parse_rate
psro_format_rate
avg_output_length
kl
entropy
geometry_gate_value
```

Save sample rollouts periodically.

Output:

```text
results/stage3/rollout_samples.jsonl
results/stage3/reward_logs.jsonl
```

---

## 11. Evaluation

Implement:

```text
geopsro4d/eval/eval_dsr.py
geopsro4d/eval/eval_aux.py
```

### 11.1 DSR Evaluation

Evaluation should support two prompt modes:

```text
direct_answer
psro
```

For final GeoPSRO-4D, use PSRO prompt but compute accuracy only from the final Answer section.

Metrics:

```text
accuracy
answer_parse_rate
psro_format_rate
avg_output_length
accuracy_by_task_type, if task_type is available
```

Output:

```text
results/eval/dsr_bench_{model_name}.jsonl
results/eval/dsr_bench_summary.csv
```

Each JSONL record should contain:

```python
{
    "sample_id": str,
    "question": str,
    "choices": list[str] | None,
    "gold_answer": str,
    "raw_response": str,
    "pred_answer": str | None,
    "correct": bool,
    "geometry_mode": str,
    "frame_indices": list[int],
    "task_type": str | None,
}
```

### 11.2 Auxiliary Evaluation

Implement analogous evaluation for:

```text
VSI-Bench
Video-MME
SPAR-val
optional VLM4D / STI-Bench
```

---

## 12. Ablation Scripts

### 12.1 Geometry Ablation

Implement:

```text
geopsro4d/eval/eval_geometry_ablation.py
scripts/run_ablation_geometry.sh
```

Run modes:

```text
normal
zero
shuffle
depth_only
depth_camera
full
```

Output:

```text
results/ablations/geometry_ablation.csv
```

Columns:

```text
model, checkpoint, geometry_mode, dataset, accuracy, answer_parse_rate, psro_format_rate, avg_length
```

### 12.2 Reward Ablation

Implement configs for:

```text
answer_only
answer_plus_output_format
answer_plus_psro_format
answer_plus_process
full_psro_rft
```

Each run should save:

```text
checkpoint
reward logs
eval summary
rollout samples
```

Output table:

```text
results/ablations/reward_ablation.csv
```

### 12.3 RFT Data Ablation

Implement configs for:

```text
SpatialLadder only
4DThinker only
SpatialLadder + 4DThinker
```

Output:

```text
results/ablations/rft_data_ablation.csv
```

---

## 13. Command Templates

### 13.1 Cache VGGT

```bash
bash scripts/run_vggt_cache.sh
```

### 13.2 Stage 1

```bash
bash scripts/run_stage1_align.sh
```

### 13.3 Stage 2

```bash
bash scripts/run_stage2_sft.sh
```

### 13.4 Stage 3

```bash
bash scripts/run_stage3_rft.sh
```

### 13.5 Evaluation

```bash
bash scripts/run_eval_dsr.sh outputs/stage2_vggt_sft normal
bash scripts/run_eval_dsr.sh outputs/stage3_psro_rft normal
bash scripts/run_eval_dsr.sh outputs/stage3_psro_rft zero
bash scripts/run_eval_dsr.sh outputs/stage3_psro_rft shuffle
```

---

## 14. Implementation Priorities

Implement in this order:

```text
1. Dataset schema normalization and frame sampler.
2. VGGT cache extractor and cache loader.
3. GeoTokenizer, GeoResampler, GeoProjector, GeometryGate.
4. Qwen-VGGT wrapper with geometry modes.
5. Stage 1 alignment training.
6. Geometry sanity evaluation: normal / zero / shuffle.
7. Stage 2 geometry-augmented SFT.
8. PSRO parser and reward unit tests.
9. Stage 3 PSRO-RFT.
10. DSR-Bench evaluation and ablation scripts.
```

Do not start RFT before Stage 1 sanity checks pass.

---

## 15. Acceptance Checklist

The code implementation is acceptable when:

```text
[ ] VGGT cache files are generated and loadable.
[ ] Qwen and VGGT use identical sampled frames.
[ ] Geometry modes normal/zero/shuffle work.
[ ] Stage 1 can train only GeoAdapter modules.
[ ] Stage 1 logs show non-trivial geometry gate values.
[ ] Stage 1 geometry sanity table is produced.
[ ] Stage 2 can train with LLaVA-Hound + SPAR weighted sampling.
[ ] Stage 2 supports 70% geometry-on and 30% geometry-drop.
[ ] No PSRO-format SFT is implemented as a separate stage.
[ ] PSRO parser extracts Observation, Transition, Derivation, Answer.
[ ] Reward function returns component rewards and total reward.
[ ] RFT generates group rollouts with non-zero temperature.
[ ] RFT logs reward components, KL, answer parse rate, and PSRO format rate.
[ ] DSR evaluation extracts final Answer correctly.
[ ] Geometry ablation supports normal, zero, shuffle, depth-only, depth+camera, full.
[ ] No trace verifier or evidence graph module exists.
```

---

## 16. Important Failure Modes

### VGGT ignored

Symptom:

```text
normal VGGT accuracy ≈ zero/shuffle accuracy
```

Debug:

```text
check frame index alignment
check geometry tensor scale
check gate value
check whether geometry embeddings enter inputs_embeds
increase GeoAdapter learning rate
reduce noisy raw geometry tokens
```

### RFT learns format only

Symptom:

```text
PSRO format rate rises but accuracy does not
```

Debug:

```text
increase R_acc dominance
reduce process reward weight
add or increase length penalty
ensure process reward is answer-gated
lower max_new_tokens
```

### Answer parsing unstable

Symptom:

```text
raw response contains answer, but parser returns None
```

Debug:

```text
handle parentheses, punctuation, full-width letters, lowercase letters
prefer the final Answer section
reject multiple conflicting answer letters only when truly conflicting
```

### RFT unstable due to no cold-start

This is expected to be more fragile because the design intentionally removes PSRO-format SFT.

Mitigation within RFT only:

```text
clear PSRO prompt
strong format reward
KL regularization
group size 8
reasonable temperature 0.7
max_new_tokens 512
small process reward
```

---

## 17. Prohibited Implementations

Do not implement:

```text
trace labels
trace parser
trace verifier
evidence graph exporter
DSR metadata reward
hard verification of intermediate states
PSRO cold-start SFT stage
```

The only process structure is PSRO, introduced in RFT prompt and reward.
