# GeoPSRO-4D Core Idea

**Working title:** GeoPSRO-4D: Geometry-Augmented Process-Oriented Reinforcement Fine-Tuning for 4D Spatial Reasoning

This document is the revised core idea for the AAAI submission. It removes all trace-centric components and keeps a RoboRefer-style three-stage recipe:

```text
Stage 1: VGGT Geometry Alignment
Stage 2: Geometry-Augmented Spatial SFT
Stage 3: PSRO-RFT
```

The method should be written as a 4D spatial reasoning framework that integrates a frozen geometric foundation model into a multimodal LLM, then applies supervised and reinforcement post-training to improve dynamic spatial reasoning.

---

## 1. Core Positioning

The paper should no longer use the following concepts:

```text
trace
trace verifier
evidence graph
trace consistency reward
process-verified reasoning
DSR metadata verification
```

The paper should instead use the following concepts:

```text
VGGT geometry branch
geometry alignment
geometry-augmented SFT
PSRO reasoning format
process-oriented RFT
Observation-Transition-Derivation-Answer
structured dynamic spatial reasoning
```

The high-level claim is:

> Dynamic 4D spatial reasoning requires models to use geometric cues and to organize reasoning around spatial observations, temporal transitions, and answer derivation. We therefore integrate VGGT as a dedicated geometry branch, align it to the language model, perform geometry-augmented spatial SFT, and finally apply PSRO-style RFT with structured process rewards.

This is intentionally weaker and cleaner than a trace-verification claim. We do not claim that every intermediate process is formally verified. We claim that structured process-oriented RFT improves dynamic spatial reasoning.

---

## 2. RoboRefer-Style Recipe We Follow

The experimental recipe should mirror the RoboRefer structure at the level of training design:

```text
Dedicated geometric modality branch
→ modality alignment
→ spatial instruction tuning on the fused model
→ group-based RFT with outcome and process rewards
```

The analogy is:

| RoboRefer | GeoPSRO-4D |
|---|---|
| Depth branch | VGGT geometry branch |
| Depth projector alignment | VGGT geometry alignment |
| RGB/RGB-D spatial SFT | RGB/RGB+VGGT spatial SFT |
| GRPO/RFT with outcome and process reward | PSRO-RFT with answer, format, and process reward |
| Position/Orientation/Size process | Observation/Transition/Derivation process |

The paper should not say that it copies RoboRefer. It should present this as a general geometry-augmented post-training framework for 4D spatial reasoning.

---

## 3. Model Architecture

### 3.1 Base Model

Use a video/multimodal LLM such as:

```text
Qwen3-VL-4B-Instruct as the main backbone
Qwen3-VL-8B-Instruct as optional scale-up
```

The exact backbone can be adjusted based on available infrastructure, but all controlled experiments must use the same backbone unless explicitly labeled as scaling experiments.

### 3.2 Frozen VGGT Geometry Branch

VGGT is used as a frozen geometry foundation model. It receives the same frames as the RGB/VLM branch and produces geometric outputs:

```text
camera intrinsics / extrinsics
relative camera pose
depth maps
point maps
3D point tracks
visibility / confidence
intermediate geometry features, if available
```

Important input boundary:

```text
Allowed at inference:
  RGB frames
  VGGT predictions computed from RGB frames

Not allowed at inference:
  ground-truth pose
  ground-truth depth
  ground-truth trajectory
  dataset generator metadata
  DSR evidence graph
  hand-written trace labels
```

### 3.3 Geometry Token Construction

VGGT outputs should be compressed into compact geometry tokens before being passed to the LLM.

Recommended token groups:

```text
Camera tokens:
  frame-level camera and relative pose information

Frame geometry tokens:
  depth, point-map, confidence, and local 3D patch statistics

Track-tube tokens:
  high-confidence point tracks pooled across time, carrying displacement, visibility, and motion cues
```

A reasonable first implementation:

```text
sample frames: 8 for debugging, 16 for formal experiments
geometry tokens K: 64
track candidates: top 128 by confidence and temporal coverage
VGGT: frozen
Geo Resampler + Geo Projector + Geometry Gate: trainable
```

The geometry branch should be independent from the original RGB visual encoder. The fused input to the LLM is:

```text
RGB visual tokens + VGGT geometry tokens + text tokens
```

---

## 4. Stage 1: VGGT Geometry Alignment

### Goal

Stage 1 only solves one problem:

> Make the frozen VGGT geometry tokens readable by the multimodal LLM.

This stage corresponds to geometric modality alignment. It is not the full spatial SFT stage.

### Trainable and Frozen Modules

```text
Frozen:
  VGGT
  Qwen visual encoder
  Qwen LLM
  original multimodal projector

Trainable:
  Geo Tokenizer, if it has trainable MLPs
  Geo Resampler
  Geo Projector
  Geometry Gate
```

### Data

Use SPAR-234K, preferably starting with geometry-heavy subsets:

```text
relative direction
left / right / front / behind / above / below
near / far / distance
object size
viewpoint relation
egocentric relation
multi-view relation
```

LLaVA-Hound-64K is not necessary in Stage 1. It is mainly used in Stage 2 to preserve video instruction-following ability.

### Objective

Use standard next-token cross-entropy on spatial QA:

```text
L_align = - log p(answer | RGB tokens, VGGT geometry tokens, question)
```

Do not force PSRO format in Stage 1.

### Output

```text
ckpt_stage1_vggt_align
```

### Required Sanity Checks

Before entering SFT, run:

| Variant | Meaning |
|---|---|
| RGB only | base model without geometry |
| RGB + normal VGGT | normal geometry fusion |
| RGB + zero VGGT | geometry tokens replaced with zeros |
| RGB + shuffled VGGT | geometry cache from other samples |

Expected pattern:

```text
RGB + normal VGGT > RGB only
RGB + normal VGGT > RGB + zero VGGT
RGB + normal VGGT > RGB + shuffled VGGT
```

If this pattern does not appear, do not proceed to RFT. Fix VGGT frame alignment, token scale, projector, gate, and cache indexing first.

---

## 5. Stage 2: Geometry-Augmented Spatial SFT

### Goal

Stage 2 trains the fused RGB+VGGT model on large-scale spatial and video instruction data.

This stage should answer:

> Does a VGGT-augmented model learn stronger spatial and video reasoning than an RGB-only SFT baseline?

### Initialization

```text
init = ckpt_stage1_vggt_align
```

### Data

Use:

```text
LLaVA-Hound-64K
SPAR-234K
```

Recommended batch mixing:

```text
SPAR-234K: 70%
LLaVA-Hound-64K: 30%
```

SPAR provides dense spatial supervision. LLaVA-Hound preserves video instruction-following and general visual-language ability.

### Geometry Drop Training

To prevent the model from over-relying on VGGT, use mixed geometry modes:

```text
70% geometry-on:
  RGB frames + VGGT geometry tokens

30% geometry-drop:
  RGB frames + zero geometry tokens
```

This mirrors the RGB/RGB-D mixed-training philosophy: the model should use geometry when available but remain robust when geometry is noisy or absent.

### Output Format

No PSRO cold-start is used.

Stage 2 should keep the original dataset answer style:

```text
Question → Answer
```

Do not rewrite SFT samples into Observation/Transition/Derivation/Answer format. The PSRO format will be introduced only during RFT.

### Trainable and Frozen Modules

First practical implementation:

```text
Frozen:
  VGGT
  optional: Qwen visual encoder

Trainable:
  LLM LoRA
  RGB projector / multimodal projector
  Geo Resampler
  Geo Projector
  Geometry Gate
```

If full-parameter SFT is affordable, full-parameter fine-tuning can be attempted as a scale-up setting, but LoRA is the recommended first version.

### Suggested Hyperparameters

```text
base: Qwen3-VL-4B-Instruct
init: ckpt_stage1_vggt_align
epochs: 1
lr_lora: 1e-5
lr_projector: 5e-5
lr_geoadapter: 5e-5
warmup_ratio: 0.03
scheduler: cosine
sampled_frames: 8 for debug, 16 for formal training
geometry_on_ratio: 0.7
geometry_drop_ratio: 0.3
```

### Output

```text
ckpt_stage2_vggt_sft
```

---

## 6. Stage 3: PSRO-RFT

### Goal

Stage 3 directly initializes from Stage 2 and performs reinforcement fine-tuning with structured PSRO outputs.

There is no separate PSRO cold-start SFT.

```text
init = ckpt_stage2_vggt_sft
```

### RFT Data

Use:

```text
SpatialLadder multi-image: 15K
4DThinker-4DRL: 35K
```

Recommended sampling ratio:

```text
4DThinker-4DRL: 70%
SpatialLadder multi-image: 30%
```

Rationale:

```text
4DThinker-4DRL emphasizes dynamic 4D reasoning, temporal transitions, and state changes.
SpatialLadder multi-image strengthens multi-view spatial reasoning, reference-frame reasoning, and compositional spatial relations.
```

### PSRO Output Format

During RFT, the prompt should require:

```text
Observation: describe relevant entities, spatial relations, views, and visible states.
Transition: describe changes across frames/images, including motion, relation changes, occlusion, or viewpoint changes.
Derivation: reason from the observation and transition to the answer.
Answer: provide the final answer only.
```

Expected model output:

```text
Observation: ...
Transition: ...
Derivation: ...
Answer: B
```

### Reward Design

Use RoboRefer-style outcome reward + process reward, adapted to 4D QA.

Recommended reward:

```text
R = R_acc
  + 0.20 * R_out_fmt
  + 0.15 * R_psro_fmt
  + 0.10 * I(R_acc = 1) * R_proc
  - 0.05 * R_len
```

Definitions:

```text
R_acc:
  1 if final answer is correct, else 0.

R_out_fmt:
  1 if the output has a parseable final answer and no ambiguous multiple answers, else 0.

R_psro_fmt:
  score in [0, 1] for the presence and non-empty content of Observation, Transition, Derivation, Answer.

R_proc:
  weak process consistency score.
  It checks whether the response includes spatial entities, spatial predicates, temporal/change predicates, and whether Derivation references Observation/Transition.
  This is not a trace verifier.

R_len:
  penalty for excessively long, repetitive, or template-filled outputs.
```

Important:

```text
Do not over-weight process reward.
Answer correctness must dominate.
Use answer-gated process reward so that wrong answers do not receive high process reward merely for having a polished explanation.
```

### Algorithm

Use the existing PSRO/GSPO/GRPO training stack.

Recommended settings:

```text
algorithm: GSPO or GRPO
group_size: 8
temperature: 0.7
top_p: 0.9
max_new_tokens: 512
reference_model: ckpt_stage2_vggt_sft
KL: enabled
trainable: LLM LoRA + GeoAdapter + GeoProjector + GeometryGate
frozen: VGGT
```

### Output

```text
ckpt_stage3_psro_rft
```

---

## 7. Main Evaluation

### Primary Benchmark

```text
DSR-Bench
```

DSR-Bench should be the main table benchmark because the paper is about dynamic 4D spatial reasoning.

### Auxiliary Benchmarks

Use available auxiliary benchmarks:

```text
VSI-Bench
Video-MME
VLM4D, if available
STI-Bench, if available
SPAR validation, for spatial ability analysis
```

### Baselines

Controlled baselines:

```text
Qwen3-VL-4B-Instruct
Qwen3-VL-4B + SFT without VGGT
Qwen3-VL-4B + VGGT Alignment + SFT
Qwen3-VL-4B + VGGT + SFT + Answer-only RFT
Qwen3-VL-4B + VGGT + SFT + PSRO-RFT
Qwen3-VL-4B + SFT + PSRO-RFT without VGGT
```

Optional external baselines:

```text
Qwen3-VL-8B-Instruct
InternVL-style video MLLM
4DThinker / DSR-related reproduced baseline, if available
API models as reference upper bound, if allowed
```

---

## 8. Required Tables

### Main Result Table

| Method | VGGT | Stage 1 Align | SFT Data | RFT Data | Reward | DSR-Bench | VSI-Bench | Video-MME |
|---|---|---|---|---|---|---:|---:|---:|
| Qwen3-VL-4B | No | No | - | - | - | XX.X | XX.X | XX.X |
| SFT w/o VGGT | No | No | LLaVA-Hound + SPAR | - | - | XX.X | XX.X | XX.X |
| VGGT-SFT | Yes | Yes | LLaVA-Hound + SPAR | - | - | XX.X | XX.X | XX.X |
| VGGT + Answer RFT | Yes | Yes | LLaVA-Hound + SPAR | SL + 4DThinker | Answer | XX.X | XX.X | XX.X |
| GeoPSRO-4D | Yes | Yes | LLaVA-Hound + SPAR | SL + 4DThinker | Answer + Format + Process | XX.X | XX.X | XX.X |

### Geometry Ablation

| Variant | DSR-Bench | Transition | Reference Frame | SPAR-val |
|---|---:|---:|---:|---:|
| w/o VGGT | XX.X | XX.X | XX.X | XX.X |
| zero VGGT tokens | XX.X | XX.X | XX.X | XX.X |
| shuffled VGGT tokens | XX.X | XX.X | XX.X | XX.X |
| VGGT depth only | XX.X | XX.X | XX.X | XX.X |
| VGGT depth + camera | XX.X | XX.X | XX.X | XX.X |
| full VGGT with tracks | XX.X | XX.X | XX.X | XX.X |

### RFT Reward Ablation

| Reward | DSR-Bench | PSRO Format Rate | Answer Parse Rate | Avg Length |
|---|---:|---:|---:|---:|
| Answer only | XX.X | XX.X | XX.X | XX.X |
| Answer + Output Format | XX.X | XX.X | XX.X | XX.X |
| Answer + PSRO Format | XX.X | XX.X | XX.X | XX.X |
| Answer + PSRO Process | XX.X | XX.X | XX.X | XX.X |
| Full PSRO-RFT | XX.X | XX.X | XX.X | XX.X |

### RFT Data Ablation

| RFT Data | DSR-Bench | Spatial Multi-Image | Dynamic Transition |
|---|---:|---:|---:|
| SpatialLadder only | XX.X | XX.X | XX.X |
| 4DThinker only | XX.X | XX.X | XX.X |
| SpatialLadder + 4DThinker | XX.X | XX.X | XX.X |

---

## 9. Required Figures

### Figure 1: Overall Pipeline

Show:

```text
RGB frames → base visual encoder → RGB visual tokens
RGB frames → frozen VGGT → camera/depth/point/track → Geo Resampler/Projector → geometry tokens
RGB tokens + geometry tokens + question → LLM

Stage 1: VGGT Alignment
Stage 2: Geometry-Augmented SFT
Stage 3: PSRO-RFT
```

### Figure 2: VGGT Geometry Tokenization

Show camera tokens, frame geometry tokens, and track-tube tokens compressed into K geometry tokens.

### Figure 3: PSRO-RFT Reward

Show rollout generation and rewards:

```text
Observation / Transition / Derivation / Answer
→ answer parser
→ format checker
→ weak process checker
→ total reward
→ GSPO/GRPO update
```

### Figure 4: Qualitative Case

Show one dynamic spatial question with:

```text
sampled frames
a baseline answer
GeoPSRO-4D PSRO output
final answer
```

No trace overlay is required.

---

## 10. Main Experimental Order

Execute in this order:

```text
1. Prepare DSR-Bench evaluation pipeline and Qwen3-VL baseline.
2. Build VGGT cache extraction for SPAR, LLaVA-Hound, SpatialLadder, 4DThinker, DSR-Bench.
3. Implement Geo Tokenizer, Geo Resampler, Geo Projector, Geometry Gate.
4. Run Stage 1 VGGT Geometry Alignment on SPAR.
5. Run normal / zero / shuffled VGGT sanity checks.
6. Run Stage 2 Geometry-Augmented SFT on LLaVA-Hound + SPAR.
7. Evaluate SFT on DSR-Bench dev, SPAR-val, VSI-Bench, Video-MME.
8. Run Answer-only RFT baseline from Stage 2 checkpoint.
9. Run full PSRO-RFT from Stage 2 checkpoint.
10. Run geometry, SFT, RFT reward, and data ablations.
11. Fill paper tables only with real numbers.
```

---

## 11. Risks and How to Handle Them

### Risk 1: VGGT branch is ignored

Symptoms:

```text
normal VGGT ≈ zero VGGT
normal VGGT ≈ shuffled VGGT
```

Fixes:

```text
increase GeoAdapter LR
normalize geometry token scale
inspect gate values
verify frame/cache alignment
reduce number of noisy track tokens
start with depth+camera before adding tracks
```

### Risk 2: RFT learns format but not accuracy

Symptoms:

```text
PSRO Format Rate increases
DSR accuracy does not increase
outputs become longer
```

Fixes:

```text
increase answer reward dominance
lower process reward weight
add length penalty
use answer-gated process reward
reduce max_new_tokens
```

### Risk 3: No PSRO cold-start makes RL unstable

This is accepted by design. We are not adding a separate cold-start stage.

Mitigation inside RFT only:

```text
strong format reward
clear prompt
low process reward
KL to SFT checkpoint
group size 8
max_new_tokens 512
```

### Risk 4: SFT hurts general video ability

Use LLaVA-Hound replay and geometry-drop training.

---

## 12. Paper Claim Boundaries

Do claim:

```text
geometry-augmented post-training
VGGT-based spatial geometry fusion
structured Observation-Transition-Derivation-Answer reasoning
process-oriented RFT
improved DSR-Bench and auxiliary spatial/video benchmarks, if supported by results
```

Do not claim:

```text
formal reasoning verification
faithful trace verification
full geometric proof of model reasoning
access to ground-truth 3D states at inference
guaranteed causal use of process text
```

---

## 13. Final One-Line Summary

GeoPSRO-4D follows a RoboRefer-style alignment-SFT-RFT recipe: it first aligns frozen VGGT geometry tokens to a video MLLM, then performs geometry-augmented spatial SFT on LLaVA-Hound and SPAR, and finally applies PSRO-style RFT on SpatialLadder and 4DThinker data to improve dynamic 4D spatial reasoning on DSR-Bench.
