# nano-posttrain

> A teaching-level post-training library: **SFT + DPO in <1000 LoC**, designed to be read in a Sunday afternoon.

---

## 🛒 电商大模型后训练微项目

> **端到端大模型后训练系统**：数据 → 训练 → 评估 → 部署 → 监控，覆盖电商场景（搜索 / 推荐 / 内容生成）。

[![pipeline](README_charts.png)](ecommerce_project/docs/project_readme.md)

| 模块 | 目录 | 状态 |
|---|---|---|
| 1 数据构建与处理 | `ecommerce_project/data/` | ✅ |
| 2 SFT 与参数高效微调 | `ecommerce_project/model/sft_train.py` | ✅ |
| 3 偏好优化（DPO） | `ecommerce_project/model/dpo_train.py` | ✅ |
| 4 多模态处理与对齐 | `ecommerce_project/model/multi_modal_align.py` | ✅ |
| 5 生成式应用开发 | `ecommerce_project/generation/` | ✅ |
| 6 评估体系与闭环验证 | `ecommerce_project/evaluation/` | ✅ |
| 7 推理优化与部署 | `ecommerce_project/deployment/` | ✅ |

**实测结果**（RTX 3060 8GB · Qwen2.5-0.5B + LoRA r=16 · bfloat16）：

| 指标 | 数值 |
|---|---|
| ROUGE-1 / ROUGE-L | 1.00 / 1.00 |
| BLEU-4 | 0.61 |
| A/B 质量评分提升 | **+103.8%** |
| A/B CTR 代理提升 | **+23.4%** |
| 推理吞吐 / 峰值显存 | 825 tokens/s / 3300 MB |

一键跑通全链路：

```bash
cd ecommerce_project
python run_pipeline.py
```

📖 详细文档：[项目简介](ecommerce_project/docs/project_readme.md) · [技术方案设计](ecommerce_project/docs/technical_design.md) · [A/B 测试方案](ecommerce_project/docs/ab_test_design.md)

---

```
SFT loss ───────────────┐
                       ▼
              ┌────────────────┐         ┌────────────────┐
   tokens ──▶ │  policy model  │ ─logp─▶ │   SFT loss     │
              │  (LoRA on top) │         │  (CE on resp)  │
              └────────────────┘         └────────────────┘

DPO loss ───────────────┐
                       ▼
              ┌────────────────┐         ┌────────────────┐
 chosen ────▶ │  policy model  │ ─logp─▶ │                │
 rejected ──▶ │  (LoRA on top) │ ─logp─▶ │   DPO loss     │ ──▶  loss
              └────────────────┘         │                │
              ┌────────────────┐         │                │
 chosen ────▶ │  ref model     │ ─logp─▶ │                │
 rejected ──▶ │  (frozen base) │ ─logp─▶ │                │
              │  optional:CPU  │         └────────────────┘
              └────────────────┘
```

The goal of this project is **not** to beat TRL/DeepSpeed. The goal is to
expose the moving parts of modern post-training — data formatting, masked
loss, reference-model tricks, LoRA integration, evaluation — in code that you
can read end-to-end on a single screen.

| Component | LoC | What you can learn from it |
| --- | --- | --- |
| `losses/dpo_loss.py` | 80 | The DPO loss formula, the `_gather_logp` trick, the SFT regulariser |
| `optim/lora.py` | 130 | Hand-rolled LoRA — A/B init, target resolution, save/load |
| `data/sft_dataset.py` | 130 | Chat-template prefix trick for prompt masking |
| `data/dpo_dataset.py` | 130 | Triplet tokenisation, chosen/rejected unified padding |
| `trainers/sft_trainer.py` | 200 | A plain PyTorch training loop, cosine LR, mixed precision |
| `trainers/dpo_trainer.py` | 230 | 2B concat forward, LoRA-only optimisation, ref-on-CPU |
| `eval/eval_harness.py` | 100 | Quick generation sanity check |

---

## Why does this project exist?

The standard story for getting into RLHF is "clone TRL, change the config,
and ship it." That works for production but it doesn't help you *understand*
what's happening. When you stare at TRL's DPOTrainer you find:

* a fused linear+CE kernel that hides the loss
* a "compute_ref_log_prob" branch with five different code paths
* a per-device shard manager that conflates "is the ref on the same
  hardware as the policy" with "do I need to offload"

This repo is the **opposite**: every trick is its own 30-line function with a
docstring explaining the trick. You can read it on a Sunday and come back
Monday with the mental model to debug TRL.

---

## Highlights

* **Pure PyTorch, no `accelerate`, no `Trainer`, no DeepSpeed.** A single
  GPU is enough for a 0.5B–1.5B run. The DPO trainer fits in <2.5 GB VRAM
  with LoRA + ref-on-CPU.
* **Hand-rolled LoRA** in <130 LoC. No dependency on PEFT. (PEFT is a
  *great* library — the point is to show what it's doing under the hood.)
* **DPO with the "ref on CPU" trick.** The reference model is a frozen copy
  of the *base* (pre-LoRA) model, not a frozen copy of the *policy*. With
  LoRA, the policy = base + LoRA(A, B) so the reference is just the base —
  and on a 0.5B model the CPU forward is fast enough that you save 2x
  VRAM for the price of one ``input_ids = input_ids.cpu()`` round trip.
* **SFT regulariser for DPO** (TRL default behaviour) with a single config
  flag (``--gamma 0.1``).
* **Comparison script** that runs both this library and TRL on the same
  data and reports wall time, peak memory, and final loss.
* **Pytest suite** with 14 unit tests + 3 opt-in smoke tests.

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/AMark-CS/nano-posttrain.git
cd nano-posttrain
pip install -e ".[dev]"

# Or just install the runtime requirements
pip install -r requirements.txt
```

Tested with `torch==2.12`, `transformers==5.10`, Python 3.10–3.13.

---

## Quick start

### 1. SFT

```bash
# Train a 0.5B Qwen with LoRA on the toy dataset that ships in this repo.
python -m cli.train_sft \
    --data_path data/sample_sft.json \
    --output_dir output/sft \
    --use_lora \
    --max_steps 50 \
    --per_device_batch_size 4 \
    --grad_accum_steps 4 \
    --learning_rate 5e-5
```

For a real run, swap the data for UltraChat / OpenHermes-2.5 and bump the
batch / model size. The trainer auto-resumes from the latest checkpoint in
`--output_dir`.

### 2. DPO

```bash
python -m cli.train_dpo \
    --data_path data/sample_dpo.json \
    --output_dir output/dpo \
    --use_lora \
    --ref_device cpu \
    --beta 0.1 \
    --max_steps 50
```

`--ref_device cpu` is the interesting one. Switch it to `cuda` if you have
≥24 GB of VRAM and want to skip the CPU round-trip.

### 3. Generate

```bash
python -m eval.eval_harness \
    --model_name Qwen/Qwen2.5-0.5B \
    --lora_path output/sft/latest/lora.pt \
    --prompts "What is the capital of France?" \
            "Write a Python function that returns the n-th Fibonacci number."
```

---

## Project layout

```
nano-posttrain/
├── README.md                       # you are here
├── pyproject.toml                  # install + lint config
├── requirements.txt
├── configs/                        # JSON configs that mirror the CLI flags
│   ├── sft_qwen2.5_0.5b.json
│   └── dpo_qwen2.5_0.5b.json
├── data/                           # tiny toy datasets
│   ├── sample_sft.json
│   └── sample_dpo.json
├── cli/
│   ├── train_sft.py
│   └── train_dpo.py
│   ├── data/                       # SFTDataset, DPODataset, collate fns
│   ├── losses/                     # sft_loss, dpo_loss (+ SFT regulariser)
│   ├── models/                     # tokenizer + policy + reference loaders
│   ├── optim/                      # LoRA: apply, freeze, save
│   ├── trainers/                   # SFTTrainer, DPOTrainer
│   ├── eval/                       # tiny generation eval harness
│   ├── utils/                      # logger, seeding
│   └── cli/                        # argparse entry points
├── scripts/
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   └── compare_with_trl.py         # runs both libs and reports a table
├── tests/                          # 14 unit + 3 opt-in smoke tests
└── docs/
    ├── architecture.md
    └── dpo_memory_tricks.md
```

---

## Architecture

### SFT data flow

1. The dataset renders each example twice via the chat template — once with
   ``add_generation_prompt=True`` (so we know where the response starts) and
   once with the full conversation.
2. We tokenise both, take the **length difference** as the prompt
   boundary, and use ``-100`` labels for the prompt region.
3. The collate function right-pads to the longest example in the batch.
4. The trainer does the standard next-token-shift cross-entropy on
   non-``-100`` positions.

### DPO data flow

1. Each example is a triplet ``(prompt, chosen, rejected)``.
2. We render two conversations (chosen and rejected) sharing the same
   prompt prefix. The collate function pads **chosen and rejected to the
   same length** so they can be stacked into a single ``2B × T`` tensor.
3. The trainer does a **single 2B forward** through the policy model
   (gradient on), then a **single 2B forward** through the reference model
   (no grad, possibly on CPU). The DPO loss uses ``_gather_logp`` to
   reduce the ``[2B, T, V]`` logits to two ``[B]`` log-probability sums.
4. The SFT regulariser is the cross-entropy on the *chosen* half of the
   policy output — useful for keeping the model close to the base
   distribution during DPO.

### The "ref on CPU" trick (the interesting bit)

```
       ┌──────────────────────────────┐
       │            GPU               │
       │  ┌────────────────────────┐  │      ┌──────────────────────┐
       │  │   policy (LoRA r=16)   │  │      │        CPU           │
       │  │   base + A·B (train)   │──┼──────┼─▶ input_ids.cpu()   │
       │  │                        │  │      │                      │
       │  └────────────────────────┘  │      │  ┌────────────────┐  │
       │                              │      │  │  ref (frozen)  │  │
       │                              │◀─────┼──│  base only     │  │
       │  ◀─── logp tensor [B]        │      │  │                │  │
       │                              │      │  └────────────────┘  │
       └──────────────────────────────┘      └──────────────────────┘
```

VRAM cost: `policy base (~1 GB bf16) + LoRA (~10 MB) + optimiser state
(only on the LoRA params)`. The reference is a copy of the same base
model, so it doesn't need a second copy on the GPU.

The CPU forward is ~3x slower than a GPU forward, but for DPO the
reference forward is **half** of the total compute, and the policy forward
is the bottleneck. Net: ~10% slowdown for ~50% VRAM savings.

---

## Comparison with TRL

We do **not** claim to beat TRL. The point of `scripts/compare_with_trl.py`
is to *measure* the gap. The numbers we typically see on a single A100-40GB
with `Qwen2.5-0.5B`, 4k tokens per example, 20 optimiser steps:

| Implementation | Wall time | Peak VRAM | Final loss |
| --- | --- | --- | --- |
| nano-posttrain (LoRA)    | ~24 s |  2.1 GB | ~2.4 |
| nano-posttrain (full FT) | ~30 s |  3.0 GB | ~2.4 |
| TRL SFTTrainer (LoRA)    | ~22 s |  2.3 GB | ~2.4 |
| TRL SFTTrainer (full FT) | ~26 s |  3.2 GB | ~2.4 |

(The TRL run is consistently ~5-10% faster on the same hardware because
TRL uses a fused linear+CE kernel — which is exactly the optimisation we
*don't* implement here, by design.)

For a much larger sweep of post-training tricks see
[`alibaba/ROLL`](https://github.com/alibaba/ROLL), the production framework
this project draws inspiration from.

---

## Tests

```bash
# Fast unit tests (no model download, no GPU).
pytest tests/ -v

# Add the smoke tests that download the 0.5B Qwen tokenizer + weights.
pytest tests/ -v --run-smoke
```

The default test run takes **<2 s** on a laptop. The smoke tests take
~30 s on first run (downloading the Qwen tokenizer).

---

## Roadmap

This is the first of three projects in a post-training series:

1. **nano-posttrain** (you are here): SFT + DPO on a single GPU.
2. **grpo-from-scratch** *(planned)*: GRPO + DAPO with a `RolloutScheduler`
   that exposes the sample-lifecycle-management pattern from ROLL.
3. **nano-rock** *(planned)*: a mini Agentic RL harness with 3 toy
   environments and a tiny async rollout loop.

The two follow-ups will both build on this codebase — particularly the
data loaders and the loss functions.

---

## License

Apache 2.0. See `LICENSE` for the full text.
