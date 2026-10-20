# Architecture notes

This document is the "long-form" companion to the README. Where the README
is the sales pitch, this file is the engineering notebook.

## Why no `accelerate`?

Three reasons:

1. **Readability.** `accelerate.prepare()` quietly monkey-patches your
   model, your dataloader, and your optimizer. That is great in production,
   terrible for understanding. Our trainer's training loop is a vanilla
   PyTorch `for batch in loader: loss.backward(); optimizer.step()` — you
   already know what it does.
2. **No multi-GPU.** This project is explicitly single-GPU. If you want
   multi-GPU, see `alibaba/ROLL` or `verl`.
3. **No `torch.compile` / fused kernels.** Those buy you 1.3-1.5x speedup
   in exchange for a much harder debugging experience. We are not chasing
   the last 5% — we are chasing the first 95% of clarity.

## Why hand-rolled LoRA?

PEFT's LoRA is ~3000 LoC of code, with support for ~17 model architectures
and a dozen quantisation modes. For a teaching project, that is overkill.
The hand-rolled version in `optim/lora.py` is 130 LoC and does exactly
one thing: wrap a `nn.Linear` with a low-rank A·B update.

The four "magic lines" of LoRA are:

```python
nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
nn.init.zeros_(self.lora_B.weight)
self.scaling = alpha / r
return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
```

That's it. The kaiming/zeros init is what makes the LoRA delta start at
zero, so the wrapped layer behaves exactly like the base layer at step 0.

## The DPO loss, in three lines

The maths is famously fiddly. Here is the **whole** loss function:

```python
pi_logratios = policy_chosen_logp - policy_rejected_logp
ref_logratios = ref_chosen_logp - ref_rejected_logp
loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

Three things to notice:

1. We never need to compute a `pi/ref` ratio in the KL sense — the DPO
   derivation has already cancelled the partition function.
2. `beta` is a temperature. Higher beta = more aggressive deviation from
   the reference. TRL's default is 0.1; the DPO paper recommends 0.1–0.5.
3. The metric `rewards/acc = (beta * (pi_logratios - ref_logratios) > 0).mean()`
   tells you how often the policy ranks chosen above rejected. It should
   rise from ~0.5 to ~0.7+ during training. If it doesn't, something is
   wrong with the data (likely chosen and rejected are too similar, or
   the prompt is doing all the work).

## Why a single 2B forward?

Most "naive" DPO implementations do two separate `B`-sized forwards:

```python
# Naive
chosen_out = model(chosen_ids)
rejected_out = model(rejected_ids)
```

This is correct, but it does two kernel launches and pays the per-launch
overhead twice. The 2B forward is one launch, one set of attention
matmul, one backward.

The catch: the activation memory is now 2x a 1B forward. For small models
that is fine. For large models you want to chunk.

## Why the SFT regulariser?

The DPO loss only constrains **relative** preference. There is no term
that prevents the policy from drifting into a region of weight space
that gives garbage outputs but still ranks chosen > rejected on the
training distribution.

Adding a small SFT loss on the *chosen* response (typically
`gamma = 0.1 * DPO loss`) anchors the policy to the base distribution.
This is the TRL default in `DPOLoss -> DPOTrainer` and is what
`--gamma` controls in our CLI.

See the "Tricks or Traps" paper by the ROLL team (arXiv 2508.08221) for
a systematic ablation.

## Why is the reference on CPU?

The reference model is a frozen copy of the *base* model — it is
identical to the policy base. So why pay 2x VRAM for it when we can pay
3x wall time for the offload?

For Qwen-0.5B the math is:
* ref VRAM on GPU: ~1.0 GB
* ref forward on GPU: ~50 ms
* ref forward on CPU: ~150 ms
* ref round-trip (cpu + back): ~170 ms

If your DPO step is 600 ms (forward + backward + optim) on GPU, then
the ref on CPU adds 170 ms → ~28% slower. If your DPO step would
otherwise OOM because of the 2x ref weight, the slowdown is irrelevant.

For larger models the slowdown grows: a 7B ref on CPU is ~5x slower
than the GPU forward. At that point you want to keep the ref on the
GPU and just pay the 2x VRAM. Our `--ref_device` flag is the lever.

## What's *not* in this repo

A short list, so you don't go looking:

* Multi-GPU / FSDP / DeepSpeed — use `alibaba/ROLL` or `verl-project/verl`.
* GRPO / REINFORCE / PPO — coming in `grpo-from-scratch`.
* Agentic RL — coming in `nano-rock`.
* Speculative decoding / quantization — see `vllm-project/vllm`.
* Full DPO with iterative reference updating — that's a *different*
  algorithm, see "Iterative DPO" or "Nash-MD".

The point of this project is to be the foundation. The two follow-ups
build on top.
