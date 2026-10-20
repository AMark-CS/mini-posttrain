# DPO memory tricks — the long-form notes

This file collects every memory-related trick we use (or don't use) in the
DPO trainer, with the maths.

## The baseline

A vanilla DPO implementation keeps two copies of the model in VRAM:

| | params | grads | Adam state (bf16) |
|--|--|--|--|
| Policy model | 2P | 2P | 8P |
| Reference model | 2P | 0 | 0 |
| **Total** | **4P** | **2P** | **8P** |

For Qwen-0.5B that's ~2.4 GB params + 1.2 GB grads + 4.8 GB optim = **8.4 GB**
just for the model. On a 24 GB card we have 15 GB to spare, which is fine.
On a 12 GB card, this doesn't fit.

## Trick 1: LoRA on the policy

LoRA replaces every targeted `nn.Linear` weight matrix `W` (shape
`[d_in, d_out]`) with `W + (alpha/r) * A @ B`, where `A` is `[d_in, r]`
and `B` is `[r, d_out]`. With `r = 16` and `alpha = 32` for a model with
~24 such matrices, the LoRA params are <0.5% of the base params.

| | params | grads | Adam state |
|--|--|--|--|
| Policy (LoRA r=16) | 0.005P | 0.005P | 0.02P |
| Reference (full) | 2P | 0 | 0 |
| **Total** | **2.005P** | **0.005P** | **0.02P** |

This is a **~4x** memory win on the policy. For Qwen-0.5B that's
~1.0 GB params + 2.5 MB grads + 10 MB Adam. The base is frozen so we
don't even need to keep a "policy fp32 master copy" — `requires_grad=False`
means the optimiser skips it entirely.

## Trick 2: Frozen base = reference

A subtle but important observation: with LoRA, the *base* of the policy
is a frozen copy of the original model. That is also exactly the DPO
reference. So we don't need a *second* full copy — we just need to
remember that the reference forward is the same forward as the policy
forward **without the LoRA delta**.

In `optim/lora.py`, the `LoRALinear.forward` is:
```python
return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
```

If we set `scaling = 0` (or, equivalently, set `lora_B.weight = 0`),
the LoRA delta is zero and the wrapped layer behaves exactly like
`self.base(x)`. We use this implicitly by loading a *separate, un-LoRA'd*
copy of the base for the reference.

## Trick 3: Ref on CPU

The reference forward is half the total DPO compute (per step), but it
doesn't need gradients. We can therefore:

1. Move the activation tensor to CPU.
2. Run the forward through the CPU reference.
3. Move the resulting `[B]` logp tensor back to the GPU.

The CPU forward is ~3-5x slower than the GPU forward, but it lets us
keep **the entire reference model out of VRAM**. This is what
`--ref_device cpu` enables.

The cost-benefit:

| ref device | VRAM used by ref | ref forward time |
|--|--|--|
| GPU bf16 | 1.0 GB (Qwen-0.5B) | 50 ms |
| CPU bf16 | 0 | 150-200 ms |

If you can spare the 1 GB, keep the ref on the GPU. If you can't, pay
the wall time.

## Trick 4: 2B concat forward

```python
stacked = torch.cat([chosen_ids, rejected_ids], dim=0)  # [2B, T]
out = model(stacked)                                      # [2B, T, V]
```

Doing this once instead of twice halves the kernel-launch overhead. The
*activation* memory is 2x a 1B forward (because we keep activations for
both halves in the autograd graph), but the *weight* memory is unchanged
and the *kernel time* is essentially 1x (slightly more than 1x because
of L2 cache pressure).

## Trick 5: Gradient checkpointing

We do **not** use it by default. With LoRA, the policy backward is small
(only the LoRA params get gradients), so the activation memory for the
backward is small. Adding gradient checkpointing slows down the forward
by ~30% in exchange for a small additional memory saving. Not worth it
for a 0.5B model on a 24 GB card.

For a 7B model, gradient checkpointing starts to matter. The CLI flag
`--grad_checkpointing` is wired up but commented in the trainer.

## Putting it all together

For a Qwen-0.5B DPO run with the defaults in this repo:

| Component | VRAM (MB) | Notes |
|--|--|--|
| Policy (LoRA r=16, bf16) | 1000 | base ~1000 MB + LoRA ~10 MB |
| Adam state (LoRA only) | 30 | 2 * 4-byte state * LoRA size |
| Reference (CPU, bf16) | 0 | held in RAM, not VRAM |
| Forward activations (B=2, T=1024) | 250 | back-of-envelope |
| **Total** | **~1.3 GB** | fits on a 4 GB card with room to spare |

For Qwen-1.5B the same configuration runs in ~3.5 GB. For Qwen-7B you
need to either (a) drop `--ref_device cpu` and put the ref on the GPU,
(b) reduce `--per_device_batch_size`, or (c) enable
`--grad_checkpointing`.
