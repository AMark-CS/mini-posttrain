# DPO memory notes

## What occupies memory

A DPO step holds policy weights, trainable gradients/optimizer state, activations/logits, and a frozen reference. Frozen parameters still occupy memory. LoRA reduces trainable gradients and optimizer state; it does not remove the base model or the activations needed to propagate adapter gradients.

For a rough parameter-only estimate, bfloat16 weights use two bytes per parameter. Optimizer state precision and implementation affect total memory. Activation/logit memory depends on batch size, sequence length, vocabulary size, attention implementation, and dtype. Allocator reservations and CUDA context memory add overhead.

No peak-memory measurement was collected in the [2026-10-20 verification](verification.md). Earlier 1.3-GB/2.5-GB capacity claims and fixed CPU/GPU timings are not validated results.

## CPU reference

`--ref_device cpu` keeps the separately loaded frozen reference in system RAM. The trainer transfers input IDs to it, aligns labels with its logits, runs without gradients, and transfers sequence log-probabilities back to the policy device.

This avoids GPU residency for reference weights and forward intermediates. It uses system RAM, CPU compute, and device transfers. No universal 50% total-VRAM reduction or fixed slowdown follows from the implementation. Moving a reference from CPU to GPU increases GPU memory demand.

## LoRA

The base parameters are frozen and only low-rank matrices are optimized. The verified Qwen rank-16 configuration wrapped 169 modules and trained 11,243,520 of 505,276,288 total parameters (2.23%). The adapter fraction depends on rank and target modules.

The implementation loads a separate unwrapped base as reference; it does not share the policy's base allocation or dynamically disable adapters for reference forwards.

## Concatenated forward

Chosen/rejected tensors are concatenated into a batch of size `2B`. This reduces the number of model invocations compared with separate forwards, but does not halve FLOPs or guarantee a kernel speedup. Both alternatives' activations and full vocabulary logits still consume memory.

## Practical validation

Record model/dtype, reference device, batch size, sequence lengths, adapter targets, accumulation, and hardware. Warm up before timing. Measure allocated and reserved peak GPU memory and wall time over a fixed workload; compare CPU/GPU reference modes under identical settings. Keep inference throughput separate from training throughput.

Use smaller batches/sequence lengths when memory is limited. Gradient accumulation changes effective batch size without proportionally increasing per-forward activation memory. SFT exposes `--grad_checkpointing`; the DPO CLI does not currently expose that flag. Inspect each trainer before assuming a feature is supported.

The two-step CPU-reference DPO check establishes that the current path runs on the verified hardware. It is not a capacity guarantee for other model sizes or workloads.
