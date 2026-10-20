# Architecture notes

## Training core

The core uses single-device PyTorch loops with explicit forward/backward passes, AdamW, gradient accumulation, clipping, mixed precision, and learning-rate scheduling. It does not implement distributed training or automatic checkpoint resume. SFT/DPO CLI modules are under `nano_posttrain.cli`; sample data is under `nano_posttrain/data`.

## LoRA

`nano_posttrain.optim.lora.LoRALinear` keeps the base linear layer and adds:

```python
self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * (alpha / r)
```

A uses Kaiming initialization and B starts at zero, so the initial low-rank update is zero. The trainers freeze non-adapter parameters. Rank-16 all-linear wrapping on the verified Qwen model produced 169 wrapped modules and 11,243,520 trainable parameters (2.23% of the wrapped model). Adapter export strips the trailing `.weight`; restoration adds it back. Preserve matching rank, alpha, and target settings.

## SFT

Datasets render the prompt prefix and complete conversation using the tokenizer's chat template. Prompt positions receive `-100` labels. The collator right-pads batches; shifted cross-entropy ignores masked labels. Long examples are truncated according to the configured length.

## DPO

Chosen and rejected sequences share a padded length and are concatenated into one `2B` batch. Policy and frozen reference forwards produce response sequence log-probabilities. The objective is:

```python
pi_logratios = policy_chosen_logp - policy_rejected_logp
ref_logratios = ref_chosen_logp - ref_rejected_logp
loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

An optional `gamma * sft_loss` term adds cross-entropy on chosen responses. The core DPO CLI defaults gamma to zero; the e-commerce wrapper defaults it to 0.1. Preference accuracy measures whether the policy's chosen/rejected log-ratio exceeds the reference's. It is not generation accuracy or a guaranteed training trajectory.

The reference is a separate frozen model loaded from the same base identifier. With CPU offload, input IDs move to the CPU, labels align to reference logits, and the sequence log-probabilities return to the policy device. Actual overhead depends on CPU/GPU, batch size, sequence length, and dtype.

## Checkpoints and scope

SFT saves to `step_latest`; DPO saves to `latest`. With LoRA these contain adapter tensors and tokenizer files, not a full model or resumable optimizer state. Each standalone trainer starts from its configured model; DPO does not automatically restore the SFT adapter.

The evaluation harness provides qualitative generations. The core does not include GRPO, serving infrastructure, quantization, or an online experiment system. Experimental GRPO and application scaffolding live in ecom-posttrain.

See [verification](verification.md) for measured behavior and known test failures. Performance ratios and hardware-capacity estimates from older notes have been removed because no controlled benchmark supports them.
