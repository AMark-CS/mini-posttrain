"""Model loaders.

A single helper layer for the three models every post-training script needs:

* a **policy** model (trainable) — used for SFT and the chosen/rejected logp in DPO
* a **reference** model — frozen copy used to compute the DPO reference log-prob
* a **tokenizer** — always shared

We default to ``Qwen2.5-0.5B`` because it runs in 4 GB of VRAM and trains in
under 30 minutes on a 4090 for a 5k-example SFT run. Bump to ``Qwen2.5-1.5B``
or ``Qwen3-4B-Base`` if you have the headroom.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_posttrain.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def load_tokenizer(name_or_path: str = DEFAULT_MODEL, **kwargs: Any) -> Any:
    tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if tok.pad_token is None:
        # Most Qwen/Llama tokenizers ship with no pad token. Reuse eos for
        # both pad and eos — this matches TRL's default behaviour.
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(
    name_or_path: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target: str = "all-linear",
) -> Any:
    """Load a *trainable* policy model.

    If ``use_lora=True`` we wrap the base model with PEFT LoRA and freeze the
    base weights. We do **not** call ``enable_input_require_grads`` here —
    that's only needed for gradient checkpointing, which is a separate flag.
    """
    from nano_posttrain.optim import apply_lora

    log.info("loading policy model %s (dtype=%s, lora=%s)", name_or_path, dtype, use_lora)
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if use_lora:
        model = apply_lora(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target,
        )
    return model.to(device)


def load_reference_model(
    name_or_path: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Any:
    """Load a *frozen* reference model.

    Why CPU by default? On a 24 GB GPU, holding the policy + the LoRA params +
    the reference in bf16 is fine for ~0.5B but it is *not* fine for 7B+.

    The "DPO ref-on-CPU" trick: at every step we move the **activation batch**
    to CPU, do a forward pass through the reference model there, and ship the
    per-token logp tensor back to the GPU. CPU forward is ~3-4x slower than
    GPU forward but we save 2x the model memory on the GPU. This is exactly
    the pattern trl uses for DPO with QLoRA (see
    https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model ).
    """
    log.info("loading reference model %s on %s (dtype=%s)", name_or_path, device, dtype)
    ref = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    ref.config.use_cache = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref.to(device)
