"""LoRA helpers.

We deliberately do not import peft's high-level :class:`LoraConfig` constructor
in the trainer — the trainer takes plain ints. The reason is that you can read
this file in one screen and understand what is going on:

* pick the target modules (Qwen: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
* inject LoRA A/B as :class:`torch.nn.Linear` next to each base linear
* freeze the base weights, leave only LoRA trainable
* expose a tiny :func:`lora_state_dict` so we can save only the trainable params
"""
from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn

from nano_posttrain.utils.logger import get_logger

log = get_logger(__name__)

# The "kitchen-sink" target list for Qwen2/3-style models. Most other open
# causal LMs (Llama, Mistral, Yi, DeepSeek) use the same module names.
_DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """Wraps a frozen base :class:`nn.Linear` with a low-rank A·B update.

    Forward: ``y = x W^T + b + (alpha/r) * (x A) B``
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        # Freeze the base — we only train A and B.
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        # Standard Kaiming init for A; zeros for B so the delta is identity at
        # step 0, matching the LoRA paper.
        self.lora_A = nn.Linear(in_f, r, bias=False, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(r, out_f, bias=False, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _resolve_targets(spec: str | Iterable[str], named_modules: Iterable[str]) -> list[str]:
    """Resolve a target spec against the model.

    ``"all-linear"`` matches every ``nn.Linear`` name. Otherwise we treat the
    spec as a comma-separated list of suffixes (``q_proj,k_proj,...``).
    """
    all_linears = [n for n, _ in named_modules]
    if spec == "all-linear":
        # Filter out lm_head and embed_tokens — those are typically left alone.
        return [n for n in all_linears if not n.endswith(".lm_head")]
    if isinstance(spec, str):
        wanted = tuple(s.strip() for s in spec.split(",") if s.strip())
    else:
        wanted = tuple(spec)
    matches = [n for n in all_linears if n.endswith(wanted)]
    if not matches:
        raise ValueError(
            f"No modules matched {spec!r}. "
            f"Available module names: {all_linears[:5]}..."
        )
    return matches


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str | Iterable[str] = "all-linear",
) -> nn.Module:
    """Wrap matching :class:`nn.Linear` modules with :class:`LoRALinear`.

    Returns the model (mutated in place, but returned for chaining).
    """
    named_modules = [
        (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n
    ]
    targets = _resolve_targets(target_modules, named_modules)
    log.info("applying LoRA r=%d alpha=%d to %d modules", r, alpha, len(targets))
    replaced = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, attr)
        if isinstance(base, LoRALinear):
            continue
        wrapped = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        replaced += 1
    log.info("replaced %d Linear modules with LoRALinear", replaced)
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter that isn't an ``lora_A`` / ``lora_B`` matrix."""
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)


def mark_only_lora_trainable(model: nn.Module) -> tuple[int, int]:
    """``(trainable_params, total_params)``. Useful for the log line."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the LoRA-only state-dict, with the ``base_model.model.`` prefix
    stripped (peft-style) so you can merge it cleanly later.
    """
    out: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in n or "lora_B" in n):
            # Drop the trailing ``.weight`` suffix — we save the matrix, not
            # the parameter name convention.
            key = n.replace(".weight", "")
            out[key] = p.detach().cpu()
    return out
