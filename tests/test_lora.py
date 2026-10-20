"""Tests for the LoRA module.

We test on a tiny model so the test runs without any HF download.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nano_posttrain.optim import (
    apply_lora,
    freeze_non_lora,
    lora_state_dict,
    mark_only_lora_trainable,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 4, bias=False)


def test_lora_replaces_only_target_modules() -> None:
    m = _TinyModel()
    apply_lora(m, r=4, alpha=8, dropout=0.0, target_modules="q_proj")
    from nano_posttrain.optim.lora import LoRALinear
    assert isinstance(m.q_proj, LoRALinear)
    assert not isinstance(m.k_proj, LoRALinear)
    assert not isinstance(m.lm_head, LoRALinear)


def test_lora_init_is_identity() -> None:
    """At init, the LoRA delta should be zero (B is initialised to 0)."""
    torch.manual_seed(0)
    m = _TinyModel()
    base_layer = m.q_proj
    x = torch.randn(2, 8)
    y_base = base_layer(x).detach().clone()
    apply_lora(m, r=4, alpha=8, dropout=0.0, target_modules="q_proj")
    y_wrapped = m.q_proj(x)
    assert torch.allclose(y_wrapped, y_base, atol=1e-5)


def test_freeze_non_lora() -> None:
    m = _TinyModel()
    apply_lora(m, r=4, alpha=8, dropout=0.0, target_modules="q_proj,k_proj")
    freeze_non_lora(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    # Should be exactly the lora_A / lora_B matrices.
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable)
    tr, tot = mark_only_lora_trainable(m)
    assert tr > 0
    assert tot > tr


def test_lora_state_dict_keys() -> None:
    m = _TinyModel()
    apply_lora(m, r=4, alpha=8, dropout=0.0, target_modules="q_proj")
    freeze_non_lora(m)
    sd = lora_state_dict(m)
    assert all("lora_A" in k or "lora_B" in k for k in sd)
    # No ``.weight`` suffix.
    assert all(not k.endswith(".weight") for k in sd)
