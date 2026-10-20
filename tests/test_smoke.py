"""End-to-end smoke test with a tiny dataset. Skipped by default (no GPU
required, but the tokenizer does download a small vocab from HF the first
time).

Run with: ``pytest tests/test_smoke.py -v -m smoke`` (marker is added below)
or      : ``pytest tests/test_smoke.py -v --run-smoke`` via the
``--run-smoke`` flag we wire up in ``conftest.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch


# Mark every test in this module with "smoke" so the default ``pytest``
# collection does not run them (the user opts in with ``--run-smoke``).
pytestmark = pytest.mark.smoke


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _tiny_jsonl(tmp: Path) -> Path:
    p = tmp / "tiny.jsonl"
    p.write_text(
        json.dumps({"prompt": "1+1?", "response": "2."})
        + "\n" + json.dumps({"prompt": "2+2?", "response": "4."})
        + "\n"
    )
    return p


def test_sft_dataset_with_real_tokenizer(tmp_path: Path) -> None:
    """Build a SFTDataset with the real Qwen tokenizer and check shapes."""
    from transformers import AutoTokenizer
    from nano_posttrain.data import SFTDataset
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ds = SFTDataset(_tiny_jsonl(tmp_path), tok, max_length=128)
    assert len(ds) >= 1
    ex = ds[0]
    # The tokenised sequence must be < max_length and must contain at least
    # one *unmasked* label (otherwise we trained on nothing).
    assert len(ex["input_ids"]) <= 128
    assert any(x != -100 for x in ex["labels"])


def test_dpo_dataset_with_real_tokenizer(tmp_path: Path) -> None:
    from transformers import AutoTokenizer
    from nano_posttrain.data import DPODataset
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    p = tmp_path / "dpo.jsonl"
    p.write_text(json.dumps({
        "prompt": "What is 1+1?",
        "chosen": "It is 2.",
        "rejected": "It is 3.",
    }))
    ds = DPODataset(p, tok, max_length=128, max_prompt_length=64)
    assert len(ds) == 1
    ex = ds[0]
    # The labels should not all be -100 — we should be training on *something*.
    assert any(x != -100 for x in ex["chosen_labels"])
    assert any(x != -100 for x in ex["rejected_labels"])


def test_lora_against_real_qwen_linear() -> None:
    """Apply LoRA to a real Qwen module and verify trainable count."""
    from transformers import AutoModelForCausalLM
    from nano_posttrain.optim import apply_lora, mark_only_lora_trainable
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
    )
    apply_lora(model, r=8, alpha=16, dropout=0.0, target_modules="q_proj,k_proj,v_proj")
    from nano_posttrain.optim.lora import LoRALinear
    for n, m in model.named_modules():
        if n.endswith(("q_proj", "k_proj", "v_proj")):
            assert isinstance(m, LoRALinear), f"{n} was not wrapped"
    tr, tot = mark_only_lora_trainable(model)
    assert tr > 0
    assert tr < tot
