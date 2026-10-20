"""Tests for SFT/DPO data preprocessing.

We only test the structural correctness of the masking — the chat-template
subtraction trick is the easiest place to silently introduce a bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from nano_posttrain.data import DPODataset, SFTDataset, dpo_collate_fn, sft_collate_fn


DATA = Path(__file__).resolve().parent.parent / "data"


class _StubTokenizer:
    """A tiny stand-in tokenizer that splits on whitespace and assigns each
    token a sequential id. We bypass the real tokenizer so the test runs
    instantly and deterministically.
    """

    def __init__(self) -> None:
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = ""
        for m in messages:
            out += f"<{m['role']}>" + " " + m["content"] + " "
        if add_generation_prompt:
            out += "<assistant>"
        return out

    def __call__(self, text, add_special_tokens=True):
        toks = text.split()
        return type("X", (), {"input_ids": [hash(t) % 1000 + 10 for t in toks]})()


def test_sft_loss_masks_prompt() -> None:
    ds = SFTDataset(
        [{"prompt": "What is 2+2?", "response": "4."}],
        tokenizer=_StubTokenizer(),
        max_length=64,
    )
    assert len(ds) == 1
    ex = ds[0]
    # First non-(-100) label should be in the response region, not the prompt.
    labels = ex["labels"]
    first_resp = next((i for i, x in enumerate(labels) if x != -100))
    # The prompt "What is 2+2?" has 3 tokens, "<user>" + 3 + "<assistant>".
    # We just check first_resp > 0.
    assert first_resp > 0


def test_sft_collate_pads_to_max_in_batch() -> None:
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    out = sft_collate_fn(batch, pad_token_id=0)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_dpo_dataset_skips_overlong_prompts() -> None:
    long_prompt = " ".join(["word"] * 200)
    ds = DPODataset(
        [{"prompt": long_prompt, "chosen": "ok", "rejected": "no"}],
        tokenizer=_StubTokenizer(),
        max_prompt_length=16,
    )
    assert len(ds) == 0


def test_dpo_collate_stacks_chosen_and_rejected() -> None:
    batch = [
        {
            "chosen_input_ids": [1, 2, 3], "chosen_labels": [-100, -100, 3],
            "rejected_input_ids": [4, 5], "rejected_labels": [-100, 5],
        }
    ]
    out = dpo_collate_fn(batch, pad_token_id=0)
    assert len(out["chosen_input_ids"][0]) == 3
    assert len(out["rejected_input_ids"][0]) == 3
    assert out["rejected_input_ids"][0][-1] == 0  # right-pad


def test_sample_sft_json_loads() -> None:
    """Smoke-test the shipped sample file."""
    p = DATA / "sample_sft.json"
    assert p.exists(), p
    with p.open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert all("messages" in r for r in data)
