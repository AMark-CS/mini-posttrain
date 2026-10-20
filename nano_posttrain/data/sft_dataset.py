"""SFT dataset that tokenizes ``(prompt, response)`` pairs and masks the loss
on every prompt token. The result is a sequence of::

    input_ids      = [BOS] prompt tokens        response tokens      [EOS]
    labels         = [-100] -100 ...            resp tokens          resp    [EOS]
    attention_mask = [  1 ]   1 ...               1 ...                 1

We support two on-disk formats:

* HuggingFace ``datasets.Dataset`` loaded with ``datasets.load_dataset`` — used
  for UltraChat, OpenHermes, ShareGPT, etc. The expected columns are
  ``messages`` (a list of role/content dicts) or ``prompt``/``response`` strings.
* Plain ``.jsonl`` files with one of ``{"messages": [...]}`` or
  ``{"prompt": "...", "response": "..."}`` per line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from nano_posttrain.utils.logger import get_logger

log = get_logger(__name__)


def _to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in example and example["messages"]:
        return list(example["messages"])
    if "prompt" in example and "response" in example:
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    if "instruction" in example and "output" in example:
        # Alpaca-style
        msgs = []
        if example.get("input"):
            msgs.append({"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"})
        else:
            msgs.append({"role": "user", "content": example["instruction"]})
        msgs.append({"role": "assistant", "content": example["output"]})
        return msgs
    raise ValueError(f"Example has no recognised SFT fields: {list(example)[:5]}")


@dataclass
class _TokenizedExample:
    input_ids: list[int]
    labels: list[int]


class SFTDataset(Dataset):
    """Tokenised SFT dataset with prompt-masked labels.

    The implementation is intentionally explicit: we apply the chat template
    twice — once for the prompt prefix, once for the full conversation — and
    use the length difference to mark the loss-mask boundary. This is the
    same trick TRL's SFTTrainer uses internally.
    """

    def __init__(
        self,
        data: Sequence[dict[str, Any]] | str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        mask_prompt: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self._examples: list[_TokenizedExample] = []
        self._load(data)
        log.info("SFTDataset: %d examples", len(self._examples))

    def _load(self, data: Sequence[dict[str, Any]] | str | Path) -> None:
        if isinstance(data, (str, Path)):
            data = self._load_path(Path(data))
        for ex in data:
            tok = self._tokenize(_to_messages(ex))
            if tok is not None:
                self._examples.append(tok)

    @staticmethod
    def _load_path(path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    def _tokenize(self, messages: list[dict[str, str]]) -> _TokenizedExample | None:
        tok = self.tokenizer
        # 1) Render the prompt alone to find where the response starts.
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tok.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_length:
            # Truncate from the *left* of the response, never the prompt —
            # the alternative is you end up training on incomplete prompts.
            resp_ids = full_ids[len(prompt_ids):]
            keep = self.max_length - len(prompt_ids)
            if keep <= 16:
                return None  # prompt alone is too long, skip
            resp_ids = resp_ids[:keep]
            full_ids = prompt_ids + resp_ids

        if self.mask_prompt:
            labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
        else:
            labels = list(full_ids)
        return _TokenizedExample(input_ids=full_ids, labels=labels)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self._examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def sft_collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length SFT examples."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attn = []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def make_sft_collate(pad_token_id: int) -> Callable[[list[dict[str, list[int]]]], dict[str, torch.Tensor]]:
    """Return a *picklable* collate function.

    A lambda would break ``DataLoader(num_workers>0)`` on Windows, where the
    default start method is ``spawn`` and every worker re-imports the module.
    ``functools.partial`` around a module-level function pickles cleanly.
    """
    return partial(sft_collate_fn, pad_token_id=pad_token_id)
