"""DPO dataset.

Each example is a triplet ``(prompt, chosen, rejected)`` where ``chosen`` and
``rejected`` are full assistant turns. The loss is computed on the *response*
tokens only — DPO is an off-policy method that conditions on a fixed prompt
and tries to push mass from ``rejected`` to ``chosen``.

We render two full conversations and use the prefix-length trick to locate
the response boundary, exactly like the SFT dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Sequence

from nano_posttrain.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class _Triplet:
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]


class DPODataset:
    """Triplet dataset for direct-preference optimisation."""

    def __init__(
        self,
        data: Sequence[dict[str, Any]] | str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self._examples: list[_Triplet] = []
        self._load(data)
        log.info("DPODataset: %d triplets", len(self._examples))

    def _load(self, data: Sequence[dict[str, Any]] | str | Path) -> None:
        if isinstance(data, (str, Path)):
            data = self._load_path(Path(data))
        for ex in data:
            t = self._tokenize(ex)
            if t is not None:
                self._examples.append(t)

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

    def _tokenize(self, ex: dict[str, Any]) -> _Triplet | None:
        if not {"prompt", "chosen", "rejected"}.issubset(ex):
            raise ValueError(f"DPO example missing fields: {list(ex)[:5]}")
        tok = self.tokenizer
        prompt_msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        if len(prompt_ids) > self.max_prompt_length:
            return None  # Skip over-long prompts rather than truncate — DPO is sensitive.

        def render_response(resp: str) -> tuple[list[int], list[int]]:
            msgs = prompt_msgs + [{"role": "assistant", "content": resp}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) > self.max_length:
                # Truncate response tail to fit.
                keep = self.max_length - len(prompt_ids)
                ids = prompt_ids + ids[len(prompt_ids):][:keep]
            labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
            return ids, labels

        chosen_ids, chosen_labels = render_response(ex["chosen"])
        rejected_ids, rejected_labels = render_response(ex["rejected"])
        return _Triplet(chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        t = self._examples[idx]
        return {
            "chosen_input_ids": t.chosen_input_ids,
            "chosen_labels": t.chosen_labels,
            "rejected_input_ids": t.rejected_input_ids,
            "rejected_labels": t.rejected_labels,
        }


def dpo_collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, list]:
    """Pad chosen and rejected to the *same* length so they can be stacked
    into a 2B * T tensor for a single forward pass.

    The unified max length is the larger of the longest chosen and the
    longest rejected. Within a batch this is virtually free (we're already
    padding), and it lets the trainer use the simple ``torch.cat(..., dim=0)``
    trick.
    """
    max_chosen = max(len(b["chosen_input_ids"]) for b in batch)
    max_rejected = max(len(b["rejected_input_ids"]) for b in batch)
    shared_max = max(max_chosen, max_rejected)

    def pad(seq: list[int], labels: list[int], target: int) -> tuple[list[int], list[int]]:
        n = len(seq)
        pad_n = target - n
        return seq + [pad_token_id] * pad_n, labels + [-100] * pad_n

    chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
    for b in batch:
        cids, clabels = pad(b["chosen_input_ids"], b["chosen_labels"], shared_max)
        rids, rlabels = pad(b["rejected_input_ids"], b["rejected_labels"], shared_max)
        chosen_ids.append(cids)
        chosen_labels.append(clabels)
        rejected_ids.append(rids)
        rejected_labels.append(rlabels)

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
    }


def make_dpo_collate(pad_token_id: int) -> Callable[[list[dict[str, list[int]]]], dict[str, list]]:
    return partial(dpo_collate_fn, pad_token_id=pad_token_id)
