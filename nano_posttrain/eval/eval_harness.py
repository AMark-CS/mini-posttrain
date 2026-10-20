"""A minimal generation/reward evaluation harness.

We don't try to be a replacement for ``lm-eval-harness``. The point is to
have a tiny script you can read in one sitting to evaluate a policy model
qualitatively:

* load a model + LoRA checkpoint
* run a list of prompts through the model
* dump generations to stdout or to a JSONL

The reference model is *not* used here — this is just a sanity check that
training actually changed the policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from nano_posttrain.models import load_model, load_tokenizer
from nano_posttrain.utils.logger import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Path to a lora.pt checkpoint produced by SFTTrainer/DPOTrainer.")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Inline prompts. If empty, --prompts_file is used.")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="JSONL with {prompt: str} per line.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--output", type=str, default=None, help="Optional JSONL output path.")
    return p.parse_args()


def load_lora_into(model: torch.nn.Module, lora_path: str) -> None:
    """Restore a saved LoRA state-dict into an already-LoRA-wrapped model.

    The state-dict keys produced by :func:`lora_state_dict` look like
    ``model.layers.0.self_attn.q_proj.lora_A`` (with ``.weight`` stripped).
    We add ``.weight`` back, then ``load_state_dict(strict=False)``.
    """
    sd = torch.load(lora_path, map_location="cpu")
    renamed = {f"{k}.weight": v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    # Any "missing" keys should be the *base* weights, which is what we want.
    unexpected = [k for k in unexpected if ("lora_A" in k or "lora_B" in k)]
    if unexpected:
        raise RuntimeError(f"LoRA state-dict had unexpected keys: {unexpected[:5]}")
    log.info("loaded %d LoRA tensors from %s", len(sd), lora_path)


def collect_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts
    if args.prompts_file:
        out = []
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.append(rec.get("prompt") or rec.get("instruction") or line)
        return out
    # Default: 5 ultra-short sanity prompts.
    return [
        "Explain the bias-variance tradeoff in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number.",
        "What is the difference between SFT and DPO?",
        "Summarise the story of the tortoise and the hare in one sentence.",
        "What is 2+2?",
    ]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(
        args.model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
        use_lora=args.lora_path is not None,
    )
    if args.lora_path is not None:
        load_lora_into(model, args.lora_path)
    model.eval()

    out_path = Path(args.output) if args.output else None
    fout = open(out_path, "w") if out_path else None

    for prompt in collect_prompts(args):
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out_ids = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out_ids[0, ids.size(1):], skip_special_tokens=True)
        rec = {"prompt": prompt, "response": gen}
        print(f"\n>>> {prompt}\n{gen}")
        if fout:
            fout.write(json.dumps(rec) + "\n")
    if fout:
        fout.close()
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
