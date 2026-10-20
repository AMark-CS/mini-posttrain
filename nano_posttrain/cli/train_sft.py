"""CLI: train SFT. Run with ``python -m cli.train_sft ...``."""
from __future__ import annotations

import argparse
import json
import os
import sys

from nano_posttrain.trainers import SFTTrainer
from nano_posttrain.trainers.sft_trainer import SFTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an SFT model with nano-posttrain.")
    p.add_argument("--config", type=str, default=None, help="Path to a JSON config file (optional).")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output/sft")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--use_wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="nano-posttrain")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--max_examples", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        cfg_dict = vars(args).copy()
        cfg_dict.pop("config", None)
    cfg = SFTConfig(**{k: v for k, v in cfg_dict.items() if k in SFTConfig.__dataclass_fields__})
    trainer = SFTTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    sys.exit(main())
