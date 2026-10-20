"""A plain SFT trainer.

We deliberately do **not** use HF :class:`Trainer` or :class:`accelerate`. The
goal is to show exactly what is happening per step:

* forward pass with the policy model
* compute the masked cross-entropy loss
* backward, clip, step
* (optional) WandB log

WandB is opt-in via the ``WANDB_DISABLED`` env var or by setting
``use_wandb=False``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nano_posttrain.data import SFTDataset, sft_collate_fn, make_sft_collate
from nano_posttrain.losses import sft_loss
from nano_posttrain.models import load_model, load_tokenizer
from nano_posttrain.utils.logger import get_logger
from nano_posttrain.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    data_path: str = ""
    output_dir: str = "./output/sft"
    max_length: int = 1024
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_epochs: int = 1
    max_steps: int = -1
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 500
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    use_wandb: bool = True
    wandb_project: str = "nano-posttrain"
    grad_checkpointing: bool = False
    max_examples: int = -1  # -1 means use all


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class SFTTrainer:
    """Single-GPU SFT trainer. ~150 LoC of actual training logic."""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(config.seed)
        self._build()

    def _build(self) -> None:
        log.info("config: %s", self.config)
        self.tokenizer = load_tokenizer(self.config.model_name)
        dtype = _DTYPE_MAP[self.config.dtype]
        self.model = load_model(
            self.config.model_name,
            dtype=dtype,
            device=self.device,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        if self.config.grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        if self.config.use_lora:
            from nano_posttrain.optim import freeze_non_lora, mark_only_lora_trainable
            freeze_non_lora(self.model)
            tr, tot = mark_only_lora_trainable(self.model)
            log.info("LoRA trainable params: %d / %d (%.2f%%)", tr, tot, 100 * tr / tot)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.dataset = SFTDataset(
            self.config.data_path,
            self.tokenizer,
            max_length=self.config.max_length,
        )
        if self.config.max_examples > 0:
            self.dataset._examples = self.dataset._examples[: self.config.max_examples]
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=True,
            collate_fn=make_sft_collate(self.tokenizer.pad_token_id),
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False,
        )
        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        self._wandb = None
        if not self.config.use_wandb or os.environ.get("WANDB_DISABLED") == "1":
            return
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project=self.config.wandb_project, config=self.config.__dict__)
            self._wandb = wandb
        except Exception as e:  # pragma: no cover
            log.warning("wandb init failed: %s — continuing without", e)
            self._wandb = None

    def _lr_at(self, step: int, total: int) -> float:
        warmup = max(1, int(total * self.config.warmup_ratio))
        if step < warmup:
            return self.config.learning_rate * step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return self.config.learning_rate * 0.5 * (1 + float(torch.cos(torch.tensor(progress * 3.1415926))))

    def fit(self) -> None:
        cfg = self.config
        steps_per_epoch = max(1, len(self.loader) // cfg.grad_accum_steps)
        total_steps = steps_per_epoch * cfg.num_epochs if cfg.max_steps == -1 else cfg.max_steps
        log.info("training: %d optimizer steps (%d micro-batches per epoch)", total_steps, len(self.loader))

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        step = 0
        micro = 0
        t0 = time.time()
        running_loss = 0.0
        self.optimizer.zero_grad()
        for epoch in range(cfg.num_epochs):
            for batch in self.loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = sft_loss(out.logits, batch["labels"]) / cfg.grad_accum_steps
                loss.backward()
                running_loss += loss.item() * cfg.grad_accum_steps
                micro += 1

                if micro % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip
                        )
                    lr = self._lr_at(step, total_steps)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
                    if step % cfg.log_every == 0:
                        dt = time.time() - t0
                        tok = cfg.per_device_batch_size * cfg.grad_accum_steps * batch["input_ids"].size(1) * cfg.log_every
                        log.info(
                            "step %d/%d | loss=%.4f | lr=%.2e | tok/s=%.0f",
                            step, total_steps, running_loss / (cfg.log_every * cfg.grad_accum_steps), lr, tok / dt
                        )
                        if self._wandb is not None:
                            self._wandb.log({"train/loss": running_loss / (cfg.log_every * cfg.grad_accum_steps),
                                             "train/lr": lr,
                                             "train/tok_per_s": tok / dt,
                                             "train/step": step})
                        running_loss = 0.0
                        t0 = time.time()
                    if step % cfg.save_every == 0:
                        self._save()
                    if step >= total_steps:
                        self._save()
                        return
        self._save()

    def _save(self) -> None:
        cfg = self.config
        out = Path(cfg.output_dir) / f"step_{self._safe_step()}"
        out.mkdir(parents=True, exist_ok=True)
        if cfg.use_lora:
            from nano_posttrain.optim import lora_state_dict
            torch.save(lora_state_dict(self.model), out / "lora.pt")
        else:
            self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        log.info("saved checkpoint to %s", out)

    def _safe_step(self) -> str:
        # In the wild you'd use a real step counter; this is a teaching file.
        return "latest"
