"""DPO trainer with the *LoRA + ref-on-CPU* memory trick.

The two big ideas this file is trying to demonstrate:

1. **The reference model is a frozen copy of the *base* (pre-LoRA) model**, not
   a frozen copy of the policy. With LoRA, ``policy = base + LoRA(A,B)`` so
   the reference is just ``base``. We don't need to hold two LoRA copies.

2. **CPU offload for the reference forward.** In a naive impl you keep both
   models on the GPU, doubling VRAM. We move the activation tensor to CPU
   (cheap — it's just a [B, T] int tensor) and run the reference forward
   there. The reference's per-token log-probs come back as a [B] float —
   we ship those to the GPU for the DPO loss.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nano_posttrain.data import DPODataset, dpo_collate_fn, make_dpo_collate
from nano_posttrain.losses import dpo_loss, dpo_loss_with_sft_regularizer, sft_loss
from nano_posttrain.models import load_model, load_reference_model, load_tokenizer
from nano_posttrain.utils.logger import get_logger
from nano_posttrain.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class DPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    data_path: str = ""
    output_dir: str = "./output/dpo"
    max_length: int = 1024
    max_prompt_length: int = 512
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    num_epochs: int = 1
    max_steps: int = -1
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    beta: float = 0.1
    gamma: float = 0.0  # SFT regulariser weight (TRL default is 0.0)
    grad_clip: float = 1.0
    log_every: int = 5
    save_every: int = 500
    dtype: str = "bfloat16"
    use_wandb: bool = True
    wandb_project: str = "nano-posttrain"
    ref_device: str = "cpu"  # "cpu" for memory savings, "cuda" if you have headroom
    max_examples: int = -1


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class DPOTrainer:
    """Single-GPU DPO trainer with LoRA + optional ref-on-CPU."""

    def __init__(self, config: DPOConfig, *,
                 on_step: Callable[[dict[str, int | float]], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> None:
        self.config = config
        self.on_step = on_step
        self.should_stop = should_stop or (lambda: False)
        self.completed_steps = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(config.seed)
        self._build()

    def _build(self) -> None:
        cfg = self.config
        log.info("config: %s", cfg)
        self.tokenizer = load_tokenizer(cfg.model_name)
        dtype = _DTYPE_MAP[cfg.dtype]
        # Policy: trainable LoRA on top of base.
        self.model = load_model(
            cfg.model_name,
            dtype=dtype,
            device=self.device,
            use_lora=cfg.use_lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )
        # Reference: frozen base, *without* LoRA, on CPU by default.
        self.ref_model = load_reference_model(
            cfg.model_name,
            dtype=dtype,
            device=cfg.ref_device,
        )
        if cfg.use_lora:
            from nano_posttrain.optim import freeze_non_lora, mark_only_lora_trainable
            freeze_non_lora(self.model)
            tr, tot = mark_only_lora_trainable(self.model)
            log.info("LoRA trainable params: %d / %d (%.2f%%)", tr, tot, 100 * tr / tot)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.dataset = DPODataset(
            cfg.data_path,
            self.tokenizer,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_prompt_length,
        )
        if cfg.max_examples > 0:
            self.dataset._examples = self.dataset._examples[: cfg.max_examples]
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=make_dpo_collate(self.tokenizer.pad_token_id),
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

    @staticmethod
    def _seq_logp(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Forward through ``model`` and return the sum of log-probs of the
        *label* tokens, shape ``[B]``.

        We use a *single* 2B forward rather than two B forwards — the
        equivalent gradients when ``model`` is the policy model are zero for
        the reference. Memory-wise the 2B forward costs ~2x the activation
        memory of a 1B forward, but we save a kernel launch.
        """
        out = model(input_ids=input_ids, use_cache=False)
        # Inline import to avoid circular dep
        from nano_posttrain.losses.dpo_loss import _gather_logp
        # The reference model may live on a different device than the labels
        # (the whole point of the "ref on CPU" trick), so align them here.
        labels = labels.to(out.logits.device)
        return _gather_logp(out.logits, labels)

    def _ref_logp(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Run the reference forward.

        If the ref is on CPU, we move the input batch there, run, and ship the
        resulting [B] logp back to the GPU. If the ref is on GPU, just call
        directly under ``no_grad``.
        """
        if self.config.ref_device == "cpu":
            input_ids_cpu = input_ids.cpu()
            with torch.no_grad():
                ref_lp = self._seq_logp(self.ref_model, input_ids_cpu, labels, self.config.ref_device)
            return ref_lp.to(self.device)
        with torch.no_grad():
            return self._seq_logp(self.ref_model, input_ids, labels, self.config.ref_device)

    def _lr_at(self, step: int, total: int) -> float:
        warmup = max(1, int(total * self.config.warmup_ratio))
        if step < warmup:
            return self.config.learning_rate * step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return self.config.learning_rate * 0.5 * (1 + float(torch.cos(torch.tensor(progress * 3.1415926))))

    def fit(self) -> None:
        cfg = self.config
        self.model.train()
        steps_per_epoch = max(1, len(self.loader) // cfg.grad_accum_steps)
        total_steps = steps_per_epoch * cfg.num_epochs if cfg.max_steps == -1 else cfg.max_steps
        log.info("training: %d optimizer steps", total_steps)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        step = 0
        micro = 0
        t0 = time.time()
        running = {"loss": 0.0, "n": 0}
        update_loss = 0.0
        self.optimizer.zero_grad()

        for epoch in range(cfg.num_epochs):
            for batch in self.loader:
                if self.should_stop():
                    self._save()
                    return
                chosen_ids = torch.tensor(batch["chosen_input_ids"], dtype=torch.long, device=self.device)
                chosen_labels = torch.tensor(batch["chosen_labels"], dtype=torch.long, device=self.device)
                rejected_ids = torch.tensor(batch["rejected_input_ids"], dtype=torch.long, device=self.device)
                rejected_labels = torch.tensor(batch["rejected_labels"], dtype=torch.long, device=self.device)

                # ----- policy forward: 2B combined, GPU, with grad -----
                stacked_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
                stacked_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
                out = self.model(input_ids=stacked_ids, use_cache=False)
                from nano_posttrain.losses.dpo_loss import _gather_logp
                policy_lp = _gather_logp(out.logits, stacked_labels)
                policy_chosen_lp, policy_rejected_lp = policy_lp.chunk(2, dim=0)

                # ----- reference forward: 2B combined, no grad, possibly CPU -----
                ref_stacked = stacked_ids
                ref_labels = stacked_labels
                ref_lp = self._ref_logp(ref_stacked, ref_labels)
                ref_chosen_lp, ref_rejected_lp = ref_lp.chunk(2, dim=0)

                # ----- DPO loss -----
                if cfg.gamma > 0.0:
                    sft_val = sft_loss(out.logits[: chosen_ids.size(0)], chosen_labels)
                    loss, metrics = dpo_loss_with_sft_regularizer(
                        policy_chosen_lp, policy_rejected_lp,
                        ref_chosen_lp, ref_rejected_lp,
                        sft_val, beta=cfg.beta, gamma=cfg.gamma,
                    )
                else:
                    loss, metrics = dpo_loss(
                        policy_chosen_lp, policy_rejected_lp,
                        ref_chosen_lp, ref_rejected_lp,
                        beta=cfg.beta,
                    )

                (loss / cfg.grad_accum_steps).backward()
                running["loss"] += float(loss.detach())
                update_loss += float(loss.detach())
                running["n"] += 1
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
                    self.completed_steps = step
                    if self.on_step is not None:
                        self.on_step({"step": step, "total_steps": total_steps,
                                      "loss": update_loss / cfg.grad_accum_steps, "learning_rate": lr,
                                      "preference_accuracy": metrics["rewards/acc"], "margin": metrics["margin"]})
                    update_loss = 0.0
                    if step % cfg.log_every == 0:
                        dt = time.time() - t0
                        avg = running["loss"] / max(1, running["n"])
                        log.info(
                            "step %d/%d | loss=%.4f margin=%.3f acc=%.2f | lr=%.2e | s/it=%.2f",
                            step, total_steps, avg, metrics["margin"], metrics["rewards/acc"],
                            lr, dt / max(1, cfg.log_every * cfg.grad_accum_steps),
                        )
                        if self._wandb is not None:
                            self._wandb.log({"train/loss": avg, **{f"train/{k}": v for k, v in metrics.items()},
                                             "train/lr": lr, "train/step": step})
                        running = {"loss": 0.0, "n": 0}
                        t0 = time.time()
                    if step % cfg.save_every == 0:
                        self._save()
                    if step >= total_steps:
                        self._save()
                        return
        self._save()

    def _save(self) -> None:
        cfg = self.config
        out = Path(cfg.output_dir) / "latest"
        out.mkdir(parents=True, exist_ok=True)
        if cfg.use_lora:
            from nano_posttrain.optim import lora_state_dict
            torch.save(lora_state_dict(self.model), out / "lora.pt")
        else:
            self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        log.info("saved DPO checkpoint to %s", out)
