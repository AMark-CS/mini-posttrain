"""Re-export for the SFT loss to keep imports tidy."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Vanilla causal-LM cross-entropy on already-masked labels."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
