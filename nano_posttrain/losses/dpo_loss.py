"""Loss functions.

The SFT loss is just a token-mean cross-entropy on the (already-masked) labels.
The DPO loss is the direct-preference loss from Rafailov et al. 2023 with a
couple of practical bits:

* we use ``logsigmoid`` for numerical stability over ``-F.softplus(-beta * ...)``
* we expose an "SFT regulariser" term (the SFT loss on the chosen sequence)
  weighted by ``gamma`` — this is the ``DPOLoss -> DPOLossTrainer`` behaviour
  in TRL.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Vanilla causal-LM cross-entropy on already-masked labels.

    ``logits`` shape: ``[B, T, V]``.  ``labels`` shape: ``[B, T]`` with -100
    on tokens that should be ignored. We shift both by one so we predict
    ``logits[:, t, :]`` against ``labels[:, t+1]`` (next-token prediction).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _gather_logp(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Sum the log-probabilities of the *label* tokens.

    Returns shape ``[B]`` — one sum per sequence. We do the next-token shift
    here as well so the caller doesn't have to remember.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # ``gather`` along the vocab dim picks out the log-prob of the *true* next token.
    chosen_lp = log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (chosen_lp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Plain DPO loss.

    L = -E[ log sigmoid( beta * (logp_pi(y_w|x) - logp_pi(y_l|x)
                                 - logp_ref(y_w|x) + logp_ref(y_l|x)) ) ]

    Returns ``(loss, metrics)``. The metrics dict is what you'd want to log to
    WandB — accuracies and margins that tell you *whether* the model is
    actually learning the preference, not just that the loss is going down.
    """
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        margins = (pi_logratios - ref_logratios).detach()
        chosen_rewards = (beta * (policy_chosen_logp - ref_chosen_logp)).detach()
        rejected_rewards = (beta * (policy_rejected_logp - ref_rejected_logp)).detach()
        acc = (logits > 0).float().mean()
    metrics = {
        "loss": float(loss.detach()),
        "margin": float(margins.mean()),
        "rewards/chosen": float(chosen_rewards.mean()),
        "rewards/rejected": float(rejected_rewards.mean()),
        "rewards/acc": float(acc),
    }
    return loss, metrics


def dpo_loss_with_sft_regularizer(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    sft_loss_value: torch.Tensor,
    beta: float = 0.1,
    gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO + optional SFT loss on the chosen sequence (the TRL default).

    ``gamma=0.0`` recovers the plain DPO loss. Setting ``gamma=0.1`` tends to
    stabilise training and reduces "verbosity drift" — see the
    "Tricks or Traps" paper by the ROLL team for ablation results.
    """
    dpo, metrics = dpo_loss(
        policy_chosen_logp,
        policy_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        beta=beta,
    )
    if gamma > 0.0:
        total = dpo + gamma * sft_loss_value
    else:
        total = dpo
    metrics["sft_loss"] = float(sft_loss_value.detach())
    metrics["total_loss"] = float(total.detach())
    return total, metrics
