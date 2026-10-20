"""Tests for the DPO loss.

We don't need a GPU for these — the loss is a pure function of two tensors.
The whole suite runs in <1 second on a laptop.
"""
from __future__ import annotations

import torch

from nano_posttrain.losses import dpo_loss, dpo_loss_with_sft_regularizer, sft_loss


def test_dpo_loss_symmetry_at_init() -> None:
    """If policy == ref, the loss should be -log(sigmoid(0)) = log(2)."""
    B = 4
    pi_w = torch.zeros(B)
    pi_l = torch.zeros(B)
    ref_w = torch.zeros(B)
    ref_l = torch.zeros(B)
    loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)
    # -log(sigmoid(0)) = -log(0.5) = log(2) ≈ 0.6931
    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-5), f"got {loss}"
    # The accuracy metric uses (logits > 0); at exactly zero this is 0, which
    # is fine — it's a *strict* "do I prefer chosen" check, not a coin flip.
    assert 0.0 <= metrics["rewards/acc"] <= 1.0


def test_dpo_loss_decreases_when_policy_prefers_chosen() -> None:
    """If the policy already strongly prefers chosen over rejected, the loss
    should be *below* the no-signal baseline of log(2)."""
    B = 4
    pi_w = torch.tensor([5.0] * B)
    pi_l = torch.tensor([0.0] * B)
    ref_w = torch.tensor([0.0] * B)
    ref_l = torch.tensor([0.0] * B)
    loss, _ = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5)
    # log(2) ≈ 0.6931 is the no-signal baseline; a strong preference should
    # push the loss close to zero.
    assert float(loss) < 0.7, f"got {float(loss)}"


def test_dpo_loss_increases_when_policy_prefers_rejected() -> None:
    """Mirror of the previous test: a policy that prefers rejected should
    incur a loss *above* the no-signal baseline."""
    B = 4
    pi_w = torch.tensor([0.0] * B)
    pi_l = torch.tensor([5.0] * B)
    ref_w = torch.tensor([0.0] * B)
    ref_l = torch.tensor([0.0] * B)
    loss, _ = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5)
    assert float(loss) > 0.7, f"got {float(loss)}"


def test_sft_regularizer_adds_loss() -> None:
    B = 4
    pi_w = torch.zeros(B)
    pi_l = torch.zeros(B)
    ref_w = torch.zeros(B)
    ref_l = torch.zeros(B)
    sft_val = torch.tensor(2.0)
    loss_plain, _ = dpo_loss(pi_w, pi_l, ref_w, ref_l)
    loss_aug, metrics = dpo_loss_with_sft_regularizer(pi_w, pi_l, ref_w, ref_l, sft_val, gamma=0.5)
    assert torch.isclose(loss_aug, loss_plain + 0.5 * sft_val, atol=1e-6)
    assert "sft_loss" in metrics
    assert "total_loss" in metrics


def test_sft_loss_ignores_minus_100() -> None:
    """Loss computed with -100 mask equals loss over the unmasked positions
    only."""
    V = 100
    T = 6
    B = 2
    logits = torch.randn(B, T, V)
    labels_a = torch.tensor([[0, 1, 2, -100, -100, -100]] * B)
    labels_b = torch.tensor([[0, 1, 2, 3, 4, 5]] * B)
    # sft_loss shifts: shift_logits = logits[:, :-1, :], shift_labels = labels[:, 1:].
    # For labels_a, the unmasked positions in shift_labels are indices 0 and 1
    # (label values 1 and 2). Those map to logits[:, 0, :] and logits[:, 1, :].
    shift_logits = logits[:, :2, :]
    shift_labels = labels_a[:, 1:3]
    expected = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1)
    )
    assert torch.isclose(sft_loss(logits, labels_a), expected)
    # Sanity: fully-unmasked version is *different* (averages over more tokens).
    assert not torch.isclose(sft_loss(logits, labels_a), sft_loss(logits, labels_b))
