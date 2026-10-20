"""CPU-only trainer progress/cancellation checks with a tiny language model."""
import torch
from nano_posttrain.trainers.sft_trainer import SFTTrainer, SFTConfig
from nano_posttrain.trainers.dpo_trainer import DPOTrainer, DPOConfig


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 6)
        self.head = torch.nn.Linear(6, 12)

    def forward(self, input_ids, **kwargs):
        return type("Output", (), {"logits": self.head(self.embedding(input_ids))})()


def test_sft_reports_actual_updates_and_cancels(monkeypatch, tmp_path):
    def build(self):
        self.device = "cpu"
        self.model = TinyLM()
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self._wandb = None
        self.loader = [{"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}] * 4
    monkeypatch.setattr(SFTTrainer, "_build", build)
    monkeypatch.setattr(SFTTrainer, "_save", lambda self: None)
    rows = []
    trainer = SFTTrainer(SFTConfig(output_dir=str(tmp_path), max_steps=4, grad_accum_steps=1),
                         on_step=rows.append, should_stop=lambda: len(rows) >= 2)
    trainer.fit()
    assert [r["step"] for r in rows] == [1, 2]
    assert trainer.completed_steps == 2
    assert trainer.model.training
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)


def test_dpo_reports_preference_metrics(monkeypatch, tmp_path):
    def build(self):
        self.device = "cpu"
        self.model = TinyLM()
        self.model.eval()
        self.ref_model = TinyLM()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self._wandb = None
        self.loader = [{"chosen_input_ids": [[1, 2, 3]], "chosen_labels": [[-100, 2, 3]], "rejected_input_ids": [[1, 3, 4]], "rejected_labels": [[-100, 3, 4]]}] * 3
    monkeypatch.setattr(DPOTrainer, "_build", build)
    monkeypatch.setattr(DPOTrainer, "_save", lambda self: None)
    rows = []
    trainer = DPOTrainer(DPOConfig(output_dir=str(tmp_path), max_steps=2, grad_accum_steps=1), on_step=rows.append)
    trainer.fit()
    assert trainer.completed_steps == 2
    assert trainer.model.training
    assert [r["step"] for r in rows] == [1, 2]
    assert "preference_accuracy" in rows[-1]
