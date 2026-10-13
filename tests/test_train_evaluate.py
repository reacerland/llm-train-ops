import json
import math

import numpy as np
import pytest
import torch

from trainops.train.data import load_mixture
from trainops.train.evaluate import evaluate_split


class TinyModel(torch.nn.Module):
    """鸭子类型模型： embedding + linear, forward(input_ids) -> logits."""

    def __init__(self, vocab=50, d_model=32, seq_len=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d_model)
        self.pos = torch.nn.Embedding(seq_len, d_model)
        self.head = torch.nn.Linear(d_model, vocab)

    def forward(self, input_ids):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.head(self.emb(input_ids) + self.pos(pos))


def write_mixture(tmp_path, num_seqs=64, seq_len=16, vocab=50):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, vocab, size=(num_seqs, seq_len))
    (tmp_path / "tokens.bin").write_bytes(seqs.astype(np.uint16).tobytes())
    import json
    (tmp_path / "meta.json").write_text(json.dumps(
        {"format": "trainops-mix/v1", "dtype": "uint16",
         "max_seq_len": seq_len, "vocab_size": vocab,
         "num_sequences": num_seqs}), encoding="utf-8")
    return tmp_path


def make_model(vocab=50):
    torch.manual_seed(0)
    return TinyModel(vocab=vocab)


def test_evaluate_metrics_shape_and_ranges(tmp_path):
    mixture = load_mixture(write_mixture(tmp_path / "mix"))
    model = make_model()
    m = evaluate_split(model, mixture, batch_size=4, num_batches=3,
                       seed=123, device=torch.device("cpu"), use_bf16=False)
    assert set(m) == {"loss", "ppl", "top1", "top5", "tokens"}
    assert m["tokens"] == 3 * 4 * (16 - 1)     # 每 batch 的有效预测位置数
    assert m["loss"] > 0
    assert m["ppl"] == pytest.approx(math.exp(m["loss"]), rel=1e-6)
    assert 0.0 <= m["top1"] <= m["top5"] <= 1.0


def test_evaluate_deterministic_same_seed(tmp_path):
    mixture = load_mixture(write_mixture(tmp_path / "mix"))
    model = make_model()
    m1 = evaluate_split(model, mixture, 4, 3, seed=123,
                        device=torch.device("cpu"), use_bf16=False)
    m2 = evaluate_split(model, mixture, 4, 3, seed=123,
                        device=torch.device("cpu"), use_bf16=False)
    assert m1 == m2


def test_evaluate_preserves_training_mode(tmp_path):
    mixture = load_mixture(write_mixture(tmp_path / "mix"))
    model = make_model()
    model.train()
    evaluate_split(model, mixture, 4, 2, seed=1,
                   device=torch.device("cpu"), use_bf16=False)
    assert model.training
    model.eval()
    evaluate_split(model, mixture, 4, 2, seed=1,
                   device=torch.device("cpu"), use_bf16=False)
    assert not model.training


def test_evaluate_does_not_touch_gradients(tmp_path):
    mixture = load_mixture(write_mixture(tmp_path / "mix"))
    model = make_model()
    evaluate_split(model, mixture, 4, 2, seed=1,
                   device=torch.device("cpu"), use_bf16=False)
    assert all(p.grad is None for p in model.parameters())
