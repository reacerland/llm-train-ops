import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trainops.train.checkpoint import load_checkpoint, save_checkpoint
from trainops.train.data import Batcher, load_mixture

FIXTURES_DIR = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
from gpt import GPT  # noqa: E402

ARGS = dict(vocab_size=50, d_model=32, n_layer=2, n_head=2, max_seq_len=16)


def make_mixture(tmp_path, num_seqs=20, seq_len=16):
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, 50, size=(num_seqs, seq_len)).astype(np.uint16)
    (tmp_path / "tokens.bin").write_bytes(seqs.tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"format": "trainops-mix/v1", "dtype": "uint16", "max_seq_len": seq_len,
         "vocab_size": 50, "num_sequences": num_seqs}), encoding="utf-8")
    return load_mixture(tmp_path)


def make_state(mixture, seed=123):
    torch.manual_seed(seed)
    model = GPT(**ARGS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    batcher = Batcher(mixture, batch_size=2, seed=42)
    return {"model": model, "optimizer": opt, "scheduler": sched,
            "batcher": batcher}


def train_steps(state, n):
    for _ in range(n):
        batch = torch.from_numpy(state["batcher"].next_batch())
        logits = state["model"](batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 50),
                               batch[:, 1:].reshape(-1))
        state["optimizer"].zero_grad()
        loss.backward()
        state["optimizer"].step()
        state["scheduler"].step()


def params_equal(a, b):
    return all(torch.equal(pa, pb)
               for pa, pb in zip(a.parameters(), b.parameters()))


def test_checkpoint_roundtrip(tmp_path):
    mixture = make_mixture(tmp_path)
    state = make_state(mixture)
    train_steps(state, 5)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, model=state["model"], optimizer=state["optimizer"],
                    scheduler=state["scheduler"], step=5,
                    batcher=state["batcher"])
    assert not (tmp_path / "checkpoint.pt.tmp").exists()  # no temp residue

    fresh = make_state(mixture, seed=999)
    step = load_checkpoint(ckpt, model=fresh["model"],
                           optimizer=fresh["optimizer"],
                           scheduler=fresh["scheduler"],
                           batcher=fresh["batcher"])
    assert step == 5
    assert params_equal(fresh["model"], state["model"])

    # both continue 3 steps: identical trajectories (fp32 CPU)
    train_steps(state, 3)
    train_steps(fresh, 3)
    assert params_equal(fresh["model"], state["model"])


def test_load_checkpoint_missing(tmp_path):
    mixture = make_mixture(tmp_path)
    state = make_state(mixture)
    with torch.no_grad():
        pass
    import pytest
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", model=state["model"],
                        optimizer=state["optimizer"],
                        scheduler=state["scheduler"], batcher=state["batcher"])


def test_atomic_write_preserves_old_on_tmp_only(tmp_path):
    # a stale tmp file must not affect loading the real checkpoint
    mixture = make_mixture(tmp_path)
    state = make_state(mixture)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, model=state["model"], optimizer=state["optimizer"],
                    scheduler=state["scheduler"], step=3,
                    batcher=state["batcher"])
    (tmp_path / "checkpoint.pt.tmp").write_bytes(b"garbage")
    step = load_checkpoint(ckpt, model=state["model"],
                           optimizer=state["optimizer"],
                           scheduler=state["scheduler"],
                           batcher=state["batcher"])
    assert step == 3
