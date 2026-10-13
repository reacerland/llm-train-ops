import json

import numpy as np
import pytest

from trainops.train.data import (Batcher, MixtureNotFoundError, load_mixture)


def write_mixture(tmp_path, num_seqs=6, seq_len=4, vocab_size=100,
                  dtype="uint16"):
    seqs = np.arange(num_seqs * seq_len, dtype=np.uint16).reshape(num_seqs,
                                                                  seq_len)
    seqs = (seqs % vocab_size).astype(dtype)
    (tmp_path / "tokens.bin").write_bytes(seqs.tobytes())
    meta = {"format": "trainops-mix/v1", "dtype": dtype,
            "max_seq_len": seq_len, "vocab_size": vocab_size,
            "num_sequences": num_seqs}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_load_mixture_ok(tmp_path):
    m = load_mixture(write_mixture(tmp_path))
    assert m.sequences.shape == (6, 4)
    assert m.vocab_size == 100
    assert m.max_seq_len == 4
    assert m.num_sequences == 6


def test_load_mixture_missing(tmp_path):
    with pytest.raises(MixtureNotFoundError, match="run mix first"):
        load_mixture(tmp_path / "nope")


def test_load_mixture_missing_tokens(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps(
        {"format": "trainops-mix/v1", "dtype": "uint16", "max_seq_len": 4,
         "vocab_size": 100}), encoding="utf-8")
    with pytest.raises(MixtureNotFoundError, match="run mix first"):
        load_mixture(tmp_path)


def test_load_mixture_empty(tmp_path):
    write_mixture(tmp_path, num_seqs=0)
    with pytest.raises(ValueError, match="no sequences"):
        load_mixture(tmp_path)


def test_load_mixture_bad_format(tmp_path):
    write_mixture(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["format"] = "something-else"
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        load_mixture(tmp_path)


def test_next_batch_dtype_and_shape(tmp_path):
    m = load_mixture(write_mixture(tmp_path))
    b = Batcher(m, batch_size=2, seed=42)
    batch = b.next_batch()
    assert batch.shape == (2, 4)
    assert batch.dtype == np.int64


def test_batcher_deterministic(tmp_path):
    m = load_mixture(write_mixture(tmp_path))
    a, b = Batcher(m, 2, 42), Batcher(m, 2, 42)
    for _ in range(5):
        np.testing.assert_array_equal(a.next_batch(), b.next_batch())


def test_batcher_epoch_wrap_covers_all(tmp_path):
    m = load_mixture(write_mixture(tmp_path, num_seqs=6))
    b = Batcher(m, batch_size=2, seed=42)
    seen = set()
    for _ in range(3):  # one full epoch: 6 seqs / batch 2
        for row in b.next_batch():
            seen.add(tuple(row))
    assert len(seen) == 6
    # 4th batch starts a new epoch (data repeats, no crash)
    b.next_batch()


def test_batcher_state_roundtrip(tmp_path):
    m = load_mixture(write_mixture(tmp_path))
    orig = Batcher(m, 2, 42)
    orig.next_batch()
    orig.next_batch()
    state = orig.state_dict()

    restored = Batcher(m, 2, 999)  # different seed on purpose
    restored.load_state_dict(state)
    for _ in range(5):
        np.testing.assert_array_equal(orig.next_batch(),
                                      restored.next_batch())
