"""Mixture loading (mix output: tokens.bin + meta.json) and deterministic
batching for the train loop."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_FORMAT = "trainops-mix/v1"
DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class MixtureNotFoundError(Exception):
    """Raised when the mixture directory (mix output) is missing."""


@dataclass
class Mixture:
    sequences: np.memmap     # shape (num_sequences, max_seq_len)
    vocab_size: int
    max_seq_len: int
    num_sequences: int


def load_mixture(directory: Path) -> Mixture:
    """Load a mix output directory (tokens.bin + meta.json) via memmap."""
    directory = Path(directory)
    tokens_path = directory / "tokens.bin"
    meta_path = directory / "meta.json"
    if not tokens_path.is_file() or not meta_path.is_file():
        raise MixtureNotFoundError(
            f"mixture not found at {directory} (need tokens.bin + meta.json); "
            f"run mix first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != META_FORMAT:
        raise ValueError(
            f"{meta_path}: unsupported format {meta.get('format')!r}, "
            f"expected {META_FORMAT!r}")
    dtype = DTYPES[meta["dtype"]]
    max_seq_len = meta["max_seq_len"]
    flat = np.memmap(tokens_path, dtype=dtype, mode="r")
    if flat.size % max_seq_len != 0:
        raise ValueError(
            f"{tokens_path}: {flat.size} tokens is not a multiple of "
            f"max_seq_len={max_seq_len}")
    sequences = flat.reshape(-1, max_seq_len)
    return Mixture(sequences=sequences, vocab_size=meta["vocab_size"],
                   max_seq_len=max_seq_len, num_sequences=sequences.shape[0])


class Batcher:
    """Seeded shuffle-over-epochs batcher.

    Each epoch iterates a fresh rng permutation of the sequence indices;
    exhausting one epoch starts the next (multi-epoch when total_tokens
    exceeds the dataset). State is fully serializable for checkpoint resume.
    """

    def __init__(self, mixture: Mixture, batch_size: int, seed: int):
        self.mixture = mixture
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(mixture.num_sequences)
        self.pos = 0

    def next_batch(self) -> np.ndarray:
        n = self.mixture.num_sequences
        idx = np.empty(self.batch_size, dtype=np.int64)
        taken = 0
        while taken < self.batch_size:
            if self.pos >= n:  # epoch exhausted: reshuffle and start over
                self.order = self.rng.permutation(n)
                self.pos = 0
            take = min(self.batch_size - taken, n - self.pos)
            idx[taken:taken + take] = self.order[self.pos:self.pos + take]
            self.pos += take
            taken += take
        return np.asarray(self.mixture.sequences[idx], dtype=np.int64)

    def state_dict(self) -> dict:
        return {
            "order": self.order.tolist(),
            "pos": self.pos,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.order = np.asarray(state["order"])
        self.pos = state["pos"]
        self.rng.bit_generator.state = state["rng"]
