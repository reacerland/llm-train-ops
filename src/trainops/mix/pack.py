"""Packing engine: weighted document-level sampling without replacement.

Pure logic — streams, weights, seq_len, rng and sink are all injected, so
the whole thing is testable in memory. The only nondeterminism is the
injected ``numpy.random.Generator``: same seed, same streams, same weights
reproduce the exact same byte stream.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourcePackStats:
    docs_sampled: int = 0
    tokens: int = 0


@dataclass
class PackResult:
    sequences: int = 0
    tokens_written: int = 0
    dropped_tail_tokens: int = 0
    exhausted_sources: list[str] = field(default_factory=list)
    per_source: dict[str, SourcePackStats] = field(default_factory=dict)


def pack(
    streams: Mapping[str, Iterator[Sequence[int]]],
    weights: Mapping[str, float],
    seq_len: int,
    rng: np.random.Generator,
    sink: Callable[[list[int]], None],
) -> PackResult:
    """Draw documents from ``streams`` by fixed ``weights`` (no replacement
    within a source), concatenate into a continuous token stream, and cut it
    into ``seq_len``-sized sequences passed to ``sink``.

    Stops the first time a chosen source has no document left; full
    sequences still buffered are flushed, the incomplete tail is dropped
    and counted.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    names = list(streams)  # fixed order: determinism does not depend on dict order
    probs = np.array([float(weights[n]) for n in names], dtype=float)
    if (probs <= 0).any():
        raise ValueError("all weights must be positive")
    probs /= probs.sum()

    result = PackResult(per_source={n: SourcePackStats() for n in names})
    buffer: list[int] = []

    def flush_full() -> None:
        while len(buffer) >= seq_len:
            sink(buffer[:seq_len])
            del buffer[:seq_len]
            result.sequences += 1
            result.tokens_written += seq_len

    while True:
        name = names[rng.choice(len(names), p=probs)]
        try:
            doc = next(streams[name])
        except StopIteration:
            result.exhausted_sources.append(name)
            break
        buffer.extend(doc)
        stats = result.per_source[name]
        stats.docs_sampled += 1
        stats.tokens += len(doc)
        flush_full()
    flush_full()  # stop condition: flush what's still complete
    result.dropped_tail_tokens = len(buffer)
    return result
