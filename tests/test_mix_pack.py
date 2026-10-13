import numpy as np
import pytest

from trainops.mix.pack import pack


def run_pack(docs_by_source, weights, seq_len, seed=42):
    streams = {name: iter(docs) for name, docs in docs_by_source.items()}
    collector: list[list[int]] = []
    result = pack(streams, weights, seq_len,
                  rng=np.random.default_rng(seed), sink=collector.append)
    return result, collector


def single_token_docs(n, tag):
    return [[f"{tag}{i}"] for i in range(n)]


def test_deterministic_same_seed():
    docs = {"A": single_token_docs(50, "a"), "B": single_token_docs(50, "b")}
    _, runs1 = run_pack(docs, {"A": 0.5, "B": 0.5}, seq_len=1, seed=7)
    _, runs2 = run_pack(docs, {"A": 0.5, "B": 0.5}, seq_len=1, seed=7)
    assert runs1 == runs2
    assert runs1  # 确实产出了序列


def test_weighted_distribution_approx():
    docs = {"A": single_token_docs(2000, "a"), "B": single_token_docs(2000, "b")}
    result, _ = run_pack(docs, {"A": 0.75, "B": 0.25}, seq_len=1)
    a = result.per_source["A"].docs_sampled
    b = result.per_source["B"].docs_sampled
    assert a / (a + b) == pytest.approx(0.75, abs=0.05)


def test_exhaustion_stops_and_reports_source():
    docs = {"A": single_token_docs(100, "a"), "B": single_token_docs(3, "b")}
    result, collector = run_pack(docs, {"A": 0.5, "B": 0.5}, seq_len=1)
    assert result.exhausted_sources == ["B"]
    assert result.per_source["B"].docs_sampled == 3  # 不放回：B 最多被抽 3 次
    assert result.per_source["A"].docs_sampled < 100  # 抽干即停，A 没抽完


def test_tail_dropped_and_counted():
    result, collector = run_pack({"A": [["a"] * 12]}, {"A": 1.0}, seq_len=10)
    assert result.sequences == 1
    assert result.tokens_written == 10
    assert result.dropped_tail_tokens == 2
    assert collector == [["a"] * 10]


def test_long_doc_spans_sequences():
    doc = list(range(25))
    result, collector = run_pack({"A": [doc]}, {"A": 1.0}, seq_len=10)
    assert result.sequences == 2
    assert collector == [list(range(10)), list(range(10, 20))]
    assert result.dropped_tail_tokens == 5


def test_flush_full_sequences_on_stop():
    docs = {"A": [["a"] * 7, ["b"] * 8]}
    result, collector = run_pack(docs, {"A": 1.0}, seq_len=10)
    # 停止时 buffer 里仍有 15 token：先冲刷出 1 条完整序列，再丢 5
    assert result.sequences == 1
    assert collector == [["a"] * 7 + ["b"] * 3]
    assert result.dropped_tail_tokens == 5


def test_unnormalized_weights_equivalent():
    docs = {"A": single_token_docs(2000, "a"), "B": single_token_docs(2000, "b")}
    result, _ = run_pack(docs, {"A": 75, "B": 25}, seq_len=1)
    a = result.per_source["A"].docs_sampled
    b = result.per_source["B"].docs_sampled
    assert a / (a + b) == pytest.approx(0.75, abs=0.05)
