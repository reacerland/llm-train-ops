import json

import pytest

from trainops.fetch.command import run_fetch

ROWS = [{"idx": i, "text": f"row {i}"} for i in range(10)]
ROW_BYTES = len((json.dumps(ROWS[0], ensure_ascii=False) + "\n").encode("utf-8"))
ALL_BYTES = ROW_BYTES * len(ROWS)


def fake_stream_factory(rows):
    def stream_fn(dataset, subset, split):
        yield from rows

    return stream_fn


def _data_file(tmp_path):
    return tmp_path / "datasets" / "org__ds" / "default" / "train" / "data.jsonl"


def test_first_fetch_writes_until_target(tmp_path):
    result = run_fetch("org/ds", "default", "train", ALL_BYTES, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.downloaded is True
    assert result.records == 10
    assert result.bytes == ALL_BYTES
    assert result.dataset_exhausted is False
    lines = _data_file(tmp_path).read_text().splitlines()
    assert len(lines) == 10
    assert json.loads(lines[0])["idx"] == 0


def test_first_fetch_stops_at_target_mid_stream(tmp_path):
    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 3, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.records == 3
    assert result.bytes == ROW_BYTES * 3
    assert result.dataset_exhausted is False


def test_cache_hit_returns_without_downloading(tmp_path):
    first = run_fetch("org/ds", "default", "train", 10**9, tmp_path,
                      stream_fn=fake_stream_factory(ROWS))
    assert first.dataset_exhausted is True

    def exploding_stream(dataset, subset, split):
        raise AssertionError("stream_fn must not be called on cache hit")

    second = run_fetch("org/ds", "default", "train", first.bytes, tmp_path,
                       stream_fn=exploding_stream)
    assert second.downloaded is False
    assert second.bytes == first.bytes
    assert second.records == 10


def test_incremental_fetch_appends_only_delta(tmp_path):
    run_fetch("org/ds", "default", "train", ROW_BYTES * 3, tmp_path,
              stream_fn=fake_stream_factory(ROWS))
    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 5, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.records == 5
    lines = _data_file(tmp_path).read_text().splitlines()
    assert [json.loads(line)["idx"] for line in lines] == [0, 1, 2, 3, 4]


def test_resume_after_interruption(tmp_path):
    state = {"calls": 0}

    def flaky_stream(dataset, subset, split):
        state["calls"] += 1
        for i, row in enumerate(ROWS):
            if state["calls"] == 1 and i == 4:
                raise ConnectionError("network down")
            yield row

    with pytest.raises(ConnectionError):
        run_fetch("org/ds", "default", "train", ROW_BYTES * 8, tmp_path, stream_fn=flaky_stream)

    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 8, tmp_path, stream_fn=flaky_stream)
    assert result.records == 8
    lines = _data_file(tmp_path).read_text().splitlines()
    assert [json.loads(line)["idx"] for line in lines] == list(range(8))


def test_dataset_exhausted_before_target(tmp_path):
    result = run_fetch("org/ds", "default", "train", 10**9, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.dataset_exhausted is True
    assert result.records == 10
    assert result.bytes == ALL_BYTES


def test_subset_none_uses_default_cache_dir(tmp_path):
    result = run_fetch("org/ds", None, "train", ROW_BYTES, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.cache_dir == tmp_path / "datasets" / "org__ds" / "default" / "train"
    assert result.records == 1
