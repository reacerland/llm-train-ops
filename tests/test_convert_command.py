import json

import pytest
from datasets import load_from_disk

from trainops.convert.command import (
    CacheNotFoundError,
    ConvertResult,
    OutputExistsError,
    run_convert,
)
from trainops.convert.transform import MissingFieldError
from trainops.fetch.cache import CacheMeta, DatasetCache


def seed_cache(tmp_path, rows, dataset="org/ds", subset="default", split="train",
               name=None):
    """Write rows as a fetch-style cache entry; returns the DatasetCache."""
    cache = DatasetCache(tmp_path, dataset, subset, split, name=name)
    cache.dir.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    cache.data_path.write_text(data, encoding="utf-8")
    cache.save_meta(CacheMeta(dataset=dataset, subset=subset, split=split,
                              records=len(rows), bytes=len(data.encode("utf-8")),
                              name=name))
    return cache


ROWS = [{"text": f"row {i}", "meta": {"id": i}, "title": f"t{i}"} for i in range(5)]


def test_convert_writes_arrow_dir_next_to_cache(tmp_path):
    cache = seed_cache(tmp_path, ROWS)
    result = run_convert("org/ds", "default", "train", ["text", "meta"], tmp_path,
                         progress=False)
    assert result.output_dir == cache.dir / "data.arrow"
    assert result.records == 5
    assert result.skipped == 0
    assert result.output_dir.is_dir()
    ds = load_from_disk(str(result.output_dir))
    assert ds.column_names == ["text", "meta"]
    assert ds.num_rows == 5
    assert ds[0]["text"] == "row 0"
    assert ds[0]["meta"] == '{"id": 0}'  # non-str value -> JSON string
    assert "title" not in ds.column_names  # whitelist projection


def test_convert_keeps_nulls_and_loads_from_disk(tmp_path):
    rows = [{"text": "a", "opt": None}, {"text": "b", "opt": "x"}]
    seed_cache(tmp_path, rows)
    result = run_convert("org/ds", "default", "train", ["text", "opt"], tmp_path,
                         progress=False)
    ds = load_from_disk(str(result.output_dir))
    assert ds["opt"] == [None, "x"]


def test_convert_name_alias_locates_cache(tmp_path):
    seed_cache(tmp_path, ROWS, name="my-ds")
    result = run_convert("org/ds", "default", "train", ["text"], tmp_path,
                         name="my-ds", progress=False)
    assert result.output_dir == tmp_path / "datasets" / "my-ds" / "default" / "train" / "data.arrow"
    assert load_from_disk(str(result.output_dir)).num_rows == 5


def test_convert_missing_field_raises_by_default(tmp_path):
    seed_cache(tmp_path, [{"text": "a"}, {"text": "b", "title": "x"}])
    with pytest.raises(MissingFieldError, match="record 1"):
        run_convert("org/ds", "default", "train", ["text", "title"], tmp_path,
                    progress=False)


def test_convert_skip_bad_counts_and_keeps_good_rows(tmp_path):
    rows = [{"text": "a"}, {"text": "b", "title": "x"}, {"text": "c"}]
    seed_cache(tmp_path, rows)
    result = run_convert("org/ds", "default", "train", ["text", "title"], tmp_path,
                         skip_bad=True, progress=False)
    assert result.records == 1
    assert result.skipped == 2
    ds = load_from_disk(str(result.output_dir))
    assert ds["text"] == ["b"]


def _rewrite_data(cache, data):
    """Overwrite data.jsonl and keep meta.json consistent (ensure_consistent-safe)."""
    cache.data_path.write_text(data, encoding="utf-8")
    cache.save_meta(CacheMeta(dataset=cache.dataset, subset=cache.subset,
                              split=cache.split, records=len(data.splitlines()),
                              bytes=len(data.encode("utf-8")), name=cache.name))


def test_convert_invalid_json_line_raises_with_context(tmp_path):
    cache = seed_cache(tmp_path, [])
    _rewrite_data(cache, '{"text": "a"}\nnot-json\n{"text": "b"}\n')
    with pytest.raises(ValueError, match=":2:"):
        run_convert("org/ds", "default", "train", ["text"], tmp_path, progress=False)


def test_convert_invalid_json_line_skip_bad(tmp_path):
    cache = seed_cache(tmp_path, [])
    _rewrite_data(cache, '{"text": "a"}\nnot-json\n{"text": "b"}\n')
    result = run_convert("org/ds", "default", "train", ["text"], tmp_path,
                         skip_bad=True, progress=False)
    assert result.records == 2
    assert result.skipped == 1


def test_convert_cache_not_found(tmp_path):
    with pytest.raises(CacheNotFoundError, match="run fetch first"):
        run_convert("org/ds", "default", "train", ["text"], tmp_path, progress=False)


def test_convert_corrupted_cache_raises(tmp_path):
    """data.jsonl smaller than meta.bytes: corruption must surface, not convert garbage."""
    from trainops.fetch.cache import CacheCorruptedError

    cache = seed_cache(tmp_path, [{"text": "a"}])
    cache.save_meta(CacheMeta(dataset="org/ds", subset="default", split="train",
                              records=99, bytes=10**6))
    with pytest.raises(CacheCorruptedError):
        run_convert("org/ds", "default", "train", ["text"], tmp_path, progress=False)


def test_convert_output_exists_requires_overwrite(tmp_path):
    seed_cache(tmp_path, ROWS)
    run_convert("org/ds", "default", "train", ["text"], tmp_path, progress=False)
    with pytest.raises(OutputExistsError):
        run_convert("org/ds", "default", "train", ["text"], tmp_path, progress=False)
    # --overwrite rebuilds cleanly
    result = run_convert("org/ds", "default", "train", ["text"], tmp_path,
                         overwrite=True, progress=False)
    assert result.records == 5
    assert load_from_disk(str(result.output_dir)).num_rows == 5


def test_convert_empty_fields_rejected(tmp_path):
    seed_cache(tmp_path, ROWS)
    with pytest.raises(ValueError, match="non-empty"):
        run_convert("org/ds", "default", "train", [], tmp_path, progress=False)


def test_convert_empty_cache_produces_empty_dataset(tmp_path):
    seed_cache(tmp_path, [])
    result = run_convert("org/ds", "default", "train", ["text"], tmp_path,
                         progress=False)
    assert result.records == 0
    ds = load_from_disk(str(result.output_dir))
    assert ds.num_rows == 0
    assert ds.column_names == ["text"]
