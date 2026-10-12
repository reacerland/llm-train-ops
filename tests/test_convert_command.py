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
