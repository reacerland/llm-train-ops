import pytest

from trainops.fetch.cache import CacheMeta, DatasetCache, dataset_slug


def test_dataset_slug():
    assert dataset_slug("AI-ModelScope/xxx") == "AI-ModelScope__xxx"


def test_cache_dir_layout(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "main", "test")
    assert cache.dir == tmp_path / "datasets" / "org__ds" / "main" / "test"
    assert cache.data_path == cache.dir / "data.jsonl"
    assert cache.meta_path == cache.dir / "meta.json"


def test_load_meta_missing_returns_empty(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    meta = cache.load_meta()
    assert meta.dataset == "org/ds"
    assert meta.records == 0
    assert meta.bytes == 0


def test_save_and_load_meta_roundtrip(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.save_meta(CacheMeta(dataset="org/ds", subset="default", split="train", records=10, bytes=2048))
    loaded = cache.load_meta()
    assert loaded.records == 10
    assert loaded.bytes == 2048
    assert loaded.updated_at != ""


def test_name_overrides_cache_dir(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train", name="my-ds")
    assert cache.dir == tmp_path / "datasets" / "my-ds" / "default" / "train"


def test_meta_roundtrip_with_name(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train", name="my-ds")
    cache.save_meta(CacheMeta(dataset="org/ds", subset="default", split="train", name="my-ds"))
    loaded = cache.load_meta()
    assert loaded.name == "my-ds"
    assert loaded.dataset == "org/ds"


def test_load_meta_without_name_field_is_compatible(tmp_path):
    """meta.json written by older versions (no 'name' field) still loads."""
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    cache.meta_path.write_text(
        '{"dataset": "org/ds", "subset": "default", "split": "train", '
        '"records": 3, "bytes": 30, "updated_at": "x"}')
    loaded = cache.load_meta()
    assert loaded.name is None
    assert loaded.records == 3


def test_iter_records_yields_line_bytes_records_and_errors(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    line1 = '{"text": "hi"}\n'
    cache.data_path.write_text(line1 + '{"text": "中文"}\n', encoding="utf-8")
    items = list(cache.iter_records())
    assert items[0] == (len(line1.encode("utf-8")), {"text": "hi"}, None)
    assert items[1] == (len('{"text": "中文"}\n'.encode("utf-8")), {"text": "中文"}, None)


def test_iter_records_missing_file_raises(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    with pytest.raises(FileNotFoundError):
        list(cache.iter_records())


def test_iter_records_invalid_json_yields_error_and_continues(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    cache.data_path.write_text('{"a": 1}\nnot-json\n{"a": 2}\n', encoding="utf-8")
    items = list(cache.iter_records())
    assert len(items) == 3
    assert items[1][1] is None
    assert ":2:" in items[1][2]
    assert items[2][1] == {"a": 2}


def test_iter_records_skips_blank_lines(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    cache.data_path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    items = list(cache.iter_records())
    assert [record for _, record, _ in items] == [{"a": 1}, {"a": 2}]
