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
