import pytest

from trainops.fetch.cache import CacheCorruptedError, CacheMeta, DatasetCache


def _make_cache(tmp_path):
    return DatasetCache(tmp_path, "org/ds", "default", "train")


def test_append_returns_utf8_byte_count(tmp_path):
    cache = _make_cache(tmp_path)
    n = cache.append('{"text": "你好"}\n')
    assert n == len('{"text": "你好"}\n'.encode("utf-8"))
    assert cache.data_path.read_bytes() == '{"text": "你好"}\n'.encode("utf-8")


def test_ensure_consistent_ok_when_fully_empty(tmp_path):
    cache = _make_cache(tmp_path)
    cache.ensure_consistent(cache.load_meta())  # no data, no meta -> fine


def test_ensure_consistent_truncates_extra_bytes(tmp_path):
    """Crash after appending but before checkpoint: file bigger than meta -> truncate."""
    cache = _make_cache(tmp_path)
    cache.append('{"a": 1}\n')
    cache.save_meta(CacheMeta("org/ds", "default", "train", records=1, bytes=9))
    cache.append('{"a": 2}\n')  # written but never checkpointed
    cache.ensure_consistent(cache.load_meta())
    assert cache.data_path.read_bytes() == b'{"a": 1}\n'


def test_ensure_consistent_raises_when_file_smaller_than_meta(tmp_path):
    cache = _make_cache(tmp_path)
    cache.append('{"a": 1}\n')
    cache.save_meta(CacheMeta("org/ds", "default", "train", records=5, bytes=100))
    with pytest.raises(CacheCorruptedError):
        cache.ensure_consistent(cache.load_meta())


def test_ensure_consistent_raises_when_data_without_meta(tmp_path):
    cache = _make_cache(tmp_path)
    cache.append('{"a": 1}\n')  # data file exists, meta file never written
    with pytest.raises(CacheCorruptedError):
        cache.ensure_consistent(cache.load_meta())


def test_ensure_consistent_raises_when_meta_without_data(tmp_path):
    cache = _make_cache(tmp_path)
    cache.save_meta(CacheMeta("org/ds", "default", "train", records=5, bytes=100))
    with pytest.raises(CacheCorruptedError):
        cache.ensure_consistent(cache.load_meta())
