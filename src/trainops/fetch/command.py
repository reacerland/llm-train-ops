"""Fetch orchestration: quota check, incremental download, checkpointing."""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import DatasetCache
from .streamer import stream_dataset

CHECKPOINT_INTERVAL = 1000
DEFAULT_SUBSET = "default"

StreamFn = Callable[[str, str | None, str], Iterator[dict[str, Any]]]


@dataclass
class FetchResult:
    cache_dir: Path
    records: int
    bytes: int
    downloaded: bool
    dataset_exhausted: bool = False


def run_fetch(
    dataset: str,
    subset: str | None,
    split: str,
    max_size: int,
    cache_dir: Path,
    stream_fn: StreamFn = stream_dataset,
) -> FetchResult:
    """Fetch up to max_size bytes of raw records into the local cache.

    - subset=None means "no named subset"; it is cached under the 'default'
      directory and passed to the SDK as None.
    - Cache hit (cached bytes >= max_size): return without downloading.
    - Otherwise resume after meta.records rows and append until the quota is met
      or the dataset is exhausted. Metadata is checkpointed every
      CHECKPOINT_INTERVAL records and always on exit (including exceptions).
    """
    cache = DatasetCache(cache_dir, dataset, subset or DEFAULT_SUBSET, split)
    meta = cache.load_meta()
    cache.ensure_consistent(meta)

    if meta.bytes >= max_size:
        return FetchResult(cache_dir=cache.dir, records=meta.records, bytes=meta.bytes,
                           downloaded=False)

    exhausted = True
    try:
        for idx, row in enumerate(stream_fn(dataset, subset, split)):
            if idx < meta.records:
                continue  # resume: skip already-cached records
            line = json.dumps(row, ensure_ascii=False) + "\n"
            meta.bytes += cache.append(line)
            meta.records += 1
            if meta.records % CHECKPOINT_INTERVAL == 0:
                cache.save_meta(meta)
            if meta.bytes >= max_size:
                exhausted = False
                break
    finally:
        cache.save_meta(meta)

    return FetchResult(cache_dir=cache.dir, records=meta.records, bytes=meta.bytes,
                       downloaded=True, dataset_exhausted=exhausted)
