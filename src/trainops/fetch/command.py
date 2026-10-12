"""Fetch orchestration: quota check, incremental download, checkpointing."""

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
    name: str | None = None,
    progress: bool = True,
) -> FetchResult:
    """Fetch up to max_size bytes of raw records into the local cache.

    - subset=None means "no named subset"; it is cached under the 'default'
      directory and passed to the SDK as None.
    - name overrides the cache key (default: dataset slug).
    - progress=True shows a tqdm byte progress bar on TTY; in non-TTY
      environments a plain progress line is printed at each checkpoint.
    - Cache hit (cached bytes >= max_size): return without downloading.
    - Otherwise resume after meta.records rows and append until the quota is met
      or the dataset is exhausted. Metadata is checkpointed every
      CHECKPOINT_INTERVAL records and always on exit (including exceptions).
    """
    cache = DatasetCache(cache_dir, dataset, subset or DEFAULT_SUBSET, split, name=name)
    meta = cache.load_meta()
    cache.ensure_consistent(meta)

    if meta.bytes >= max_size:
        return FetchResult(cache_dir=cache.dir, records=meta.records, bytes=meta.bytes,
                           downloaded=False)

    pbar = tqdm(total=max_size, initial=meta.bytes, unit="B", unit_scale=True,
                unit_divisor=1024, desc=f"fetch {dataset}",
                disable=None if progress else True)
    exhausted = True
    stream = stream_fn(dataset, subset, split)
    try:
        for idx, row in enumerate(stream):
            if idx < meta.records:
                continue  # resume: skip already-cached records
            line = json.dumps(row, ensure_ascii=False) + "\n"
            meta.bytes += cache.append(line)
            meta.records += 1
            pbar.update(meta.bytes - pbar.n)
            pbar.set_postfix(records=meta.records)
            if meta.records % CHECKPOINT_INTERVAL == 0:
                cache.save_meta(meta)
                if progress and pbar.disable:  # non-TTY fallback
                    print(f"progress: {meta.bytes}/{max_size} bytes "
                          f"({meta.bytes / max_size:.0%}), {meta.records} records",
                          file=sys.stderr)
            if meta.bytes >= max_size:
                exhausted = False
                break
    finally:
        close = getattr(stream, "close", None)  # abort the HTTP/file read promptly
        if callable(close):
            close()
        pbar.close()
        cache.save_meta(meta)

    return FetchResult(cache_dir=cache.dir, records=meta.records, bytes=meta.bytes,
                       downloaded=True, dataset_exhausted=exhausted)
