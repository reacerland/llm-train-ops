"""Append-only JSONL dataset cache with atomic checkpoint metadata."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class CacheCorruptedError(Exception):
    """Raised when data.jsonl and meta.json disagree in an unrecoverable way."""


@dataclass
class CacheMeta:
    dataset: str
    subset: str
    split: str
    records: int = 0
    bytes: int = 0
    updated_at: str = ""
    name: str | None = None


def dataset_slug(dataset: str) -> str:
    """'AI-ModelScope/xxx' -> 'AI-ModelScope__xxx' (safe directory name)."""
    return dataset.replace("/", "__")


class DatasetCache:
    """One cache entry keyed by (dataset, subset, split)."""

    def __init__(self, cache_dir: Path, dataset: str, subset: str, split: str,
                 name: str | None = None):
        key = name or dataset_slug(dataset)
        self.dir = Path(cache_dir) / "datasets" / key / subset / split
        self.data_path = self.dir / "data.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.dataset = dataset
        self.subset = subset
        self.split = split
        self.name = name

    def load_meta(self) -> CacheMeta:
        """Load checkpoint metadata; returns an empty meta when none exists."""
        if not self.meta_path.exists():
            return CacheMeta(self.dataset, self.subset, self.split, name=self.name)
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return CacheMeta(**data)

    def save_meta(self, meta: CacheMeta) -> None:
        """Atomically persist checkpoint metadata (temp file + rename)."""
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)
        os.replace(tmp, self.meta_path)

    def append(self, line: str) -> int:
        """Append one JSONL line to data.jsonl; returns the UTF-8 byte count written."""
        self.dir.mkdir(parents=True, exist_ok=True)
        data = line.encode("utf-8")
        with open(self.data_path, "ab") as f:
            f.write(data)
        return len(data)

    def ensure_consistent(self, meta: CacheMeta) -> None:
        """Reconcile data.jsonl with meta before a run.

        - File bigger than meta (crashed between checkpoints): truncate to meta.bytes.
        - File smaller than meta, or one of them missing while the other has data: raise.
        """
        if not self.data_path.exists():
            if meta.records > 0:
                raise CacheCorruptedError(
                    f"meta records {meta.records} rows but {self.data_path} is missing; "
                    f"remove {self.dir} and retry"
                )
            return
        if not self.meta_path.exists():
            raise CacheCorruptedError(
                f"{self.data_path} exists without meta.json; remove {self.dir} and retry"
            )
        actual = self.data_path.stat().st_size
        if actual > meta.bytes:
            with open(self.data_path, "r+b") as f:
                f.truncate(meta.bytes)
        elif actual < meta.bytes:
            raise CacheCorruptedError(
                f"{self.data_path} has {actual} bytes but meta says {meta.bytes}; "
                f"remove {self.dir} and retry"
            )
