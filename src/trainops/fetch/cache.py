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


def dataset_slug(dataset: str) -> str:
    """'AI-ModelScope/xxx' -> 'AI-ModelScope__xxx' (safe directory name)."""
    return dataset.replace("/", "__")


class DatasetCache:
    """One cache entry keyed by (dataset, subset, split)."""

    def __init__(self, cache_dir: Path, dataset: str, subset: str, split: str):
        self.dir = Path(cache_dir) / "datasets" / dataset_slug(dataset) / subset / split
        self.data_path = self.dir / "data.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.dataset = dataset
        self.subset = subset
        self.split = split

    def load_meta(self) -> CacheMeta:
        """Load checkpoint metadata; returns an empty meta when none exists."""
        if not self.meta_path.exists():
            return CacheMeta(self.dataset, self.subset, self.split)
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
