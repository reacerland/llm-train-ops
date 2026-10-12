"""Convert orchestration: cache JSONL -> HF arrow dataset directory."""

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm

from trainops.fetch.cache import DatasetCache

from .transform import MissingFieldError, extract_fields

BATCH_SIZE = 1000
PROGRESS_INTERVAL = 1000  # records between non-TTY progress lines
DEFAULT_SUBSET = "default"
ARROW_DIR_NAME = "data.arrow"


class CacheNotFoundError(Exception):
    """Raised when the source cache has no data.jsonl."""


class OutputExistsError(Exception):
    """Raised when the output directory exists and overwrite is False."""


@dataclass
class ConvertResult:
    output_dir: Path
    records: int
    skipped: int
    bytes: int


def run_convert(
    dataset: str,
    subset: str | None,
    split: str,
    fields: list[str],
    cache_dir: Path,
    output: Path | None = None,
    skip_bad: bool = False,
    overwrite: bool = False,
    name: str | None = None,
    progress: bool = True,
) -> ConvertResult:
    """Convert a cached dataset into an HF arrow directory (save_to_disk format).

    - Streams the cache in batches of BATCH_SIZE records, writing Arrow
      RecordBatches to a temporary IPC file, so memory stays bounded for
      GB-scale caches. The final Dataset is memory-mapped from that file and
      saved to disk; the temp file is removed afterwards.
    - Missing fields (and invalid JSON lines) raise by default; skip_bad=True
      skips them and counts them in ConvertResult.skipped.
    - The output directory must not exist unless overwrite=True.
    """
    if not fields:
        raise ValueError("fields must be a non-empty list")

    cache = DatasetCache(cache_dir, dataset, subset or DEFAULT_SUBSET, split, name=name)
    if not cache.data_path.exists():
        raise CacheNotFoundError(
            f"cache not found for dataset '{dataset}' (subset={subset or DEFAULT_SUBSET}, "
            f"split={split}) at {cache.dir}; run fetch first"
        )
    cache.ensure_consistent(cache.load_meta())

    output_dir = Path(output) if output else cache.dir / ARROW_DIR_NAME
    if output_dir.exists() and not overwrite:
        raise OutputExistsError(f"output directory already exists: {output_dir}")

    from datasets import Dataset  # lazy: heavy import

    schema = pa.schema([pa.field(f, pa.string()) for f in fields])
    fd, tmp_name = tempfile.mkstemp(suffix=".arrow")
    Path(tmp_name).unlink(missing_ok=True)  # writer creates the file itself
    tmp_path = Path(tmp_name)
    records = 0
    skipped = 0
    read_bytes = 0
    total_bytes = cache.data_path.stat().st_size
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"convert {dataset}", disable=None if progress else True)
    try:
        with pa.RecordBatchStreamWriter(str(tmp_path), schema) as writer:
            batch_rows: list[dict] = []

            def flush() -> None:
                if batch_rows:
                    writer.write_table(pa.Table.from_pylist(batch_rows, schema=schema))
                    batch_rows.clear()

            for idx, (line_bytes, record) in enumerate(cache.iter_records(), start=1):
                read_bytes += line_bytes
                pbar.update(line_bytes)
                try:
                    if not isinstance(record, dict):
                        raise ValueError(f"record {idx} is not a JSON object")
                    batch_rows.append(extract_fields(record, fields, record_index=idx))
                    records += 1
                except (MissingFieldError, ValueError) as exc:
                    if not skip_bad:
                        raise
                    skipped += 1
                if records % PROGRESS_INTERVAL == 0:
                    if progress and pbar.disable:  # non-TTY fallback
                        print(f"progress: {read_bytes}/{total_bytes} bytes "
                              f"({read_bytes / max(total_bytes, 1):.0%}), "
                              f"{records} records", file=sys.stderr)
                if len(batch_rows) >= BATCH_SIZE:
                    flush()
            flush()
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        pbar.close()

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        ds = Dataset.from_file(str(tmp_path))
        ds.save_to_disk(str(output_dir))
    finally:
        tmp_path.unlink(missing_ok=True)

    return ConvertResult(output_dir=output_dir, records=records, skipped=skipped,
                         bytes=_dir_size(output_dir))


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
