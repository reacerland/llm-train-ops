# convert 工具（缓存 → HF Arrow）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `trainops convert` 命令，把 fetch 缓存的 JSONL 数据集按字段白名单转换为 HF Arrow 目录（`save_to_disk` 格式），供训练时 `load_from_disk` 直接加载。

**Architecture:** 新增 `trainops/convert/` 包：`transform.py` 为无 IO 纯函数（字段提取 + 类型规范化），`command.py` 为编排层（分批读取缓存 JSONL → 逐批写 Arrow RecordBatch 到临时 IPC 文件 → 内存映射包装为 `Dataset` → `save_to_disk`），`fetch/cache.py` 增加 `iter_records()` 供复用。CLI 入口注册到 `cli.py`，风格与 `fetch` 对称。

**Tech Stack:** Python ≥ 3.10、typer、pyarrow（`RecordBatchFileWriter`）、datasets（`Dataset.from_file`/`save_to_disk`/`load_from_disk`）、pytest。无新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-12-convert-tool-design.md`

**Testing commands:**
- 单个测试：`.venv/bin/pytest tests/test_convert_transform.py::test_name -v`
- 全部：`.venv/bin/pytest -q`

---

### Task 1: transform.py — 纯转换函数

**Files:**
- Create: `src/trainops/convert/__init__.py`（空文件）
- Create: `src/trainops/convert/transform.py`
- Test: `tests/test_convert_transform.py`

- [ ] **Step 1: 创建包目录与空 `__init__.py`**

```bash
mkdir -p src/trainops/convert && touch src/trainops/convert/__init__.py
```

- [ ] **Step 2: 写失败的测试**

创建 `tests/test_convert_transform.py`：

```python
import pytest

from trainops.convert.transform import (
    MissingFieldError,
    extract_fields,
    normalize_value,
    parse_fields,
)


def test_parse_fields_basic():
    assert parse_fields("text, title") == ["text", "title"]


def test_parse_fields_strips_and_drops_empties():
    assert parse_fields(" text ,, ") == ["text"]


def test_parse_fields_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_fields(" , ,")


def test_normalize_value_string_passthrough():
    assert normalize_value("hello") == "hello"


def test_normalize_value_none_passthrough():
    assert normalize_value(None) is None


def test_normalize_value_number_becomes_json_string():
    assert normalize_value(42) == "42"
    assert normalize_value(3.5) == "3.5"


def test_normalize_value_bool_list_dict_become_json_strings():
    assert normalize_value(True) == "true"
    assert normalize_value([1, "二"]) == '[1, "二"]'
    assert normalize_value({"a": 1}) == '{"a": 1}'


def test_normalize_value_unicode_not_escaped():
    assert normalize_value({"t": "中文"}) == '{"t": "中文"}'


def test_extract_fields_projects_and_normalizes():
    record = {"text": "hi", "n": 3, "extra": "x"}
    assert extract_fields(record, ["text", "n"], record_index=0) == {"text": "hi", "n": "3"}


def test_extract_fields_missing_raises_with_index_and_field():
    with pytest.raises(MissingFieldError) as exc_info:
        extract_fields({"a": 1}, ["a", "b"], record_index=7)
    assert exc_info.value.record_index == 7
    assert exc_info.value.field == "b"
    assert "record 7" in str(exc_info.value)
    assert "'b'" in str(exc_info.value)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_convert_transform.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'trainops.convert'`）

- [ ] **Step 4: 实现 transform.py**

创建 `src/trainops/convert/transform.py`：

```python
"""Pure field-extraction and value-normalization for cache -> arrow conversion."""

import json


class MissingFieldError(Exception):
    """Raised when a record lacks a required field."""

    def __init__(self, record_index: int, field: str):
        self.record_index = record_index
        self.field = field
        super().__init__(f"record {record_index}: missing field '{field}'")


def parse_fields(spec: str) -> list[str]:
    """'text, title' -> ['text', 'title']; rejects an empty result."""
    fields = [part.strip() for part in spec.split(",") if part.strip()]
    if not fields:
        raise ValueError("field list is empty")
    return fields


def normalize_value(value):
    """str stays str, None stays None, everything else -> JSON string."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def extract_fields(record: dict, fields: list[str], record_index: int = 0) -> dict:
    """Project a raw record onto the field whitelist, normalizing value types."""
    return {f: normalize_value(_require(record, f, record_index)) for f in fields}


def _require(record: dict, field: str, record_index: int):
    if field not in record:
        raise MissingFieldError(record_index, field)
    return record[field]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_convert_transform.py -v`
Expected: 10 passed

- [ ] **Step 6: 提交**

```bash
git add src/trainops/convert/ tests/test_convert_transform.py
git commit -m "feat: add convert transform (field extraction + value normalization)"
```

---

### Task 2: DatasetCache.iter_records() — 逐行流式读取缓存

**Files:**
- Modify: `src/trainops/fetch/cache.py`（新增方法；顶部 import 增加 `Iterator`、`Any`）
- Test: `tests/test_cache.py`（追加）

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_cache.py` 末尾：

```python
def test_iter_records_yields_line_bytes_and_records(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    line1 = '{"text": "hi"}\n'
    cache.data_path.write_text(line1 + '{"text": "中文"}\n', encoding="utf-8")
    items = list(cache.iter_records())
    assert items[0] == (len(line1.encode("utf-8")), {"text": "hi"})
    assert items[1] == (len('{"text": "中文"}\n'.encode("utf-8")), {"text": "中文"})


def test_iter_records_missing_file_raises(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    with pytest.raises(FileNotFoundError):
        list(cache.iter_records())


def test_iter_records_invalid_json_raises_with_line_number(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    cache.data_path.write_text('{"a": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        list(cache.iter_records())


def test_iter_records_skips_blank_lines(tmp_path):
    cache = DatasetCache(tmp_path, "org/ds", "default", "train")
    cache.dir.mkdir(parents=True)
    cache.data_path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    items = list(cache.iter_records())
    assert [record for _, record in items] == [{"a": 1}, {"a": 2}]
```

并在 `tests/test_cache.py` 顶部 import 区补充：

```python
import pytest
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_cache.py -k iter_records -v`
Expected: FAIL（`AttributeError: 'DatasetCache' object has no attribute 'iter_records'`）

- [ ] **Step 3: 实现 iter_records()**

修改 `src/trainops/fetch/cache.py`。

顶部 import 区改为（在 `import json`、`import os` 等现有 import 后追加）：

```python
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
```

在 `DatasetCache` 类中、`ensure_consistent` 方法之后追加：

```python
    def iter_records(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield (utf-8 byte length of the JSONL line, parsed record) one at a time.

        Raises FileNotFoundError when data.jsonl does not exist, and ValueError
        (with file:line context) when a line is not valid JSON.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"cache data file not found: {self.data_path} (run fetch first)"
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.data_path}:{line_number}: invalid JSON line: {exc}"
                    ) from exc
                yield len(line.encode("utf-8")), record
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_cache.py -v`
Expected: 全部 passed（原 7 个 + 新 4 个）

- [ ] **Step 5: 提交**

```bash
git add src/trainops/fetch/cache.py tests/test_cache.py
git commit -m "feat: add DatasetCache.iter_records for streaming jsonl reads"
```

---

### Task 3: command.py — run_convert 主流程（含类型规范化集成验证）

**Files:**
- Create: `src/trainops/convert/command.py`
- Test: `tests/test_convert_command.py`

- [ ] **Step 1: 写失败的测试（happy path + 类型规则）**

创建 `tests/test_convert_command.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_convert_command.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'trainops.convert.command'`）

- [ ] **Step 3: 实现 command.py**

创建 `src/trainops/convert/command.py`：

```python
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
        with pa.RecordBatchFileWriter(str(tmp_path), schema) as writer:
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_convert_command.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add src/trainops/convert/command.py tests/test_convert_command.py
git commit -m "feat: add run_convert (stream cache jsonl to HF arrow directory)"
```

---

### Task 4: command.py — 错误路径与 skip-bad / overwrite

**Files:**
- Modify: `src/trainops/convert/command.py`（Task 3 已创建，本任务不改实现，只补测试验证既有行为）
- Test: `tests/test_convert_command.py`（追加）

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_convert_command.py`：

```python
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
```

- [ ] **Step 2: 运行测试**

Run: `.venv/bin/pytest tests/test_convert_command.py -v`
Expected: 全部 passed（这些路径在 Task 3 的实现中已覆盖；若有失败，根据报错信息修正 `command.py` 后重跑，直至通过）

- [ ] **Step 3: 提交**

```bash
git add tests/test_convert_command.py
git commit -m "test: cover convert error paths, skip-bad, overwrite, empty cache"
```

---

### Task 5: CLI — 注册 convert 子命令

**Files:**
- Modify: `src/trainops/cli.py`（import 区 + 新 `@app.command()`）
- Test: `tests/test_cli.py`（追加）

- [ ] **Step 1: 写失败的测试**

在 `tests/test_cli.py` 顶部 import 区追加：

```python
from trainops.convert.command import (
    CacheNotFoundError,
    ConvertResult,
    OutputExistsError,
)
```

追加到 `tests/test_cli.py` 末尾：

```python
def _fake_run_convert(captured):
    def fake(dataset, subset, split, fields, cache_dir, output=None, skip_bad=False,
             overwrite=False, name=None):
        captured.update(dataset=dataset, subset=subset, split=split, fields=fields,
                        cache_dir=cache_dir, output=output, skip_bad=skip_bad,
                        overwrite=overwrite, name=name)
        return ConvertResult(output_dir=output or cache_dir / "data.arrow",
                             records=10, skipped=0, bytes=4096)

    return fake


def test_convert_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_convert", _fake_run_convert(captured))
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text, meta", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["dataset"] == "org/ds"
    assert captured["subset"] is None
    assert captured["split"] == "train"
    assert captured["fields"] == ["text", "meta"]
    assert captured["output"] is None
    assert captured["skip_bad"] is False
    assert captured["overwrite"] is False
    assert "10 records" in result.output
    assert "data.arrow" in result.output


def test_convert_command_flags(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_convert", _fake_run_convert(captured))
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--name", "my-ds", "--subset", "s1", "--split", "test",
                                 "--fields", "text", "--skip-bad", "--overwrite",
                                 "--output", str(tmp_path / "out"),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["name"] == "my-ds"
    assert captured["subset"] == "s1"
    assert captured["split"] == "test"
    assert captured["skip_bad"] is True
    assert captured["overwrite"] is True
    assert captured["output"] == tmp_path / "out"


def test_convert_command_skipped_warning(tmp_path, monkeypatch):
    captured = {}
    fake = _fake_run_convert(captured)

    def fake_with_skips(**kwargs):
        result = fake(**kwargs)
        result.skipped = 3
        return result

    monkeypatch.setattr("trainops.cli.run_convert", fake_with_skips)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "skipped 3" in result.output


def test_convert_command_empty_fields_exit_2(tmp_path):
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", " , ", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "empty" in result.output


def test_convert_command_cache_not_found_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise CacheNotFoundError("run fetch first")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "fetch" in result.output


def test_convert_command_output_exists_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise OutputExistsError("output directory already exists: /x")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output


def test_convert_command_missing_field_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise MissingFieldError(3, "title")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "title" in result.output
```

注意：`test_convert_command_missing_field_exit_1` 用到 `MissingFieldError`，顶部 import 还需追加：

```python
from trainops.convert.transform import MissingFieldError
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/test_cli.py -k convert -v`
Expected: FAIL（`SystemExit: No such command 'convert'` 或 import 错误）

- [ ] **Step 3: 实现 CLI**

修改 `src/trainops/cli.py`。

import 区追加（在现有 `from trainops.fetch.cache import CacheCorruptedError` 之前）：

```python
from trainops.convert.command import (
    CacheNotFoundError,
    OutputExistsError,
    run_convert,
)
from trainops.convert.transform import MissingFieldError, parse_fields
```

在 `fetch` 命令函数之后追加：

```python
@app.command()
def convert(
    dataset: str = typer.Option(..., help="ModelScope dataset id (must already be cached by fetch)"),
    name: str | None = typer.Option(None, help="Cache alias used at fetch time"),
    subset: str | None = typer.Option(None, help="Dataset subset name"),
    split: str = typer.Option("train", help="Dataset split"),
    fields: str = typer.Option(..., help="Comma-separated field whitelist, e.g. 'text' or 'text,title'"),
    skip_bad: bool = typer.Option(False, "--skip-bad",
                                  help="Skip records missing fields or with invalid JSON lines"),
    output: Path | None = typer.Option(None, help="Output directory (default: <cache>/.../data.arrow)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output directory"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Convert a cached ModelScope dataset into an HF arrow dataset directory."""
    try:
        field_list = parse_fields(fields)
    except ValueError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        result = run_convert(dataset, subset, split, field_list, cache_dir,
                             output=output, skip_bad=skip_bad, overwrite=overwrite,
                             name=name)
    except CacheNotFoundError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except OutputExistsError as exc:
        typer.secho(f"error: {exc} (use --overwrite to replace it)",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except (MissingFieldError, ValueError) as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"error: failed to convert dataset '{dataset}': {exc}",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"done: {result.records} records -> {result.output_dir} "
               f"({result.bytes} bytes)")
    if result.skipped:
        typer.secho(f"warning: skipped {result.skipped} bad records",
                    fg=typer.colors.YELLOW)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: 全部 passed（原 7 个 fetch 测试 + 新 7 个 convert 测试）

- [ ] **Step 5: 提交**

```bash
git add src/trainops/cli.py tests/test_cli.py
git commit -m "feat: register convert subcommand in CLI"
```

---

### Task 6: 文档与全量验证

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新 README**

把 README 首行“当前包含 `fetch` 数据获取工具。”改为：

```markdown
LLM 训练 CI/CD 工具集。当前包含 `fetch` 数据获取工具与 `convert` 缓存转 HF Arrow 工具。
```

在 `## fetch` 一节的进度条目之后（`## 测试` 之前）插入新节：

```markdown
## convert：缓存数据集转 HF Arrow 格式

```bash
trainops convert --dataset AI-ModelScope/xxx \
    --name my-dataset \
    --subset default --split train \
    --fields text,meta \
    --cache-dir ~/.cache/trainops
```

- 定位与 `fetch` 完全相同的缓存条目（`--dataset`/`--name`/`--subset`/`--split`），把 `data.jsonl` 转换为 HF datasets 目录
- 默认输出到缓存并排目录 `<cache-dir>/datasets/<name>/<subset>/<split>/data.arrow`，可用 `--output` 覆盖；已存在时报错，`--overwrite` 清空重建
- `--fields` 必填：逗号分隔的字段白名单。字符串原样保留，数字/布尔/数组/对象序列化为 JSON 字符串，所有列为 Arrow `string` 类型，便于直接接 tokenizer
- 缺字段或非法 JSON 行默认报错退出；`--skip-bad` 跳过并统计
- 分批流式转换（每批 1000 条写入 Arrow RecordBatch），GB 级缓存内存占用恒定
- 训练侧用法：`datasets.load_from_disk("<...>/data.arrow")` 后直接 `.map()` tokenize
```

- [ ] **Step 2: 全量测试**

Run: `.venv/bin/pytest -q`
Expected: 全部 passed，无 slow 测试运行

- [ ] **Step 3: 冒烟验证（真实 CLI，可选但推荐）**

```bash
.venv/bin/python -m trainops.cli fetch --dataset AI-ModelScope/xxx --max-size 1MB --cache-dir /tmp/trainops-smoke 2>&1 | tail -2
.venv/bin/python -m trainops.cli convert --dataset AI-ModelScope/xxx --fields text --cache-dir /tmp/trainops-smoke
.venv/bin/python -c "from datasets import load_from_disk; import glob; d=load_from_disk(glob.glob('/tmp/trainops-smoke/datasets/*/default/train/data.arrow')[0]); print(d)"
```

注：第一条需要网络与真实数据集；无网络环境可跳过，单测已覆盖转换逻辑。若数据集实际字段名不是 `text`，用 `--fields` 指定真实存在的字段。

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs: add convert command to README"
```
