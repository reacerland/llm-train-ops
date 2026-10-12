# trainops fetch 数据获取工具实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `trainops fetch` 命令：从 ModelScope 流式拉取数据集，按字节配额缓存到本地，支持增量续拉。

**Architecture:** typer 子命令 CLI；`fetch` 包内分四层：sizeparse（大小解析）、cache（append-only JSONL 缓存 + 原子 checkpoint 元数据 + 一致性校验）、streamer（MsDataset streaming 封装，惰性导入）、command（编排：命中判断、跳过已缓存记录、周期 checkpoint、崩溃恢复）。设计文档：`docs/superpowers/specs/2026-10-12-fetch-tool-design.md`。

**Tech Stack:** Python ≥3.10、uv、typer、modelscope、pytest。

---

### Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/trainops/__init__.py`
- Create: `src/trainops/cli.py`
- Create: `src/trainops/fetch/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "trainops"
version = "0.1.0"
description = "LLM train CI/CD toolkit"
requires-python = ">=3.10"
dependencies = [
    "modelscope>=1.20",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
trainops = "trainops.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trainops"]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with --runslow to run)",
]
```

- [ ] **Step 2: 创建包骨架文件**

`src/trainops/__init__.py`（空文件）、`src/trainops/fetch/__init__.py`（空文件）、`tests/__init__.py`（空文件）。

`src/trainops/cli.py`：

```python
import typer

app = typer.Typer(help="LLM train CI/CD toolkit")
```

`.gitignore`：

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
dist/
```

- [ ] **Step 3: 创建虚拟环境并安装**

```bash
cd /Users/wangshuangbao/workspace/llm-train-ops
uv venv .venv
uv pip install -e ".[dev]"
```

预期：安装成功（modelscope 依赖较多，可能需要几分钟）。

- [ ] **Step 4: 验证 CLI 可运行**

Run: `.venv/bin/trainops --help`
预期：输出 usage 信息，包含 "LLM train CI/CD toolkit"。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold trainops package with typer CLI"
```

---

### Task 2: 大小解析 sizeparse

**Files:**
- Create: `src/trainops/fetch/sizeparse.py`
- Test: `tests/test_sizeparse.py`

- [ ] **Step 1: 写失败测试**

`tests/test_sizeparse.py`：

```python
import pytest

from trainops.fetch.sizeparse import parse_size


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024B", 1024),
        ("512KB", 512 * 1024),
        ("500MB", 500 * 1024**2),
        ("2GB", 2 * 1024**3),
        ("1.5GB", int(1.5 * 1024**3)),
        ("500mb", 500 * 1024**2),
        (" 2 GB ", 2 * 1024**3),
    ],
)
def test_parse_size_valid(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "500", "500TB", "-1GB", "MB"])
def test_parse_size_invalid(text):
    with pytest.raises(ValueError):
        parse_size(text)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_sizeparse.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'trainops.fetch.sizeparse'`。

- [ ] **Step 3: 实现 sizeparse**

`src/trainops/fetch/sizeparse.py`：

```python
"""Parse human-readable sizes like '500MB' into bytes (1024-based)."""

import re

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
_PATTERN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)\s*", re.IGNORECASE)


def parse_size(text: str) -> int:
    """Parse '500MB' / '2GB' / '1024B' into a byte count. Raises ValueError on bad input."""
    match = _PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"invalid size: {text!r} (expected e.g. '500MB', '2GB')")
    value, unit = match.groups()
    return int(float(value) * _UNITS[unit.upper()])
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_sizeparse.py -v`
预期：全部 PASS（13 个用例）。

- [ ] **Step 5: Commit**

```bash
git add src/trainops/fetch/sizeparse.py tests/test_sizeparse.py
git commit -m "feat: add size parsing for fetch quota"
```

---

### Task 3: 缓存元数据 cache（meta 读写）

**Files:**
- Create: `src/trainops/fetch/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cache.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_cache.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'trainops.fetch.cache'`。

- [ ] **Step 3: 实现 cache 元数据部分**

`src/trainops/fetch/cache.py`：

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_cache.py -v`
预期：4 个用例全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/trainops/fetch/cache.py tests/test_cache.py
git commit -m "feat: add dataset cache metadata with atomic checkpointing"
```

---

### Task 4: 缓存写入与一致性校验（append / ensure_consistent）

**Files:**
- Modify: `src/trainops/fetch/cache.py`
- Test: `tests/test_cache_consistency.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cache_consistency.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_cache_consistency.py -v`
预期：FAIL，`AttributeError: 'DatasetCache' object has no attribute 'append'`。

- [ ] **Step 3: 实现 append 与 ensure_consistent**

在 `src/trainops/fetch/cache.py` 的 `DatasetCache` 类中追加两个方法（放在 `save_meta` 之后）：

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_cache_consistency.py tests/test_cache.py -v`
预期：全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/trainops/fetch/cache.py tests/test_cache_consistency.py
git commit -m "feat: add cache append and crash-consistency reconciliation"
```

---

### Task 5: 流式拉取编排 streamer + run_fetch

**Files:**
- Create: `src/trainops/fetch/streamer.py`
- Create: `src/trainops/fetch/command.py`
- Test: `tests/test_fetch_command.py`

- [ ] **Step 1: 写失败测试**

`tests/test_fetch_command.py`：

```python
import json

import pytest

from trainops.fetch.command import run_fetch

ROWS = [{"idx": i, "text": f"row {i}"} for i in range(10)]
ROW_BYTES = len((json.dumps(ROWS[0], ensure_ascii=False) + "\n").encode("utf-8"))
ALL_BYTES = ROW_BYTES * len(ROWS)


def fake_stream_factory(rows):
    def stream_fn(dataset, subset, split):
        yield from rows

    return stream_fn


def _data_file(tmp_path):
    return tmp_path / "datasets" / "org__ds" / "default" / "train" / "data.jsonl"


def test_first_fetch_writes_until_target(tmp_path):
    result = run_fetch("org/ds", "default", "train", ALL_BYTES, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.downloaded is True
    assert result.records == 10
    assert result.bytes == ALL_BYTES
    assert result.dataset_exhausted is False
    lines = _data_file(tmp_path).read_text().splitlines()
    assert len(lines) == 10
    assert json.loads(lines[0])["idx"] == 0


def test_first_fetch_stops_at_target_mid_stream(tmp_path):
    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 3, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.records == 3
    assert result.bytes == ROW_BYTES * 3
    assert result.dataset_exhausted is False


def test_cache_hit_returns_without_downloading(tmp_path):
    first = run_fetch("org/ds", "default", "train", 10**9, tmp_path,
                      stream_fn=fake_stream_factory(ROWS))
    assert first.dataset_exhausted is True

    def exploding_stream(dataset, subset, split):
        raise AssertionError("stream_fn must not be called on cache hit")

    second = run_fetch("org/ds", "default", "train", first.bytes, tmp_path,
                       stream_fn=exploding_stream)
    assert second.downloaded is False
    assert second.bytes == first.bytes
    assert second.records == 10


def test_incremental_fetch_appends_only_delta(tmp_path):
    run_fetch("org/ds", "default", "train", ROW_BYTES * 3, tmp_path,
              stream_fn=fake_stream_factory(ROWS))
    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 5, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.records == 5
    lines = _data_file(tmp_path).read_text().splitlines()
    assert [json.loads(line)["idx"] for line in lines] == [0, 1, 2, 3, 4]


def test_resume_after_interruption(tmp_path):
    state = {"calls": 0}

    def flaky_stream(dataset, subset, split):
        state["calls"] += 1
        for i, row in enumerate(ROWS):
            if state["calls"] == 1 and i == 4:
                raise ConnectionError("network down")
            yield row

    with pytest.raises(ConnectionError):
        run_fetch("org/ds", "default", "train", ROW_BYTES * 8, tmp_path, stream_fn=flaky_stream)

    result = run_fetch("org/ds", "default", "train", ROW_BYTES * 8, tmp_path, stream_fn=flaky_stream)
    assert result.records == 8
    lines = _data_file(tmp_path).read_text().splitlines()
    assert [json.loads(line)["idx"] for line in lines] == list(range(8))


def test_dataset_exhausted_before_target(tmp_path):
    result = run_fetch("org/ds", "default", "train", 10**9, tmp_path,
                       stream_fn=fake_stream_factory(ROWS))
    assert result.dataset_exhausted is True
    assert result.records == 10
    assert result.bytes == ALL_BYTES
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_fetch_command.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'trainops.fetch.command'`。

- [ ] **Step 3: 实现 streamer**

`src/trainops/fetch/streamer.py`：

```python
"""Thin wrapper over the ModelScope SDK streaming API."""

from collections.abc import Iterator
from typing import Any


def stream_dataset(dataset: str, subset: str, split: str) -> Iterator[dict[str, Any]]:
    """Yield raw records from a ModelScope dataset in streaming mode.

    Raises (from the SDK) when the dataset does not exist, auth fails, or the
    dataset does not support streaming.
    """
    from modelscope.msdatasets import MsDataset  # lazy: heavy import

    ds = MsDataset.load(dataset, subset_name=subset, split=split, streaming=True)
    for row in ds:
        yield dict(row)
```

- [ ] **Step 4: 实现 run_fetch 编排**

`src/trainops/fetch/command.py`：

```python
"""Fetch orchestration: quota check, incremental download, checkpointing."""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import DatasetCache
from .streamer import stream_dataset

CHECKPOINT_INTERVAL = 1000

StreamFn = Callable[[str, str, str], Iterator[dict[str, Any]]]


@dataclass
class FetchResult:
    cache_dir: Path
    records: int
    bytes: int
    downloaded: bool
    dataset_exhausted: bool = False


def run_fetch(
    dataset: str,
    subset: str,
    split: str,
    max_size: int,
    cache_dir: Path,
    stream_fn: StreamFn = stream_dataset,
) -> FetchResult:
    """Fetch up to max_size bytes of raw records into the local cache.

    - Cache hit (cached bytes >= max_size): return without downloading.
    - Otherwise resume after meta.records rows and append until the quota is met
      or the dataset is exhausted. Metadata is checkpointed every
      CHECKPOINT_INTERVAL records and always on exit (including exceptions).
    """
    cache = DatasetCache(cache_dir, dataset, subset, split)
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
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/pytest tests/test_fetch_command.py -v`
预期：6 个用例全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/trainops/fetch/streamer.py src/trainops/fetch/command.py tests/test_fetch_command.py
git commit -m "feat: add streaming fetch orchestration with incremental resume"
```

---

### Task 6: CLI 接线 fetch 子命令

**Files:**
- Modify: `src/trainops/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

`tests/test_cli.py`：

```python
from typer.testing import CliRunner

from trainops.cli import app
from trainops.fetch.command import FetchResult

runner = CliRunner()


def _fake_run_fetch(captured, **overrides):
    def fake(dataset, subset, split, max_size, cache_dir):
        captured.update(dataset=dataset, subset=subset, split=split,
                        max_size=max_size, cache_dir=cache_dir)
        defaults = dict(cache_dir=cache_dir, records=10, bytes=2048, downloaded=True)
        return FetchResult(**(defaults | overrides))

    return fake


def test_fetch_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_fetch", _fake_run_fetch(captured))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["dataset"] == "org/ds"
    assert captured["subset"] == "default"
    assert captured["split"] == "train"
    assert captured["max_size"] == 2048
    assert "2048 bytes" in result.output


def test_fetch_command_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("trainops.cli.run_fetch",
                        _fake_run_fetch({}, downloaded=False))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "cache hit" in result.output


def test_fetch_command_dataset_exhausted_warns(tmp_path, monkeypatch):
    monkeypatch.setattr("trainops.cli.run_fetch",
                        _fake_run_fetch({}, dataset_exhausted=True))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "dataset exhausted" in result.output


def test_fetch_command_invalid_size(tmp_path):
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "abc",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "invalid size" in result.output


def test_fetch_command_sdk_error_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, max_size, cache_dir):
        raise RuntimeError("dataset not found")

    monkeypatch.setattr("trainops.cli.run_fetch", failing)
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "org/ds" in result.output
    assert "dataset not found" in result.output
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_cli.py -v`
预期：FAIL（`fetch` 命令不存在 / `trainops.cli.run_fetch` 属性不存在）。

- [ ] **Step 3: 实现 CLI**

`src/trainops/cli.py` 完整替换为：

```python
from pathlib import Path

import typer

from trainops.fetch.cache import CacheCorruptedError
from trainops.fetch.command import run_fetch
from trainops.fetch.sizeparse import parse_size

app = typer.Typer(help="LLM train CI/CD toolkit")


@app.command()
def fetch(
    dataset: str = typer.Option(..., help="ModelScope dataset id, e.g. 'AI-ModelScope/xxx'"),
    subset: str = typer.Option("default", help="Dataset subset name"),
    split: str = typer.Option("train", help="Dataset split"),
    max_size: str = typer.Option(..., "--max-size", help="Target cache size, e.g. '500MB', '2GB'"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Fetch a ModelScope dataset into the local cache (streaming, size-capped)."""
    try:
        size = parse_size(max_size)
    except ValueError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        result = run_fetch(dataset, subset, split, size, cache_dir)
    except CacheCorruptedError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(
            f"error: failed to fetch dataset '{dataset}' (subset={subset}, split={split}): {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from exc

    if not result.downloaded:
        typer.echo(f"cache hit: {result.bytes} bytes, {result.records} records already at {result.cache_dir}")
    else:
        typer.echo(f"done: {result.bytes} bytes, {result.records} records cached at {result.cache_dir}")
        if result.dataset_exhausted:
            typer.secho("warning: dataset exhausted before reaching target size",
                        fg=typer.colors.YELLOW)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_cli.py -v`
预期：5 个用例全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/trainops/cli.py tests/test_cli.py
git commit -m "feat: wire fetch subcommand into CLI"
```

---

### Task 7: 集成测试（slow）+ README + 全量验证

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_integration_fetch.py`
- Create: `README.md`

- [ ] **Step 1: 创建 conftest（slow 标记默认跳过）**

`tests/conftest.py`：

```python
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests (real ModelScope downloads)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
```

- [ ] **Step 2: 写集成测试**

`tests/test_integration_fetch.py`：

```python
import json

import pytest

from trainops.fetch.command import run_fetch

pytestmark = pytest.mark.slow

DATASET = "modelscope/gsm8k"  # small public dataset on ModelScope
TARGET = 100 * 1024  # 100KB


def test_fetch_real_dataset(tmp_path):
    result = run_fetch(DATASET, "main", "train", TARGET, tmp_path)
    assert result.bytes >= TARGET or result.dataset_exhausted
    data_file = tmp_path / "datasets" / "modelscope__gsm8k" / "main" / "train" / "data.jsonl"
    assert data_file.exists()
    first = json.loads(data_file.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first, dict)

    # Same target again must be a cache hit (no download).
    again = run_fetch(DATASET, "main", "train", TARGET, tmp_path)
    assert again.downloaded is False
    assert again.bytes == result.bytes
```

- [ ] **Step 3: 运行集成测试验证真实拉取**

Run: `.venv/bin/pytest tests/test_integration_fetch.py --runslow -v`
预期：PASS（需要网络；若 ModelScope 上 `modelscope/gsm8k` 的 subset/split 名称不符，按报错调整为实际存在的组合）。

- [ ] **Step 4: 写 README**

`README.md`：

````markdown
# llm-train-ops

LLM 训练 CI/CD 工具集。当前包含 `fetch` 数据获取工具。

## 安装

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## fetch：从 ModelScope 拉取数据集到本地缓存

```bash
trainops fetch --dataset AI-ModelScope/xxx \
    --subset default --split train \
    --max-size 500MB \
    --cache-dir ~/.cache/trainops
```

- 流式拉取，按字节配额（`B/KB/MB/GB`，1024 进制）停止，不全量下载
- 原样保存：每条记录完整 JSON 序列化为 `data.jsonl` 一行
- 缓存增量复用：同一数据集再次请求更大配额时只拉差额；中断后重跑自动续拉
- 缓存位置：`<cache-dir>/datasets/<dataset-slug>/<subset>/<split>/`（`data.jsonl` + `meta.json`）

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
````

- [ ] **Step 5: 全量测试 + CLI 冒烟**

```bash
.venv/bin/pytest -v
.venv/bin/trainops fetch --help
```

预期：单元测试全部 PASS（slow 被 skip）；`fetch --help` 显示各参数说明。

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_integration_fetch.py README.md
git commit -m "test: add slow integration test; docs: add README usage"
```
