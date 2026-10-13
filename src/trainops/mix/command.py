"""Mix orchestration: config -> pretokenize -> pack -> tokens.bin + meta.json."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trainops.convert.command import ARROW_DIR_NAME, OutputExistsError

from .config import MixConfig, resolve_mix_config
from .pack import PackResult, pack
from .pretokenize import PretokenizedSource, iter_docs, pretokenize
from .tokenizer import Tokenizer, load_tokenizer

MIXTURES_DIR_NAME = "mixtures"
META_FORMAT = "trainops-mix/v1"
UINT16_LIMIT = 65536


class MixCacheNotFoundError(Exception):
    """Raised when a source's convert output (data.arrow) is missing."""


@dataclass
class MixResult:
    output_dir: Path
    num_sequences: int
    total_tokens: int
    dropped_tail_tokens: int
    exhausted_sources: list[str]
    sources_meta: list[dict]
    bytes: int


def run_mix(config_path: Path, cache_dir: Path, output: Path | None = None,
            overwrite: bool = False, progress: bool = True) -> MixResult:
    """Build a tokenized, packed, mixed training dataset from a TOML config.

    Sources are sampled document-by-document by configured ratio (no
    replacement); the first exhausted source stops the packing. Output is
    ``tokens.bin`` (flat ``max_seq_len`` sequences) + ``meta.json``.
    """
    config_path = Path(config_path)
    cache_dir = Path(cache_dir)
    config, config_output = resolve_mix_config(config_path)
    tokenizer = load_tokenizer(config.tokenizer)

    arrow_dirs = {
        s.name: cache_dir / "datasets" / s.dataset / s.subset / s.split / ARROW_DIR_NAME
        for s in config.sources
    }
    for s in config.sources:
        if not arrow_dirs[s.name].is_dir():
            raise MixCacheNotFoundError(
                f"arrow dataset for source {s.name!r} (dataset={s.dataset}, "
                f"subset={s.subset}, split={s.split}) not found at "
                f"{arrow_dirs[s.name]}; run convert first")

    output_dir = Path(output) if output else \
        (config_output or cache_dir / MIXTURES_DIR_NAME / config_path.stem)
    if output_dir.exists() and not overwrite:
        raise OutputExistsError(f"output directory already exists: {output_dir}")

    dtype = np.uint16 if tokenizer.vocab_size < UINT16_LIMIT else np.uint32

    with tempfile.TemporaryDirectory(prefix="trainops-mix-") as tmp:
        pts = _pretokenize_all(config, tokenizer, arrow_dirs, Path(tmp), progress)
        _check_overflow(pts, dtype, tokenizer)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = output_dir / "tokens.bin"
        result = _pack_all(config, pts, dtype, tokens_path, progress)

    meta = _build_meta(config, tokenizer, dtype, pts, result)
    meta_path = output_dir / "meta.json"
    _atomic_write_json(meta_path, meta)

    bytes_total = tokens_path.stat().st_size + meta_path.stat().st_size
    return MixResult(
        output_dir=output_dir,
        num_sequences=result.sequences,
        total_tokens=result.tokens_written,
        dropped_tail_tokens=result.dropped_tail_tokens,
        exhausted_sources=result.exhausted_sources,
        sources_meta=meta["sources"],
        bytes=bytes_total,
    )


def _pretokenize_all(config: MixConfig, tokenizer: Tokenizer,
                     arrow_dirs: dict[str, Path], tmp_dir: Path,
                     progress: bool) -> dict[str, PretokenizedSource]:
    pts: dict[str, PretokenizedSource] = {}
    for s in config.sources:
        if progress:
            print(f"pretokenize: source {s.name!r} ({s.dataset})...", flush=True)
        pts[s.name] = pretokenize(s, tokenizer, arrow_dirs[s.name], tmp_dir)
    return pts


def _check_overflow(pts: dict[str, PretokenizedSource], dtype: np.dtype,
                    tokenizer: Tokenizer) -> None:
    if dtype != np.uint16:
        return
    for pt in pts.values():
        if pt.max_token >= UINT16_LIMIT:
            raise ValueError(
                f"token id {pt.max_token} from source {pt.name!r} does not fit "
                f"uint16; tokenizer vocab_size={tokenizer.vocab_size} is "
                f"inconsistent with the ids it emits")


def _pack_all(config: MixConfig, pts: dict[str, PretokenizedSource],
              dtype: np.dtype, tokens_path: Path, progress: bool) -> PackResult:
    streams = {name: iter_docs(pt) for name, pt in pts.items()}
    weights = {s.name: s.ratio for s in config.sources}
    rng = np.random.default_rng(config.seed)
    pbar = tqdm(unit="seq", desc="pack", disable=not progress)
    with open(tokens_path, "wb") as f:
        def sink(seq: list[int]) -> None:
            f.write(np.asarray(seq, dtype=dtype).tobytes())
            pbar.update(1)

        try:
            result = pack(streams, weights, config.max_seq_len, rng, sink)
        finally:
            pbar.close()
    return result


def _build_meta(config: MixConfig, tokenizer: Tokenizer, dtype: np.dtype,
                pts: dict[str, PretokenizedSource],
                result: PackResult) -> dict:
    tokenizer_meta = (
        {"preset": config.tokenizer}
        if isinstance(config.tokenizer, str)
        else {"class": f"{config.tokenizer.path}:{config.tokenizer.class_name}"}
    )
    return {
        "format": META_FORMAT,
        "tokenizer": tokenizer_meta,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "max_seq_len": config.max_seq_len,
        "seed": config.seed,
        "num_sequences": result.sequences,
        "total_tokens": result.tokens_written,
        "dropped_tail_tokens": result.dropped_tail_tokens,
        "exhausted_sources": result.exhausted_sources,
        "sources": [
            {
                "name": s.name,
                "dataset": s.dataset,
                "subset": s.subset,
                "split": s.split,
                "configured_ratio": s.ratio,
                "docs_total": pts[s.name].docs_total,
                "docs_sampled": result.per_source[s.name].docs_sampled,
                "tokens": result.per_source[s.name].tokens,
            }
            for s in config.sources
        ],
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
