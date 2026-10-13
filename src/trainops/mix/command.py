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
from .pack import PackResult, SourcePackStats, pack
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
    valid: dict | None = None   # meta['valid'] section, None without valid sources


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

    train_sources = [s for s in config.sources if s.role == "train"]
    valid_sources = [s for s in config.sources if s.role == "valid"]

    with tempfile.TemporaryDirectory(prefix="trainops-mix-") as tmp:
        pts = _pretokenize_all(config, tokenizer, arrow_dirs, Path(tmp), progress)
        _check_overflow(pts, dtype, tokenizer)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = output_dir / "tokens.bin"
        # Train rng stream must not depend on the presence of valid sources:
        # train packs with default_rng(seed) exactly as before roles existed.
        result = _pack_all(train_sources, pts, dtype, tokens_path,
                           config.max_seq_len, config.seed, progress)
        valid_result = None
        valid_tokens_path = output_dir / "valid_tokens.bin"
        if valid_sources:
            valid_result = _pack_all(valid_sources, pts, dtype,
                                     valid_tokens_path, config.max_seq_len,
                                     config.seed + 1, progress)

    meta = _build_meta(config, tokenizer, dtype, pts, result, valid_result)
    meta_path = output_dir / "meta.json"
    _atomic_write_json(meta_path, meta)

    bytes_total = tokens_path.stat().st_size + meta_path.stat().st_size
    if valid_result is not None:
        bytes_total += valid_tokens_path.stat().st_size
    return MixResult(
        output_dir=output_dir,
        num_sequences=result.sequences,
        total_tokens=result.tokens_written,
        dropped_tail_tokens=result.dropped_tail_tokens,
        exhausted_sources=result.exhausted_sources,
        sources_meta=meta["sources"],
        bytes=bytes_total,
        valid=meta.get("valid"),
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


def _pack_all(sources: list, pts: dict[str, PretokenizedSource],
              dtype: np.dtype, tokens_path: Path, seq_len: int, seed: int,
              progress: bool) -> PackResult:
    streams = {s.name: iter_docs(pts[s.name]) for s in sources}
    weights = {s.name: s.ratio for s in sources}
    rng = np.random.default_rng(seed)
    pbar = tqdm(unit="seq", desc="pack", disable=not progress)
    with open(tokens_path, "wb") as f:
        def sink(seq: list[int]) -> None:
            f.write(np.asarray(seq, dtype=dtype).tobytes())
            pbar.update(1)

        try:
            result = pack(streams, weights, seq_len, rng, sink)
        finally:
            pbar.close()
    return result


def _build_meta(config: MixConfig, tokenizer: Tokenizer, dtype: np.dtype,
                pts: dict[str, PretokenizedSource],
                result: PackResult,
                valid_result: PackResult | None) -> dict:
    tokenizer_meta = (
        {"preset": config.tokenizer}
        if isinstance(config.tokenizer, str)
        else {"class": f"{config.tokenizer.path}:{config.tokenizer.class_name}"}
    )
    meta = {
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
            _source_meta(s, pts[s.name], result.per_source[s.name])
            for s in config.sources if s.role == "train"
        ],
    }
    if valid_result is not None:
        meta["valid"] = {
            "num_sequences": valid_result.sequences,
            "total_tokens": valid_result.tokens_written,
            "dropped_tail_tokens": valid_result.dropped_tail_tokens,
            "exhausted_sources": valid_result.exhausted_sources,
            "sources": [
                _source_meta(s, pts[s.name], valid_result.per_source[s.name])
                for s in config.sources if s.role == "valid"
            ],
        }
    return meta


def _source_meta(source, pt: PretokenizedSource,
                 stats: SourcePackStats) -> dict:
    return {
        "name": source.name,
        "dataset": source.dataset,
        "subset": source.subset,
        "split": source.split,
        "configured_ratio": source.ratio,
        "docs_total": pt.docs_total,
        "docs_sampled": stats.docs_sampled,
        "tokens": stats.tokens,
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
