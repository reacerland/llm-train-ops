"""Stage one of mix: encode one Arrow dataset into a temp token stream.

Each document is encoded with the configured tokenizer (which by contract
adds no special tokens), terminated with one EOS token, and written to a
flat uint32 file. A uint64 sidecar of offsets (one entry per document plus
a leading zero) lets stage two yield documents back as zero-copy views.

These files are intermediate state only — the final training artifact has
no document boundaries.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SourceConfig
from .tokenizer import Tokenizer


class PretokenizeError(Exception):
    """Raised when a source dataset cannot be pretokenized."""


@dataclass
class PretokenizedSource:
    name: str
    tokens_path: Path     # uint32 flat stream, every document ends with EOS
    offsets_path: Path    # uint64, docs_total + 1 entries (incl. sentinels)
    docs_total: int       # non-empty documents (the sampling population)
    skipped_empty: int    # documents that encoded to zero tokens
    tokens_total: int     # includes appended EOS tokens
    max_token: int        # max token id seen (incl. EOS); uint16 overflow check


def pretokenize(source: SourceConfig, tokenizer: Tokenizer,
                arrow_dir: Path, tmp_dir: Path) -> PretokenizedSource:
    from datasets import load_from_disk  # lazy: heavy import

    try:
        ds = load_from_disk(str(arrow_dir))
    except Exception as exc:
        raise PretokenizeError(
            f"source {source.name!r}: cannot load arrow dataset at {arrow_dir}: "
            f"{exc}") from exc
    if source.field not in ds.column_names:
        raise PretokenizeError(
            f"source {source.name!r}: field {source.field!r} not in dataset "
            f"columns {ds.column_names}")

    tokens_path = Path(tmp_dir) / f"{source.name}.bin"
    offsets_path = Path(tmp_dir) / f"{source.name}.idx"
    offsets = [0]
    docs_total = 0
    skipped_empty = 0
    tokens_total = 0
    max_token = 0
    eos = tokenizer.eos_token_id

    with open(tokens_path, "wb") as f:
        for row_index, text in enumerate(ds[source.field], start=1):
            if text is None:  # Arrow null: treat as an empty document
                skipped_empty += 1
                continue
            try:
                ids = list(tokenizer.encode(text))
            except Exception as exc:
                raise PretokenizeError(
                    f"source {source.name!r} row {row_index}: encode failed: "
                    f"{exc}") from exc
            if not ids:
                skipped_empty += 1
                continue
            doc = np.asarray(ids + [eos], dtype=np.uint32)
            doc.tofile(f)
            offsets.append(offsets[-1] + len(doc))
            docs_total += 1
            tokens_total += len(doc)
            max_token = max(max_token, max(ids), eos)

    if docs_total == 0:
        raise PretokenizeError(
            f"source {source.name!r} has no non-empty documents")
    np.asarray(offsets, dtype=np.uint64).tofile(offsets_path)
    return PretokenizedSource(
        name=source.name, tokens_path=tokens_path, offsets_path=offsets_path,
        docs_total=docs_total, skipped_empty=skipped_empty,
        tokens_total=tokens_total, max_token=max_token)


def iter_docs(pt: PretokenizedSource) -> Iterator[np.ndarray]:
    """Yield each document's tokens (including the trailing EOS) as a view."""
    tokens = np.memmap(pt.tokens_path, dtype=np.uint32, mode="r")
    offsets = np.memmap(pt.offsets_path, dtype=np.uint64, mode="r")
    for i in range(len(offsets) - 1):
        yield tokens[int(offsets[i]):int(offsets[i + 1])]
