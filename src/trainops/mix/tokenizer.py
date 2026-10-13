"""Tokenizer loading for mix: preset registry + custom Python class files.

The mix tool talks to tokenizers through a duck-typed interface — no
inheritance required:

    eos_token_id: int                 appended after every document
    vocab_size: int                   decides uint16 vs uint32 storage
    encode(text: str) -> list[int]    must NOT add special tokens itself
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import TokenizerClassRef


class TokenizerLoadError(Exception):
    """Raised when a preset or class-file tokenizer cannot be loaded."""


@runtime_checkable
class Tokenizer(Protocol):
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...


PRESETS: dict[str, Callable[[], Tokenizer]] = {}


def register_preset(name: str, loader: Callable[[], Tokenizer]) -> None:
    PRESETS[name] = loader


def load_tokenizer(spec: str | TokenizerClassRef) -> Tokenizer:
    """Load a tokenizer from a preset name or a class-file reference."""
    if isinstance(spec, TokenizerClassRef):
        return load_tokenizer_class(spec)
    if spec not in PRESETS:
        raise TokenizerLoadError(
            f"unknown tokenizer preset {spec!r}; available presets: "
            f"{', '.join(sorted(PRESETS))}")
    return validate_interface(PRESETS[spec](), where=f"preset {spec!r}")


def load_tokenizer_class(ref: TokenizerClassRef) -> Tokenizer:
    """Import ``class_name`` from the .py file at ``ref.path`` and instantiate."""
    path = Path(ref.path)
    if not path.is_file():
        raise TokenizerLoadError(f"tokenizer class file not found: {path}")
    module_name = f"trainops_user_tokenizer_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TokenizerLoadError(f"cannot import tokenizer class file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TokenizerLoadError(
            f"failed to import tokenizer class file {path}: {exc}") from exc
    try:
        cls = getattr(module, ref.class_name)
    except AttributeError:
        raise TokenizerLoadError(
            f"tokenizer class {ref.class_name!r} not found in {path}") from None
    try:
        instance = cls()
    except Exception as exc:
        raise TokenizerLoadError(
            f"failed to instantiate tokenizer class {ref.class_name!r}: {exc}"
        ) from exc
    return validate_interface(instance, where=f"class {ref.class_name!r} in {path}")


def validate_interface(obj: Any, where: str = "tokenizer") -> Tokenizer:
    """Check the duck-typed tokenizer interface, naming what's missing."""
    for attr, typ in (("eos_token_id", int), ("vocab_size", int)):
        value = getattr(obj, attr, None)
        if not isinstance(value, typ) or isinstance(value, bool):
            raise TokenizerLoadError(
                f"{where}: missing or invalid {attr!r} (expected int)")
    if not callable(getattr(obj, "encode", None)):
        raise TokenizerLoadError(f"{where}: missing method 'encode'")
    return obj


class _HFTokenizerWrapper:
    """Adapter giving an HF AutoTokenizer the mix duck-type interface.

    encode() is pinned to add_special_tokens=False: the tool appends EOS
    itself and never wants an implicit BOS/EOS pair from the tokenizer.
    """

    def __init__(self, hf_tokenizer: Any):
        self._hf = hf_tokenizer

    @property
    def eos_token_id(self) -> int:
        return self._hf.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._hf.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._hf.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._hf.decode(ids)


def _load_llama2() -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerLoadError(
            "the 'llama2' preset requires the 'transformers' package: "
            "pip install trainops[hf]") from exc
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Llama-2-7b-hf", local_files_only=True)
    except Exception as exc:
        raise TokenizerLoadError(
            "the 'llama2' preset tokenizer (NousResearch/Llama-2-7b-hf) is "
            "not in the local HuggingFace cache; download it once with "
            "AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf') "
            "before running offline") from exc
    return _HFTokenizerWrapper(hf_tokenizer)


PRESETS["llama2"] = _load_llama2
