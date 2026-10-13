"""TOML mix config parsing and validation (pure, no IO beyond reading the file)."""

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULT_SUBSET = "default"
DEFAULT_SPLIT = "train"
DEFAULT_FIELD = "text"
DEFAULT_SEED = 42


class ConfigError(Exception):
    """Raised when the mix config file is missing fields or has invalid values."""


@dataclass
class TokenizerClassRef:
    path: str
    class_name: str


@dataclass
class SourceConfig:
    name: str
    dataset: str
    subset: str = DEFAULT_SUBSET
    split: str = DEFAULT_SPLIT
    field: str = DEFAULT_FIELD
    ratio: float = 0.0


@dataclass
class MixConfig:
    tokenizer: str | TokenizerClassRef
    max_seq_len: int
    seed: int
    sources: list[SourceConfig]


def load_config(path: Path) -> MixConfig:
    """Parse and validate a legacy top-level mix TOML config file."""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML syntax: {exc}") from exc
    return _parse_body(raw, path)


def resolve_mix_config(path: Path) -> tuple[MixConfig, "Path | None"]:
    """Parse a mix config file that may be a task config.

    If the file has a ``[data]`` table, the mix config is parsed from that
    table (task-config form); the optional ``output`` entry and any custom
    tokenizer class path are resolved relative to the config file's
    directory, and the output dir (or None) is returned alongside.
    Otherwise the legacy top-level format is parsed unchanged and the
    output is None.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML syntax: {exc}") from exc

    data = raw.get("data")
    if data is None:
        return _parse_body(raw, path), None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: [data] must be a table, got {data!r}")

    config = _parse_body(data, path)
    if isinstance(config.tokenizer, TokenizerClassRef):
        config.tokenizer = TokenizerClassRef(
            path=str((path.parent / config.tokenizer.path).resolve()),
            class_name=config.tokenizer.class_name,
        )
    output = data.get("output")
    if output is None:
        return config, None
    if not isinstance(output, str):
        raise ConfigError(f"{path}: data.output must be a string, got {output!r}")
    return config, (path.parent / output).resolve()


def _parse_body(raw: dict, path: Path) -> MixConfig:
    """Parse and validate the mix fields of an already-decoded TOML table."""
    tokenizer = _parse_tokenizer(raw.get("tokenizer"), path)
    max_seq_len = _require(raw, "max_seq_len", path)
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ConfigError(f"{path}: max_seq_len must be a positive integer, "
                          f"got {max_seq_len!r}")
    seed = raw.get("seed", DEFAULT_SEED)

    raw_sources = raw.get("source")
    if not raw_sources:
        raise ConfigError(f"{path}: at least one [[source]] entry is required")

    sources: list[SourceConfig] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(raw_sources, start=1):
        where = f"{path}: source #{i}"
        name = _require(entry, "name", where)
        if name in seen_names:
            raise ConfigError(f"{where}: duplicate source name {name!r}")
        seen_names.add(name)
        ratio = _require(entry, "ratio", where)
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            raise ConfigError(f"{where}: ratio must be a positive number, "
                              f"got {ratio!r}")
        sources.append(SourceConfig(
            name=name,
            dataset=_require(entry, "dataset", where),
            subset=entry.get("subset", DEFAULT_SUBSET),
            split=entry.get("split", DEFAULT_SPLIT),
            field=entry.get("field", DEFAULT_FIELD),
            ratio=float(ratio),
        ))

    return MixConfig(tokenizer=tokenizer, max_seq_len=max_seq_len,
                     seed=seed, sources=sources)


def _parse_tokenizer(raw: object, path: Path) -> str | TokenizerClassRef:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) and "class" in raw:
        spec = raw["class"]
        if not isinstance(spec, str) or ":" not in spec:
            raise ConfigError(
                f"{path}: tokenizer.class must look like 'path/to.py:ClassName', "
                f"got {spec!r}")
        file_path, _, class_name = spec.rpartition(":")
        return TokenizerClassRef(path=file_path, class_name=class_name)
    raise ConfigError(f"{path}: tokenizer must be a preset name string or a "
                      f"[tokenizer] table with a 'class' entry, got {raw!r}")


def _require(entry: dict, key: str, where) -> object:
    value = entry.get(key)
    if value is None:
        raise ConfigError(f"{where}: missing required field {key!r}")
    return value
