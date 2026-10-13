"""Task config (task.toml) parsing for the train command.

One task config carries everything a CICD training task needs: the mix
dataset definition ([data]), the model python class file ([model]), the
training hyperparameters ([train]) and optional inference prompts
([prompts]). The [data] table is parsed by mix's resolve_mix_config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from trainops.mix.config import ConfigError, MixConfig, resolve_mix_config

DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
DEFAULT_WARMUP_STEPS = 0
DEFAULT_LOG_INTERVAL = 10
DEFAULT_SAVE_INTERVAL = 500
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 50


@dataclass
class ModelRef:
    file: str          # as written; resolved against the config dir at load time
    class_name: str
    args: dict[str, Any]


@dataclass
class TrainParams:
    total_tokens: int
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    log_interval: int = DEFAULT_LOG_INTERVAL
    save_interval: int = DEFAULT_SAVE_INTERVAL
    seed: int = DEFAULT_SEED


@dataclass
class PromptsConfig:
    texts: list[str]
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


@dataclass
class TaskConfig:
    path: Path
    data: MixConfig
    data_output: Path | None
    model: ModelRef
    train: TrainParams
    prompts: PromptsConfig | None


def load_task_config(path: Path) -> TaskConfig:
    """Parse and validate a task TOML config file."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML syntax: {exc}") from exc

    data, data_output = resolve_mix_config(path)
    model = _parse_model(raw.get("model"), path)
    train = _parse_train(raw.get("train"), path)
    prompts = _parse_prompts(raw.get("prompts"), path)
    return TaskConfig(path=path, data=data, data_output=data_output,
                      model=model, train=train, prompts=prompts)


def _parse_model(raw: object, path: Path) -> ModelRef:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: missing required [model] section")
    file = raw.get("file")
    if not isinstance(file, str) or not file:
        raise ConfigError(f"{path}: [model] missing required field 'file'")
    class_name = raw.get("class")
    if not isinstance(class_name, str) or not class_name:
        raise ConfigError(f"{path}: [model] missing required field 'class'")
    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise ConfigError(f"{path}: [model.args] must be a table, got {args!r}")
    return ModelRef(file=file, class_name=class_name, args=dict(args))


def _parse_train(raw: object, path: Path) -> TrainParams:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: missing required [train] section")
    total_tokens = raw.get("total_tokens")
    if not isinstance(total_tokens, int) or isinstance(total_tokens, bool) \
            or total_tokens <= 0:
        raise ConfigError(f"{path}: [train] total_tokens must be a positive "
                          f"integer, got {total_tokens!r}")
    batch_size = _positive_int(raw, "batch_size", DEFAULT_BATCH_SIZE, path)
    lr = raw.get("lr", DEFAULT_LR)
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise ConfigError(f"{path}: [train] lr must be a positive number, "
                          f"got {lr!r}")
    warmup_steps = _nonneg_int(raw, "warmup_steps", DEFAULT_WARMUP_STEPS, path)
    log_interval = _positive_int(raw, "log_interval", DEFAULT_LOG_INTERVAL, path)
    save_interval = _positive_int(raw, "save_interval", DEFAULT_SAVE_INTERVAL,
                                  path)
    seed = raw.get("seed", DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"{path}: [train] seed must be an integer, "
                          f"got {seed!r}")
    return TrainParams(total_tokens=total_tokens, batch_size=batch_size,
                       lr=float(lr), warmup_steps=warmup_steps,
                       log_interval=log_interval, save_interval=save_interval,
                       seed=seed)


def _parse_prompts(raw: object, path: Path) -> PromptsConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [prompts] must be a table, got {raw!r}")
    texts = raw.get("texts")
    if not isinstance(texts, list) or not texts \
            or not all(isinstance(t, str) for t in texts):
        raise ConfigError(f"{path}: [prompts] texts must be a non-empty list "
                          f"of strings, got {texts!r}")
    max_new_tokens = _positive_int(raw, "max_new_tokens",
                                   DEFAULT_MAX_NEW_TOKENS, path, where="[prompts]")
    return PromptsConfig(texts=list(texts), max_new_tokens=max_new_tokens)


def _positive_int(raw: dict, key: str, default: int, path: Path,
                  where: str = "[train]") -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path}: {where} {key} must be a positive integer, "
                          f"got {value!r}")
    return value


def _nonneg_int(raw: dict, key: str, default: int, path: Path,
                where: str = "[train]") -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path}: {where} {key} must be a non-negative "
                          f"integer, got {value!r}")
    return value
