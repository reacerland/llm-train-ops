"""Model loading for train: duck-typed nn.Module from a user python file.

The train command talks to models through a duck-typed interface — no
inheritance required:

    class MyModel(torch.nn.Module):
        def __init__(self, **kwargs): ...    # kwargs come from [model.args]
        def forward(self, input_ids): ...    # LongTensor [B, T] -> logits [B, T, V]
"""

import importlib.util
import sys
from pathlib import Path

import torch

from .config import ModelRef


class ModelLoadError(Exception):
    """Raised when a model class file cannot be loaded or validated."""


def load_model(ref: ModelRef, config_dir: Path) -> torch.nn.Module:
    """Import ``ref.class_name`` from ``ref.file`` and instantiate with ref.args.

    ``ref.file`` is resolved relative to ``config_dir`` (the task config
    file's directory). Any failure raises ModelLoadError naming the problem.
    """
    path = (Path(config_dir) / ref.file).resolve()
    if not path.is_file():
        raise ModelLoadError(f"model class file not found: {path}")
    module_name = f"trainops_user_model_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelLoadError(f"cannot import model class file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ModelLoadError(
            f"failed to import model class file {path}: {exc}") from exc
    try:
        cls = getattr(module, ref.class_name)
    except AttributeError:
        raise ModelLoadError(
            f"model class {ref.class_name!r} not found in {path}") from None
    try:
        model = cls(**ref.args)
    except Exception as exc:
        raise ModelLoadError(
            f"failed to instantiate model class {ref.class_name!r} with "
            f"[model.args]: {exc}") from exc
    if not isinstance(model, torch.nn.Module):
        raise ModelLoadError(
            f"model class {ref.class_name!r} in {path} is not a "
            f"torch.nn.Module instance")
    return model
