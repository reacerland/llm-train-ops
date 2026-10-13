"""Checkpoint save/restore: full training state for --resume.

A checkpoint carries model, optimizer, scheduler, batcher position and all
RNG states, so a resumed run follows the exact trajectory of an
uninterrupted one. Writes are atomic (tmp file + os.replace): an interrupt
mid-save never corrupts the previous checkpoint.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

from .data import Batcher


def save_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    step: int, batcher: Batcher) -> None:
    path = Path(path)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "batcher": batcher.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    batcher: Batcher) -> int:
    """Restore full training state from ``path``; returns the saved step.

    Loads onto CPU; the caller moves the model back to its device.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    batcher.load_state_dict(state["batcher"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]
