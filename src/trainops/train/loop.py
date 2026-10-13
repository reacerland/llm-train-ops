"""Training loop: tokens -> model -> loss.jsonl + checkpoints + final weights."""

import json
import math
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TaskConfig
from .data import Batcher, load_mixture
from .model import load_model

LOSS_LOG_NAME = "loss.jsonl"
CHECKPOINT_NAME = "checkpoint.pt"
FINAL_WEIGHTS_NAME = "model_final.safetensors"
CONFIG_SNAPSHOT_NAME = "task.toml"
SAMPLES_NAME = "samples.txt"


@dataclass
class TrainResult:
    output_dir: Path
    total_steps: int
    final_step: int
    final_loss: float


def run_training(config: TaskConfig, mixture_dir: Path, output_dir: Path,
                 *, resume: bool = False, progress: bool = True,
                 _step_hook: Callable[[int], None] | None = None
                 ) -> TrainResult:
    """Train the model from a task config on a mix-produced mixture.

    Deterministic given the same config: torch/numpy/python RNGs are seeded
    from [train].seed and the Batcher reproduces its sequence on --resume.
    """
    output_dir = Path(output_dir)
    params = config.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    mixture = load_mixture(mixture_dir)
    model = load_model(config.model, config.path.parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = math.ceil(
        params.total_tokens / (params.batch_size * mixture.max_seq_len))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_scale(s, params.warmup_steps, total_steps))
    batcher = Batcher(mixture, params.batch_size, params.seed)

    prompts = None
    if config.prompts is not None:
        from ..mix.tokenizer import load_tokenizer
        from .sample import generate_samples, require_decode
        tokenizer = load_tokenizer(config.data.tokenizer)
        require_decode(tokenizer)  # fail fast, before any training
        prompts = (tokenizer, generate_samples)

    start_step = 0
    if resume:
        start_step = load_checkpoint(
            output_dir / CHECKPOINT_NAME, model=model, optimizer=optimizer,
            scheduler=scheduler, batcher=batcher, device=device)
    model = model.to(device)
    model.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / LOSS_LOG_NAME
    tokens_per_step = params.batch_size * mixture.max_seq_len
    start_time = time.perf_counter()
    last_loss = float("nan")

    for step in range(start_step + 1, total_steps + 1):
        batch = torch.from_numpy(batcher.next_batch()).to(device)
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_bf16):
            logits = model(input_ids)
            if logits.size(-1) != mixture.vocab_size:
                raise ValueError(
                    f"model outputs vocab dim {logits.size(-1)} but the "
                    f"mixture was built with vocab_size={mixture.vocab_size}; "
                    f"model vocab and data tokenizer do not match")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()

        if step % params.log_interval == 0 or step == total_steps:
            elapsed = time.perf_counter() - start_time
            entry = {
                "step": step,
                "loss": last_loss,
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_sec": round(step * tokens_per_step / elapsed, 1),
                "elapsed_sec": round(elapsed, 2),
            }
            with open(loss_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if progress:
                print(f"step {step}/{total_steps} loss={last_loss:.4f} "
                      f"lr={entry['lr']:.2e} tok/s={entry['tokens_per_sec']:.0f}",
                      flush=True)

        if step % params.save_interval == 0 or step == total_steps:
            save_checkpoint(output_dir / CHECKPOINT_NAME, model=model,
                            optimizer=optimizer, scheduler=scheduler,
                            step=step, batcher=batcher)

        if _step_hook is not None:
            _step_hook(step)

    from safetensors.torch import save_file
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()},
              str(output_dir / FINAL_WEIGHTS_NAME))
    shutil.copy(config.path, output_dir / CONFIG_SNAPSHOT_NAME)

    if prompts is not None:
        tokenizer, generate_samples = prompts
        generate_samples(model, tokenizer, config.prompts, device,
                         output_dir / SAMPLES_NAME)

    return TrainResult(output_dir=output_dir, total_steps=total_steps,
                       final_step=total_steps, final_loss=last_loss)


def _lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 10% of the base lr."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
