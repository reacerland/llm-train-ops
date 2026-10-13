"""Periodic held-out evaluation for the train loop: PPL + top-k accuracy.

Each call evaluates a fixed, seeded selection of batches — the same batches
at every eval point, so metrics are comparable across steps and identical
across reruns with the same seed. Runs under no_grad in the model's eval
mode and restores the previous training mode afterwards.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import Mixture

TOP_K = (1, 5)


def evaluate_split(model: torch.nn.Module, mixture: Mixture,
                   batch_size: int, num_batches: int, seed: int,
                   device: torch.device, use_bf16: bool) -> dict:
    """Evaluate next-token metrics on ``num_batches`` seeded batches.

    Returns {"loss", "ppl", "top1", "top5", "tokens"} where the accuracies
    are the fraction of label positions whose gold token is in the model's
    top-k predictions, and ppl is exp(mean token-level cross-entropy).
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mixture.num_sequences,
                       size=num_batches * batch_size)

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = {k: 0 for k in TOP_K}
    tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, len(idx), batch_size):
                seqs = np.asarray(
                    mixture.sequences[idx[start:start + batch_size]],
                    dtype=np.int64)
                batch = torch.from_numpy(seqs).to(device)
                input_ids, labels = batch[:, :-1], batch[:, 1:]
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(input_ids)
                    loss_sum += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1), reduction="sum").item()
                    topk = logits.topk(max(TOP_K), dim=-1).indices
                for k in TOP_K:
                    correct[k] += int(
                        (topk[..., :k] == labels.unsqueeze(-1))
                        .any(dim=-1).sum().item())
                tokens += labels.numel()
    finally:
        if was_training:
            model.train()

    loss = loss_sum / tokens
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 100.0)),   # clamp: exp overflow guard
        "top1": correct[1] / tokens,
        "top5": correct[5] / tokens,
        "tokens": tokens,
    }
