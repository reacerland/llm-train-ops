import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from trainops.train.config import load_task_config
from trainops.train.loop import run_training

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def write_mixture(tmp_path, num_seqs=200, seq_len=16, vocab=50):
    """Patterned data: each row is consecutive ids mod vocab — learnable."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    starts = np.random.default_rng(0).integers(0, vocab, size=num_seqs)
    seqs = (starts[:, None] + np.arange(seq_len)[None, :]) % vocab
    (tmp_path / "tokens.bin").write_bytes(seqs.astype(np.uint16).tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"format": "trainops-mix/v1", "dtype": "uint16",
         "max_seq_len": seq_len, "vocab_size": vocab,
         "num_sequences": num_seqs}), encoding="utf-8")
    return tmp_path


def write_config(tmp_path, total_tokens=2560, vocab=50, save_interval=6,
                 log_interval=10, lr="5e-3", stem="task"):
    gpt = (FIXTURES_DIR / "gpt.py").resolve()
    path = tmp_path / f"{stem}.toml"
    path.write_text(textwrap.dedent(f"""\
        [data]
        tokenizer = "llama2"
        max_seq_len = 16
        [[data.source]]
        name = "web"
        dataset = "web-ds"
        ratio = 1.0

        [model]
        file = "{gpt}"
        class = "GPT"
        [model.args]
        vocab_size = {vocab}
        d_model = 32
        n_layer = 2
        n_head = 2
        max_seq_len = 16

        [train]
        total_tokens = {total_tokens}
        batch_size = 8
        lr = {lr}
        log_interval = {log_interval}
        save_interval = {save_interval}
        seed = 42
    """), encoding="utf-8")
    return load_task_config(path)


def read_jsonl(path):
    return [json.loads(line) for line in
            path.read_text().strip().splitlines()]


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Deterministic fp32 CPU trajectory (CUDA embedding backward is
    nondeterministic; GPU coverage is a separate smoke test)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


def test_loss_decreases_and_artifacts(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix")
    cfg = write_config(tmp_path)
    out = tmp_path / "out"
    result = run_training(cfg, mixture_dir, out, progress=False)
    assert result.total_steps == 20          # 2560 / (8*16)
    assert result.final_step == 20
    lines = read_jsonl(out / "loss.jsonl")
    assert len(lines) == 2                   # log_interval=10 over 20 steps
    assert lines[0]["step"] == 10 and lines[1]["step"] == 20
    assert lines[-1]["loss"] < lines[0]["loss"]
    assert (out / "model_final.safetensors").is_file()
    assert (out / "checkpoint.pt").is_file()
    assert (out / "task.toml").is_file()


def test_total_tokens_rounds_up(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix")
    cfg = write_config(tmp_path, total_tokens=2561)
    result = run_training(cfg, mixture_dir, tmp_path / "out", progress=False)
    assert result.total_steps == 21


def test_vocab_mismatch(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix", vocab=50)
    cfg = write_config(tmp_path, vocab=60)   # model vocab != data vocab
    with pytest.raises(ValueError, match="vocab"):
        run_training(cfg, mixture_dir, tmp_path / "out", progress=False)


def test_resume_matches_uninterrupted(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix")
    cfg = write_config(tmp_path)
    full = run_training(cfg, mixture_dir, tmp_path / "out_a", progress=False)

    def boom(step):
        if step == 12:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(cfg, mixture_dir, tmp_path / "out_b", _step_hook=boom,
                     progress=False)
    resumed = run_training(cfg, mixture_dir, tmp_path / "out_b", resume=True,
                           progress=False)
    assert resumed.final_step == full.final_step
    assert resumed.final_loss == pytest.approx(full.final_loss, rel=1e-3)
    assert len(read_jsonl(tmp_path / "out_b" / "loss.jsonl")) == \
        len(read_jsonl(tmp_path / "out_a" / "loss.jsonl"))


def test_resume_without_checkpoint(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix")
    cfg = write_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_training(cfg, mixture_dir, out, resume=True, progress=False)
