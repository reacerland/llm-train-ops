import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from trainops.train.config import load_task_config
from trainops.train.data import MixtureNotFoundError
from trainops.train.loop import run_training

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def write_mixture(tmp_path, num_seqs=200, seq_len=16, vocab=50,
                  valid=False):
    """Patterned data: each row is consecutive ids mod vocab — learnable."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    starts = np.random.default_rng(0).integers(0, vocab, size=num_seqs)
    seqs = (starts[:, None] + np.arange(seq_len)[None, :]) % vocab
    (tmp_path / "tokens.bin").write_bytes(seqs.astype(np.uint16).tobytes())
    meta = {"format": "trainops-mix/v1", "dtype": "uint16",
            "max_seq_len": seq_len, "vocab_size": vocab,
            "num_sequences": num_seqs}
    if valid:
        (tmp_path / "valid_tokens.bin").write_bytes(
            seqs.astype(np.uint16).tobytes())
        meta["valid"] = {"num_sequences": num_seqs}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def write_config(tmp_path, total_tokens=2560, vocab=50, save_interval=6,
                 log_interval=10, lr="5e-3", stem="task", extra_train=""):
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
    """) + extra_train, encoding="utf-8")
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


def test_eval_writes_eval_jsonl(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix", valid=True)
    cfg = write_config(tmp_path,
                       extra_train='eval_interval = 15\neval_batches = 2\n')
    out = tmp_path / "out"
    result = run_training(cfg, mixture_dir, out, progress=False)
    assert result.total_steps == 20

    lines = read_jsonl(out / "eval.jsonl")
    # eval_interval=15 命中 step 15；最终 step 20 也必评一次 × valid/train
    assert {(e["step"], e["split"]) for e in lines} == \
        {(15, "valid"), (15, "train"), (20, "valid"), (20, "train")}
    for e in lines:
        assert e["loss"] > 0 and e["ppl"] > 1
        assert 0.0 <= e["top1"] <= e["top5"] <= 1.0
        assert e["tokens"] == 2 * 8 * 15   # eval_batches × batch × (seq_len-1)
    # 收敛性：最终 valid loss 明显低于初始水平（随机数据 ln(50)≈3.9）
    final_valid = [e for e in lines
                   if e["step"] == 20 and e["split"] == "valid"][0]
    assert final_valid["loss"] < 3.0
    assert result.final_eval is not None
    assert result.final_eval["split"] == "valid"
    assert result.final_eval["ppl"] > 1


def test_eval_disabled_by_default(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix", valid=True)
    cfg = write_config(tmp_path)
    out = tmp_path / "out"
    result = run_training(cfg, mixture_dir, out, progress=False)
    assert not (out / "eval.jsonl").exists()
    assert result.final_eval is None


def test_eval_interval_without_valid_mixture_errors(tmp_path):
    mixture_dir = write_mixture(tmp_path / "mix")   # 无 valid_tokens.bin
    cfg = write_config(tmp_path, extra_train='eval_interval = 5\n')
    with pytest.raises(MixtureNotFoundError, match="valid"):
        run_training(cfg, mixture_dir, tmp_path / "out", progress=False)


def test_eval_resume_consistent(tmp_path):
    """resume 后 eval 点与指标与一次性跑完一致（eval 批次由 seed 固定）"""
    mixture_dir = write_mixture(tmp_path / "mix", valid=True)
    cfg = write_config(tmp_path,
                       extra_train='eval_interval = 10\neval_batches = 2\n')
    full = run_training(cfg, mixture_dir, tmp_path / "out_a", progress=False)

    def boom(step):
        if step == 12:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(cfg, mixture_dir, tmp_path / "out_b", _step_hook=boom,
                     progress=False)
    run_training(cfg, mixture_dir, tmp_path / "out_b", resume=True,
                 progress=False)
    a = read_jsonl(tmp_path / "out_a" / "eval.jsonl")
    b = read_jsonl(tmp_path / "out_b" / "eval.jsonl")
    assert [(e["step"], e["split"]) for e in a] == \
        [(e["step"], e["split"]) for e in b]
    for ea, eb in zip(a, b):
        assert ea["loss"] == pytest.approx(eb["loss"], rel=1e-3)
        assert ea["top1"] == pytest.approx(eb["top1"], rel=1e-3)
