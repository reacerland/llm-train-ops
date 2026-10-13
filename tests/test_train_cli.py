import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trainops.cli import app
from trainops.train.loop import TrainResult

from tests.test_mix_command import make_arrow_cache
from tests.test_train_sample import CHAR_TOKENIZER

runner = CliRunner()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def write_task_toml(tmp_path, total_tokens=1280, prompts=True):
    (tmp_path / "char_tok.py").write_text(CHAR_TOKENIZER, encoding="utf-8")
    gpt = (FIXTURES_DIR / "gpt.py").resolve()
    prompts_section = textwrap.dedent("""\

        [prompts]
        texts = ["ab"]
        max_new_tokens = 3
    """) if prompts else ""
    path = tmp_path / "task.toml"
    path.write_text(textwrap.dedent(f"""\
        [data]
        tokenizer = {{ class = 'char_tok.py:CharTok' }}
        max_seq_len = 16
        output = "mixture"
        [[data.source]]
        name = "web"
        dataset = "web-ds"
        ratio = 1.0

        [model]
        file = "{gpt}"
        class = "GPT"
        [model.args]
        vocab_size = 256
        d_model = 32
        n_layer = 1
        n_head = 2
        max_seq_len = 16

        [train]
        total_tokens = {total_tokens}
        batch_size = 8
        log_interval = 5
        save_interval = 5
        seed = 42
    """) + prompts_section, encoding="utf-8")
    return path


def _fake_run_training(captured, **overrides):
    def fake(config, mixture_dir, output_dir, *, resume=False, progress=True,
             _step_hook=None):
        captured.update(config=config, mixture_dir=mixture_dir,
                        output_dir=output_dir, resume=resume)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        defaults = dict(output_dir=output_dir, total_steps=10, final_step=10,
                        final_loss=1.23)
        return TrainResult(**(defaults | overrides))

    return fake


def test_train_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_training",
                        _fake_run_training(captured))
    cfg = write_task_toml(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(app, ["train", "--config", str(cfg),
                                 "--output", str(out),
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert captured["resume"] is False
    assert captured["output_dir"] == out
    # [data].output="mixture" resolved relative to the config file
    assert captured["mixture_dir"] == tmp_path / "mixture"
    assert "10" in result.output  # steps in summary


def test_train_resume_and_overwrite_conflict(tmp_path):
    cfg = write_task_toml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg),
                                 "--output", str(tmp_path / "out"),
                                 "--resume", "--overwrite"])
    assert result.exit_code == 2


def test_train_config_error_exit_2(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[train]\ntotal_tokens = 1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(bad),
                                 "--output", str(tmp_path / "out")])
    assert result.exit_code == 2


def test_train_output_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("trainops.cli.run_training",
                        _fake_run_training({}))
    cfg = write_task_toml(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(app, ["train", "--config", str(cfg),
                                 "--output", str(out)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output or "--resume" in result.output


def test_train_resume_without_checkpoint(tmp_path, monkeypatch):
    def fake(config, mixture_dir, output_dir, **kwargs):
        raise FileNotFoundError(f"checkpoint not found: {output_dir}")

    monkeypatch.setattr("trainops.cli.run_training", fake)
    cfg = write_task_toml(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(app, ["train", "--config", str(cfg),
                                 "--output", str(out), "--resume"])
    assert result.exit_code == 1
    assert "checkpoint" in result.output


def test_train_mixture_missing(tmp_path, monkeypatch):
    from trainops.train.data import MixtureNotFoundError

    def fake(config, mixture_dir, output_dir, **kwargs):
        raise MixtureNotFoundError(f"mixture not found; run mix first")

    monkeypatch.setattr("trainops.cli.run_training", fake)
    cfg = write_task_toml(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg),
                                 "--output", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "run mix first" in result.output


def test_end_to_end_mix_then_train(tmp_path):
    """Full pipeline: mix --config task.toml, then train on the mixture."""
    rows = ["the quick brown fox jumps over the lazy dog"] * 40
    make_arrow_cache(tmp_path / "cache", "web-ds", rows)
    cfg = write_task_toml(tmp_path)
    out = tmp_path / "out"

    r1 = runner.invoke(app, ["mix", "--config", str(cfg),
                             "--cache-dir", str(tmp_path / "cache")])
    assert r1.exit_code == 0, r1.output
    assert (tmp_path / "mixture" / "tokens.bin").is_file()

    r2 = runner.invoke(app, ["train", "--config", str(cfg),
                             "--output", str(out),
                             "--cache-dir", str(tmp_path / "cache")])
    assert r2.exit_code == 0, r2.output
    assert (out / "model_final.safetensors").is_file()
    assert (out / "loss.jsonl").is_file()
    assert (out / "checkpoint.pt").is_file()
    assert (out / "task.toml").is_file()
    samples = (out / "samples.txt").read_text(encoding="utf-8")
    assert ">>> ab" in samples
    losses = [json.loads(l)["loss"] for l in
              (out / "loss.jsonl").read_text().strip().splitlines()]
    assert len(losses) == 2  # 10 steps, log_interval 5
