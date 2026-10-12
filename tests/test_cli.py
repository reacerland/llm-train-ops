from typer.testing import CliRunner

from trainops.cli import app
from trainops.fetch.command import FetchResult

runner = CliRunner()


def _fake_run_fetch(captured, **overrides):
    def fake(dataset, subset, split, max_size, cache_dir, name=None):
        captured.update(dataset=dataset, subset=subset, split=split,
                        max_size=max_size, cache_dir=cache_dir, name=name)
        defaults = dict(cache_dir=cache_dir, records=10, bytes=2048, downloaded=True)
        return FetchResult(**(defaults | overrides))

    return fake


def test_fetch_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_fetch", _fake_run_fetch(captured))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["dataset"] == "org/ds"
    assert captured["subset"] is None
    assert captured["split"] == "train"
    assert captured["max_size"] == 2048
    assert captured["name"] is None
    assert "2048 bytes" in result.output


def test_fetch_command_with_name(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_fetch", _fake_run_fetch(captured))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--name", "my-ds",
                                 "--max-size", "2KB", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["name"] == "my-ds"


def test_fetch_command_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("trainops.cli.run_fetch",
                        _fake_run_fetch({}, downloaded=False))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "cache hit" in result.output


def test_fetch_command_dataset_exhausted_warns(tmp_path, monkeypatch):
    monkeypatch.setattr("trainops.cli.run_fetch",
                        _fake_run_fetch({}, dataset_exhausted=True))
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "dataset exhausted" in result.output


def test_fetch_command_invalid_size(tmp_path):
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "abc",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "invalid size" in result.output


def test_fetch_command_sdk_error_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, max_size, cache_dir, name=None):
        raise RuntimeError("dataset not found")

    monkeypatch.setattr("trainops.cli.run_fetch", failing)
    result = runner.invoke(app, ["fetch", "--dataset", "org/ds", "--max-size", "2KB",
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "org/ds" in result.output
    assert "dataset not found" in result.output
