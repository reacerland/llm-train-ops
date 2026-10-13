from typer.testing import CliRunner

from trainops.cli import app
from trainops.convert.command import (
    CacheNotFoundError,
    ConvertResult,
    OutputExistsError,
)
from trainops.convert.transform import MissingFieldError
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


def _fake_run_convert(captured):
    def fake(dataset, subset, split, fields, cache_dir, output=None, skip_bad=False,
             overwrite=False, name=None):
        captured.update(dataset=dataset, subset=subset, split=split, fields=fields,
                        cache_dir=cache_dir, output=output, skip_bad=skip_bad,
                        overwrite=overwrite, name=name)
        return ConvertResult(output_dir=output or cache_dir / "data.arrow",
                             records=10, skipped=0, bytes=4096)

    return fake


def test_convert_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_convert", _fake_run_convert(captured))
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text, meta", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["dataset"] == "org/ds"
    assert captured["subset"] is None
    assert captured["split"] == "train"
    assert captured["fields"] == ["text", "meta"]
    assert captured["output"] is None
    assert captured["skip_bad"] is False
    assert captured["overwrite"] is False
    assert "10 records" in result.output
    assert "data.arrow" in result.output


def test_convert_command_flags(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_convert", _fake_run_convert(captured))
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--name", "my-ds", "--subset", "s1", "--split", "test",
                                 "--fields", "text", "--skip-bad", "--overwrite",
                                 "--output", str(tmp_path / "out"),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["name"] == "my-ds"
    assert captured["subset"] == "s1"
    assert captured["split"] == "test"
    assert captured["skip_bad"] is True
    assert captured["overwrite"] is True
    assert captured["output"] == tmp_path / "out"


def test_convert_command_skipped_warning(tmp_path, monkeypatch):
    captured = {}
    fake = _fake_run_convert(captured)

    def fake_with_skips(*args, **kwargs):
        result = fake(*args, **kwargs)
        result.skipped = 3
        return result

    monkeypatch.setattr("trainops.cli.run_convert", fake_with_skips)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "skipped 3" in result.output


def test_convert_command_empty_fields_exit_2(tmp_path):
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", " , ", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "empty" in result.output


def test_convert_command_cache_not_found_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise CacheNotFoundError("run fetch first")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "fetch" in result.output


def test_convert_command_output_exists_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise OutputExistsError("output directory already exists: /x")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output


def test_convert_command_missing_field_exit_1(tmp_path, monkeypatch):
    def failing(dataset, subset, split, fields, cache_dir, **kwargs):
        raise MissingFieldError(3, "title")

    monkeypatch.setattr("trainops.cli.run_convert", failing)
    result = runner.invoke(app, ["convert", "--dataset", "org/ds",
                                 "--fields", "text", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "title" in result.output


from trainops.mix.command import MixCacheNotFoundError, MixResult  # noqa: E402
from trainops.mix.config import ConfigError  # noqa: E402
from trainops.mix.tokenizer import TokenizerLoadError  # noqa: E402


def _fake_run_mix(captured):
    def fake(config_path, cache_dir, output=None, overwrite=False, progress=True):
        captured.update(config_path=config_path, cache_dir=cache_dir,
                        output=output, overwrite=overwrite)
        return MixResult(
            output_dir=output or cache_dir / "mixtures" / "mix",
            num_sequences=128, total_tokens=128 * 8, dropped_tail_tokens=3,
            exhausted_sources=[], sources_meta=[
                {"name": "web", "configured_ratio": 0.6, "docs_total": 10,
                 "docs_sampled": 8, "tokens": 600, "dataset": "web-alias",
                 "subset": "default", "split": "train"},
            ], bytes=4096)

    return fake


def test_mix_command_success(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("trainops.cli.run_mix", _fake_run_mix(captured))
    config = tmp_path / "mix.toml"
    config.write_text("tokenizer = 'llama2'\n", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["config_path"] == config
    assert captured["cache_dir"] == tmp_path
    assert captured["output"] is None
    assert captured["overwrite"] is False
    assert "128 sequences" in result.output
    assert "60.00%" in result.output


def test_mix_command_exhausted_warning(tmp_path, monkeypatch):
    def fake(config_path, cache_dir, output=None, overwrite=False, progress=True):
        result = _fake_run_mix({})(config_path, cache_dir, output, overwrite,
                                   progress)
        result.exhausted_sources = ["code"]
        return result

    monkeypatch.setattr("trainops.cli.run_mix", fake)
    config = tmp_path / "mix.toml"
    config.write_text("tokenizer = 'llama2'\n", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "exhausted" in result.output
    assert "code" in result.output


def test_mix_command_config_error_exit_1(tmp_path, monkeypatch):
    def failing(config_path, cache_dir, output=None, overwrite=False,
                progress=True):
        raise ConfigError("mix.toml: missing required field 'max_seq_len'")

    monkeypatch.setattr("trainops.cli.run_mix", failing)
    config = tmp_path / "mix.toml"
    config.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "max_seq_len" in result.output


def test_mix_command_cache_not_found_exit_1(tmp_path, monkeypatch):
    def failing(config_path, cache_dir, output=None, overwrite=False,
                progress=True):
        raise MixCacheNotFoundError("run convert first")

    monkeypatch.setattr("trainops.cli.run_mix", failing)
    config = tmp_path / "mix.toml"
    config.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "convert" in result.output


def test_mix_command_output_exists_exit_1(tmp_path, monkeypatch):
    def failing(config_path, cache_dir, output=None, overwrite=False,
                progress=True):
        raise OutputExistsError("output directory already exists: /x")

    monkeypatch.setattr("trainops.cli.run_mix", failing)
    config = tmp_path / "mix.toml"
    config.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output


def test_mix_command_tokenizer_error_exit_1(tmp_path, monkeypatch):
    def failing(config_path, cache_dir, output=None, overwrite=False,
                progress=True):
        raise TokenizerLoadError("tokenizer class file not found: /x/tok.py")

    monkeypatch.setattr("trainops.cli.run_mix", failing)
    config = tmp_path / "mix.toml"
    config.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["mix", "--config", str(config),
                                 "--cache-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "tok.py" in result.output
