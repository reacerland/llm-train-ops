import textwrap

import pytest

from trainops.mix.config import ConfigError, resolve_mix_config
from trainops.mix.command import run_mix

from tests.test_mix_command import TINY_TOKENIZER, make_arrow_cache


def write_task_config(tmp_path, tokenizer_class, sources, output=None,
                      stem="task", seq_len=4, seed=42):
    """Write a task-style config: mix settings nested under a [data] table."""
    toml = f"""
        [data]
        tokenizer = {{ class = '{tokenizer_class}' }}
        max_seq_len = {seq_len}
        seed = {seed}
    """
    if output is not None:
        toml += f'    output = "{output}"\n'
    for name, dataset, ratio, rows in sources:
        make_arrow_cache(tmp_path / "cache", dataset, rows)
        toml += f"""
        [[data.source]]
        name = "{name}"
        dataset = "{dataset}"
        ratio = {ratio}
    """
    path = tmp_path / f"{stem}.toml"
    path.write_text(textwrap.dedent(toml), encoding="utf-8")
    return path


def write_legacy_config(tmp_path, tokenizer_class, sources, stem="mix",
                        seq_len=4, seed=42):
    """Write a legacy top-level mix config (no [data] table)."""
    toml = f"""
        tokenizer = {{ class = '{tokenizer_class}' }}
        max_seq_len = {seq_len}
        seed = {seed}
    """
    for name, dataset, ratio, rows in sources:
        make_arrow_cache(tmp_path / "cache", dataset, rows)
        toml += f"""
        [[source]]
        name = "{name}"
        dataset = "{dataset}"
        ratio = {ratio}
    """
    path = tmp_path / f"{stem}.toml"
    path.write_text(textwrap.dedent(toml), encoding="utf-8")
    return path


WEB_ROWS = ["a bb ccc", "dd ee ff", "gg hh ii", "jj kk ll"]


def test_resolve_mix_config_task_format(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    cfg = write_task_config(tmp_path, f"{tok}:TinyTokenizer", [("web", "web-ds", 1.0, WEB_ROWS)],
                            output="rel/mixture")
    config, output = resolve_mix_config(cfg)
    assert config.max_seq_len == 4
    assert config.tokenizer.path == str(tok)  # resolved against config dir
    assert output == tmp_path / "rel" / "mixture"


def test_resolve_mix_config_task_format_no_output(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    cfg = write_task_config(tmp_path, f"{tok}:TinyTokenizer", [("web", "web-ds", 1.0, WEB_ROWS)])
    config, output = resolve_mix_config(cfg)
    assert output is None


def test_resolve_mix_config_legacy_format(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    cfg = write_legacy_config(tmp_path, f"{tok}:TinyTokenizer", [("web", "web-ds", 1.0, WEB_ROWS)])
    config, output = resolve_mix_config(cfg)
    assert config.max_seq_len == 4
    assert output is None
    # legacy: tokenizer class path stays as written (not resolved)
    assert config.tokenizer.path == str(tok)


def test_run_mix_uses_data_output(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    cfg = write_task_config(tmp_path, f"{tok}:TinyTokenizer", [("web", "web-ds", 1.0, WEB_ROWS)],
                            output="rel/mixture")
    result = run_mix(cfg, tmp_path / "cache", progress=False)
    assert result.output_dir == tmp_path / "rel" / "mixture"
    assert (result.output_dir / "tokens.bin").is_file()


def test_run_mix_cli_output_overrides_config(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    cfg = write_task_config(tmp_path, f"{tok}:TinyTokenizer", [("web", "web-ds", 1.0, WEB_ROWS)],
                            output="rel/mixture")
    result = run_mix(cfg, tmp_path / "cache", output=tmp_path / "elsewhere",
                     progress=False)
    assert result.output_dir == tmp_path / "elsewhere"


def test_task_format_invalid_source_still_raises(tmp_path):
    tok = tmp_path / "tok.py"
    tok.write_text(TINY_TOKENIZER, encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(textwrap.dedent(f"""
        [data]
        tokenizer = {{ class = '{tok}:TinyTokenizer' }}
        max_seq_len = 4
    """), encoding="utf-8")
    with pytest.raises(ConfigError, match="source"):
        resolve_mix_config(bad)
