import textwrap

import pytest

from trainops.mix.config import (
    ConfigError,
    MixConfig,
    SourceConfig,
    TokenizerClassRef,
    load_config,
)


def write_config(tmp_path, toml: str):
    path = tmp_path / "mix.toml"
    path.write_text(textwrap.dedent(toml), encoding="utf-8")
    return path


def test_load_valid_config_defaults(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"
        max_seq_len = 4096

        [[source]]
        name = "web"
        dataset = "my-web"
        ratio = 0.7

        [[source]]
        name = "code"
        dataset = "my-code"
        subset = "s1"
        split = "test"
        field = "content"
        ratio = 0.3
    """)
    cfg = load_config(path)
    assert isinstance(cfg, MixConfig)
    assert cfg.tokenizer == "llama2"
    assert cfg.max_seq_len == 4096
    assert cfg.seed == 42  # 缺省
    assert len(cfg.sources) == 2
    web, code = cfg.sources
    assert web == SourceConfig(name="web", dataset="my-web", ratio=0.7)
    assert web.subset == "default" and web.split == "train" and web.field == "text"
    assert code.subset == "s1" and code.split == "test" and code.field == "content"


def test_tokenizer_class_table_form(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = { class = "my_tokenizers/my_bpe.py:MyTokenizer" }
        max_seq_len = 512

        [[source]]
        name = "web"
        dataset = "my-web"
        ratio = 1.0
    """)
    cfg = load_config(path)
    assert cfg.tokenizer == TokenizerClassRef(
        path="my_tokenizers/my_bpe.py", class_name="MyTokenizer")


def test_missing_max_seq_len(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"

        [[source]]
        name = "web"
        dataset = "my-web"
        ratio = 1.0
    """)
    with pytest.raises(ConfigError, match="max_seq_len"):
        load_config(path)


def test_missing_ratio(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"
        max_seq_len = 512

        [[source]]
        name = "web"
        dataset = "my-web"
    """)
    with pytest.raises(ConfigError, match="ratio"):
        load_config(path)


@pytest.mark.parametrize("ratio", ["0", "-0.5"])
def test_ratio_not_positive(tmp_path, ratio):
    path = write_config(tmp_path, f"""
        tokenizer = "llama2"
        max_seq_len = 512

        [[source]]
        name = "web"
        dataset = "my-web"
        ratio = {ratio}
    """)
    with pytest.raises(ConfigError, match="ratio"):
        load_config(path)


def test_empty_sources(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"
        max_seq_len = 512
    """)
    with pytest.raises(ConfigError, match="source"):
        load_config(path)


def test_duplicate_source_names(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"
        max_seq_len = 512

        [[source]]
        name = "web"
        dataset = "a"
        ratio = 0.5

        [[source]]
        name = "web"
        dataset = "b"
        ratio = 0.5
    """)
    with pytest.raises(ConfigError, match="web"):
        load_config(path)


def test_invalid_toml_syntax(tmp_path):
    path = tmp_path / "mix.toml"
    path.write_text("tokenizer = 'llama2'\nmax_seq_len = \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="TOML|toml|syntax|parse"):
        load_config(path)


def test_unnormalized_ratios_accepted(tmp_path):
    path = write_config(tmp_path, """
        tokenizer = "llama2"
        max_seq_len = 512

        [[source]]
        name = "web"
        dataset = "a"
        ratio = 7

        [[source]]
        name = "code"
        dataset = "b"
        ratio = 3
    """)
    cfg = load_config(path)
    assert [s.ratio for s in cfg.sources] == [7, 3]  # 不归一化
