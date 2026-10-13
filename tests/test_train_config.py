import textwrap

import pytest

from trainops.mix.config import ConfigError
from trainops.train.config import load_task_config


def write_toml(tmp_path, extra_train="", prompts=True, model=True, data=True,
               stem="task"):
    parts = []
    if data:
        parts.append(textwrap.dedent("""\
            [data]
            tokenizer = "llama2"
            max_seq_len = 16
            [[data.source]]
            name = "web"
            dataset = "web-ds"
            ratio = 1.0
        """))
    if model:
        parts.append(textwrap.dedent("""\
            [model]
            file = "gpt.py"
            class = "GPT"
            [model.args]
            vocab_size = 100
            d_model = 32
        """))
    parts.append(textwrap.dedent("""\
        [train]
        total_tokens = 2560
    """) + extra_train)
    if prompts:
        parts.append(textwrap.dedent("""\
            [prompts]
            texts = ["hello"]
        """))
    path = tmp_path / f"{stem}.toml"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def test_load_task_config_full(tmp_path):
    cfg = load_task_config(write_toml(tmp_path))
    assert cfg.path == tmp_path / "task.toml"
    assert cfg.data.max_seq_len == 16
    assert cfg.data_output is None
    assert cfg.model.file == "gpt.py"
    assert cfg.model.class_name == "GPT"
    assert cfg.model.args == {"vocab_size": 100, "d_model": 32}
    assert cfg.train.total_tokens == 2560
    assert cfg.train.batch_size == 8          # default
    assert cfg.prompts.texts == ["hello"]
    assert cfg.prompts.max_new_tokens == 50   # default


def test_load_task_config_defaults_and_no_prompts(tmp_path):
    cfg = load_task_config(write_toml(tmp_path, prompts=False))
    assert cfg.prompts is None
    assert cfg.train.lr == 3e-4
    assert cfg.train.seed == 42
    assert cfg.train.warmup_steps == 0
    assert cfg.train.log_interval == 10
    assert cfg.train.save_interval == 500


def test_train_overrides(tmp_path):
    cfg = load_task_config(write_toml(
        tmp_path, extra_train='batch_size = 4\nlr = 1e-3\nseed = 7\n'))
    assert cfg.train.batch_size == 4
    assert cfg.train.lr == 1e-3
    assert cfg.train.seed == 7


def test_data_output_propagated(tmp_path):
    path = write_toml(tmp_path)
    text = path.read_text().replace('[data]\n', '[data]\noutput = "mixdir"\n')
    path.write_text(text, encoding="utf-8")
    cfg = load_task_config(path)
    assert cfg.data_output == tmp_path / "mixdir"


def test_missing_model_section(tmp_path):
    with pytest.raises(ConfigError, match="\[model\]"):
        load_task_config(write_toml(tmp_path, model=False))


def test_missing_model_file_key(tmp_path):
    path = write_toml(tmp_path)
    path.write_text(path.read_text().replace('file = "gpt.py"\n', ''),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="file"):
        load_task_config(path)


def test_missing_train_section(tmp_path):
    path = write_toml(tmp_path)
    text = path.read_text()
    path.write_text(text.replace("[train]\ntotal_tokens = 2560\n", ""),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="\[train\]"):
        load_task_config(path)


def test_missing_total_tokens(tmp_path):
    path = write_toml(tmp_path)
    path.write_text(path.read_text().replace("total_tokens = 2560\n", ""),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="total_tokens"):
        load_task_config(path)


def test_invalid_total_tokens(tmp_path):
    path = write_toml(tmp_path)
    path.write_text(path.read_text().replace("total_tokens = 2560",
                                             "total_tokens = -5"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="total_tokens"):
        load_task_config(path)


def test_invalid_lr_type(tmp_path):
    with pytest.raises(ConfigError, match="lr"):
        load_task_config(write_toml(tmp_path, extra_train='lr = "fast"\n'))


def test_prompts_empty_texts(tmp_path):
    path = write_toml(tmp_path)
    path.write_text(path.read_text().replace('texts = ["hello"]', "texts = []"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="texts"):
        load_task_config(path)
