import hashlib
import json
import textwrap

import pytest
from datasets import Dataset

from trainops.convert.command import OutputExistsError
from trainops.mix.command import MixCacheNotFoundError, run_mix


TINY_TOKENIZER = textwrap.dedent("""
    class TinyTokenizer:
        eos_token_id = 9
        vocab_size = 100

        def encode(self, text):
            return [len(w) for w in text.split()]
""")


def write_tokenizer(tmp_path, source=TINY_TOKENIZER, name="tok.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def make_arrow_cache(cache_dir, alias, rows, subset="default", split="train"):
    arrow = cache_dir / "datasets" / alias / subset / split / "data.arrow"
    arrow.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"text": rows}).save_to_disk(str(arrow))
    return arrow


def write_config(tmp_path, tokenizer_class, sources, stem="mix", seq_len=4,
                 seed=42):
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


WEB_ROWS = ["a bb ccc", "dd ee ff", "gg hh ii", "jj kk ll", "mm nn oo",
            "pp qq rr", "ss tt uu", "vv ww xx", "yy zz aa", "bb cc dd",
            "ee ff gg", "hh ii jj"]          # 每篇 4 token（含 EOS），≥ seq_len
CODE_ROWS = ["xy zzz", "q r s"]             # 小源，必然先抽干


@pytest.fixture
def mix_setup(tmp_path):
    tok = write_tokenizer(tmp_path)
    config = write_config(
        tmp_path, f"{tok}:TinyTokenizer",
        [("web", "web-alias", 0.6, WEB_ROWS), ("code", "code-alias", 0.4, CODE_ROWS)])
    return tmp_path, config


def read_meta(output_dir):
    return json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))


def test_end_to_end_meta_and_tokens(mix_setup):
    tmp_path, config = mix_setup
    cache_dir = tmp_path / "cache"
    result = run_mix(config, cache_dir, progress=False)

    assert result.output_dir == cache_dir / "mixtures" / "mix"
    tokens_path = result.output_dir / "tokens.bin"
    assert tokens_path.is_file()
    meta = read_meta(result.output_dir)
    assert meta["format"] == "trainops-mix/v1"
    assert meta["vocab_size"] == 100
    assert meta["dtype"] == "uint16"
    assert meta["max_seq_len"] == 4
    assert meta["seed"] == 42
    itemsize = 2  # uint16, vocab 100 < 65536
    size = tokens_path.stat().st_size
    assert size > 0 and size % (meta["max_seq_len"] * itemsize) == 0
    assert meta["num_sequences"] == size // (meta["max_seq_len"] * itemsize)
    assert meta["total_tokens"] == meta["num_sequences"] * meta["max_seq_len"]
    assert meta["tokenizer"]["class"].endswith("tok.py:TinyTokenizer")

    by_name = {s["name"]: s for s in meta["sources"]}
    assert set(by_name) == {"web", "code"}
    assert by_name["web"]["configured_ratio"] == 0.6
    assert by_name["web"]["docs_total"] == len(WEB_ROWS)
    assert by_name["code"]["docs_total"] == len(CODE_ROWS)
    for s in meta["sources"]:
        assert 0 < s["docs_sampled"] <= s["docs_total"]
        assert s["tokens"] > 0
        for key in ("dataset", "subset", "split", "configured_ratio",
                    "docs_total", "docs_sampled", "tokens"):
            assert key in s

    # 数据量小，必然有源先抽干
    assert len(meta["exhausted_sources"]) >= 1
    assert set(meta["exhausted_sources"]) <= {"web", "code"}

    # tokens 内容合法：全部 < vocab_size
    import numpy as np
    tokens = np.memmap(tokens_path, dtype="uint16", mode="r")
    assert tokens.max() < 100


def test_deterministic_two_runs(mix_setup):
    tmp_path, config = mix_setup
    cache_dir = tmp_path / "cache"
    out1 = run_mix(config, cache_dir, output=tmp_path / "out1", progress=False)
    out2 = run_mix(config, cache_dir, output=tmp_path / "out2", progress=False)
    h1 = hashlib.sha256((out1.output_dir / "tokens.bin").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out2.output_dir / "tokens.bin").read_bytes()).hexdigest()
    assert h1 == h2
    assert read_meta(out1.output_dir) == read_meta(out2.output_dir)


def test_uint32_when_vocab_large(tmp_path):
    tok = write_tokenizer(tmp_path, textwrap.dedent("""
        class BigTokenizer:
            eos_token_id = 9
            vocab_size = 70000

            def encode(self, text):
                return [len(w) for w in text.split()]
    """))
    config = write_config(
        tmp_path, f"{tok}:BigTokenizer",
        [("web", "web-alias", 1.0, WEB_ROWS)], stem="big", seq_len=8)
    result = run_mix(config, tmp_path / "cache", progress=False)
    meta = read_meta(result.output_dir)
    assert meta["dtype"] == "uint32"
    assert meta["vocab_size"] == 70000
    assert meta["max_seq_len"] == 8
    size = (result.output_dir / "tokens.bin").stat().st_size
    assert size % (meta["max_seq_len"] * 4) == 0


def test_output_exists_and_overwrite(mix_setup):
    tmp_path, config = mix_setup
    cache_dir = tmp_path / "cache"
    run_mix(config, cache_dir, progress=False)
    with pytest.raises(OutputExistsError, match="--overwrite|already exists"):
        run_mix(config, cache_dir, progress=False)
    result = run_mix(config, cache_dir, overwrite=True, progress=False)
    assert (result.output_dir / "tokens.bin").is_file()


def test_missing_arrow_cache_errors(tmp_path):
    tok = write_tokenizer(tmp_path)
    path = tmp_path / "mix.toml"
    path.write_text(textwrap.dedent(f"""
        tokenizer = {{ class = '{tok}:TinyTokenizer' }}
        max_seq_len = 8

        [[source]]
        name = "ghost"
        dataset = "ghost-alias"
        ratio = 1.0
    """), encoding="utf-8")
    with pytest.raises(MixCacheNotFoundError, match="convert"):
        run_mix(path, tmp_path / "cache", progress=False)


def test_uint16_overflow_rejected(tmp_path):
    tok = write_tokenizer(tmp_path, textwrap.dedent("""
        class LyingTokenizer:
            eos_token_id = 70000   # 谎报 vocab_size 的小
            vocab_size = 100

            def encode(self, text):
                return [len(w) for w in text.split()]
    """))
    config = write_config(
        tmp_path, f"{tok}:LyingTokenizer",
        [("web", "web-alias", 1.0, WEB_ROWS)], stem="lie")
    with pytest.raises(ValueError, match="uint16"):
        run_mix(config, tmp_path / "cache", progress=False)


def test_run_mix_does_not_print_summary(mix_setup, capsys):
    """总结输出属 CLI 层职责；run_mix 作为库调用不应向 stdout 打印"""
    tmp_path, config = mix_setup
    run_mix(config, tmp_path / "cache", progress=False)
    captured = capsys.readouterr()
    assert "done:" not in captured.out
    assert "configured" not in captured.out


VALID_ROWS = ["q w e r", "a s d f", "z x c v", "y u i o", "t g b n",
              "m j u k"]   # 每篇 5 token（含 EOS）


def write_config_with_valid(tmp_path, tokenizer_class, seq_len=4, seed=42):
    make_arrow_cache(tmp_path / "cache", "web-alias", WEB_ROWS)
    make_arrow_cache(tmp_path / "cache", "valid-alias", VALID_ROWS,
                     split="validation")
    toml = f"""
        tokenizer = {{ class = '{tokenizer_class}' }}
        max_seq_len = {seq_len}
        seed = {seed}

        [[source]]
        name = "web"
        dataset = "web-alias"
        ratio = 1.0

        [[source]]
        name = "heldout"
        dataset = "valid-alias"
        split = "validation"
        ratio = 1.0
        role = "valid"
    """
    path = tmp_path / "mix.toml"
    path.write_text(textwrap.dedent(toml), encoding="utf-8")
    return path


@pytest.fixture
def mix_valid_setup(tmp_path):
    tok = write_tokenizer(tmp_path)
    config = write_config_with_valid(tmp_path, f"{tok}:TinyTokenizer")
    return tmp_path, config


def test_valid_source_packed_separately(mix_valid_setup):
    tmp_path, config = mix_valid_setup
    result = run_mix(config, tmp_path / "cache", progress=False)
    out = result.output_dir

    # valid 源单独打包，不进训练 tokens.bin 的统计
    assert (out / "valid_tokens.bin").is_file()
    meta = read_meta(out)
    assert {s["name"] for s in meta["sources"]} == {"web"}
    valid = meta["valid"]
    assert {s["name"] for s in valid["sources"]} == {"heldout"}
    vs = valid["sources"][0]
    assert vs["split"] == "validation"
    assert vs["docs_total"] == len(VALID_ROWS)
    assert vs["docs_sampled"] == len(VALID_ROWS)   # 单源：抽干为止

    # valid_tokens.bin 大小与 meta 一致，token 合法
    import numpy as np
    itemsize = 2
    vsize = (out / "valid_tokens.bin").stat().st_size
    assert vsize == valid["num_sequences"] * meta["max_seq_len"] * itemsize
    assert valid["total_tokens"] == valid["num_sequences"] * meta["max_seq_len"]
    tokens = np.memmap(out / "valid_tokens.bin", dtype="uint16", mode="r")
    assert tokens.max() < 100

    # 6 篇 × 5 token = 30 token → 7 条 seq（seq_len=4），尾部 2 token 丢弃
    assert valid["num_sequences"] == 7
    assert valid["dropped_tail_tokens"] == 2

    # run_mix 结果里带 valid 统计
    assert result.valid is not None
    assert result.valid["num_sequences"] == 7


def test_no_valid_source_has_no_valid_tokens(mix_setup):
    tmp_path, config = mix_setup
    result = run_mix(config, tmp_path / "cache", progress=False)
    assert not (result.output_dir / "valid_tokens.bin").exists()
    assert "valid" not in read_meta(result.output_dir)
    assert result.valid is None


def test_valid_packing_deterministic(mix_valid_setup):
    tmp_path, config = mix_valid_setup
    out1 = run_mix(config, tmp_path / "cache", output=tmp_path / "o1",
                   progress=False)
    out2 = run_mix(config, tmp_path / "cache", output=tmp_path / "o2",
                   progress=False)
    h1 = hashlib.sha256(
        (out1.output_dir / "valid_tokens.bin").read_bytes()).hexdigest()
    h2 = hashlib.sha256(
        (out2.output_dir / "valid_tokens.bin").read_bytes()).hexdigest()
    assert h1 == h2
    assert read_meta(out1.output_dir) == read_meta(out2.output_dir)


def test_valid_source_does_not_change_train_tokens(tmp_path):
    """有无 valid 源，train 打包结果逐字节一致（rng 流不被 valid 干扰）"""
    tok = write_tokenizer(tmp_path)
    cfg_train_only = write_config(
        tmp_path, f"{tok}:TinyTokenizer",
        [("web", "web-alias", 1.0, WEB_ROWS)], stem="a")
    r1 = run_mix(cfg_train_only, tmp_path / "cache", output=tmp_path / "o1",
                 progress=False)
    cfg_with_valid = write_config_with_valid(
        tmp_path, f"{tok}:TinyTokenizer")
    r2 = run_mix(cfg_with_valid, tmp_path / "cache", output=tmp_path / "o2",
                 progress=False)
    assert (r1.output_dir / "tokens.bin").read_bytes() == \
        (r2.output_dir / "tokens.bin").read_bytes()
