import sys
import textwrap

import pytest

from trainops.mix.config import TokenizerClassRef
from trainops.mix.tokenizer import (
    PRESETS,
    TokenizerLoadError,
    _HFTokenizerWrapper,
    load_tokenizer,
    load_tokenizer_class,
    register_preset,
)


class FakeTokenizer:
    eos_token_id = 9
    vocab_size = 100

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


@pytest.fixture
def fake_tokenizer_file(tmp_path):
    path = tmp_path / "my_bpe.py"
    path.write_text(textwrap.dedent("""
        class MyTokenizer:
            eos_token_id = 9
            vocab_size = 100

            def encode(self, text: str) -> list[int]:
                return [len(w) for w in text.split()]
    """), encoding="utf-8")
    return path


def test_load_preset_by_name():
    sentinel = FakeTokenizer()
    register_preset("fake-test", lambda: sentinel)
    try:
        assert load_tokenizer("fake-test") is sentinel
    finally:
        del PRESETS["fake-test"]


def test_unknown_preset_lists_available():
    with pytest.raises(TokenizerLoadError) as exc_info:
        load_tokenizer("no-such-preset")
    assert "no-such-preset" in str(exc_info.value)
    assert "llama2" in str(exc_info.value)  # 列出可用预置


def test_load_class_from_file(fake_tokenizer_file):
    tok = load_tokenizer(TokenizerClassRef(path=str(fake_tokenizer_file),
                                           class_name="MyTokenizer"))
    assert tok.eos_token_id == 9
    assert tok.vocab_size == 100
    assert tok.encode("a bb ccc") == [1, 2, 3]


def test_class_missing_attr_names_it(fake_tokenizer_file):
    path = fake_tokenizer_file
    path.write_text(textwrap.dedent("""
        class MyTokenizer:
            eos_token_id = 9

            def encode(self, text):
                return [1]
    """), encoding="utf-8")
    with pytest.raises(TokenizerLoadError, match="vocab_size"):
        load_tokenizer(TokenizerClassRef(path=str(path), class_name="MyTokenizer"))


def test_class_not_found_in_file(fake_tokenizer_file):
    with pytest.raises(TokenizerLoadError, match="Nope"):
        load_tokenizer(TokenizerClassRef(path=str(fake_tokenizer_file),
                                         class_name="Nope"))


def test_llama2_preset_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import 必失败
    with pytest.raises(TokenizerLoadError, match=r"trainops\[hf\]"):
        load_tokenizer("llama2")


def test_hf_wrapper_no_special_tokens():
    calls = []

    class FakeAuto:
        eos_token_id = 2
        vocab_size = 32000

        def encode(self, text, **kwargs):
            calls.append(kwargs)
            return [7, 8]

    wrapper = _HFTokenizerWrapper(FakeAuto())
    assert wrapper.encode("hi") == [7, 8]
    assert wrapper.eos_token_id == 2
    assert wrapper.vocab_size == 32000
    assert calls == [{"add_special_tokens": False}]


def test_load_tokenizer_class_module_missing_file(tmp_path):
    with pytest.raises(TokenizerLoadError, match="not found"):
        load_tokenizer_class(TokenizerClassRef(path=str(tmp_path / "nope.py"),
                                               class_name="X"))
