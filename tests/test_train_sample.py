import sys
import textwrap
from pathlib import Path

import pytest
import torch

from trainops.mix.tokenizer import TokenizerLoadError, load_tokenizer
from trainops.mix.config import TokenizerClassRef
from trainops.train.config import PromptsConfig
from trainops.train.sample import generate_samples, require_decode

FIXTURES_DIR = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))
from gpt import GPT  # noqa: E402

CHAR_TOKENIZER = '''class CharTok:
    eos_token_id = 0
    vocab_size = 256
    def encode(self, text):
        return [ord(c) % 255 + 1 for c in text]
    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids if i > 0)
'''


def write_char_tokenizer(tmp_path):
    path = tmp_path / "char_tok.py"
    path.write_text(CHAR_TOKENIZER, encoding="utf-8")
    return load_tokenizer(TokenizerClassRef(path=str(path),
                                            class_name="CharTok"))


def make_model(vocab=256, **overrides):
    args = dict(vocab_size=vocab, d_model=32, n_layer=1, n_head=2,
                max_seq_len=16)
    return GPT(**(args | overrides))


def test_generate_samples(tmp_path):
    tok = write_char_tokenizer(tmp_path)
    model = make_model()
    out = tmp_path / "samples.txt"
    generate_samples(model, tok, PromptsConfig(texts=["ab"], max_new_tokens=5),
                     torch.device("cpu"), out)
    content = out.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == ">>> ab"
    assert len(lines) == 2  # prompt line + generated line


def test_require_decode_missing():
    class NoDecode:
        eos_token_id = 0
        vocab_size = 10

        def encode(self, text):
            return []

    with pytest.raises(TokenizerLoadError, match="decode"):
        require_decode(NoDecode())


def test_require_decode_ok(tmp_path):
    require_decode(write_char_tokenizer(tmp_path))  # no raise


class _EosModel(torch.nn.Module):
    """Always argues for eos_token_id=0 as the next token."""

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 256)
        logits[..., 0] = 1.0
        return logits


def test_eos_stop(tmp_path):
    tok = write_char_tokenizer(tmp_path)
    out = tmp_path / "samples.txt"
    generate_samples(_EosModel(), tok,
                     PromptsConfig(texts=["ab"], max_new_tokens=50),
                     torch.device("cpu"), out)
    content = out.read_text(encoding="utf-8")
    assert content == ">>> ab\n\n"  # empty generation, stopped at once


def test_prompt_truncated_to_max_seq_len(tmp_path):
    tok = write_char_tokenizer(tmp_path)
    model = make_model()  # max_seq_len=16
    out = tmp_path / "samples.txt"
    long_prompt = "x" * 100  # 100 tokens > max_seq_len
    generate_samples(model, tok, PromptsConfig(texts=[long_prompt],
                                               max_new_tokens=3),
                     torch.device("cpu"), out)
    assert out.read_text(encoding="utf-8").startswith(">>> " + "x" * 100)
