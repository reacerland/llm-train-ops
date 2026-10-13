import numpy as np
import pytest
from datasets import Dataset

from trainops.mix.config import SourceConfig
from trainops.mix.pretokenize import PretokenizeError, iter_docs, pretokenize


class WordLenTokenizer:
    """encode = 按空格切词取词长；eos = 9。"""
    eos_token_id = 9
    vocab_size = 100

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def make_arrow(tmp_path, rows, name="arrow"):
    arrow_dir = tmp_path / name
    Dataset.from_dict({"text": rows}).save_to_disk(str(arrow_dir))
    return arrow_dir


def src(name="web", field="text", ratio=1.0):
    return SourceConfig(name=name, dataset="ds", field=field, ratio=ratio)


def test_tokens_and_offsets_written(tmp_path):
    arrow = make_arrow(tmp_path, ["a bb", "ccc", "d e f"])
    pt = pretokenize(src(), WordLenTokenizer(), arrow, tmp_path)
    tokens = np.fromfile(pt.tokens_path, dtype=np.uint32)
    assert tokens.tolist() == [1, 2, 9, 3, 9, 1, 1, 1, 9]  # encode + EOS 每篇
    offsets = np.fromfile(pt.offsets_path, dtype=np.uint64)
    assert offsets.tolist() == [0, 3, 5, 9]
    assert pt.docs_total == 3
    assert pt.tokens_total == 9
    assert pt.skipped_empty == 0
    docs = [d.tolist() for d in iter_docs(pt)]
    assert docs == [[1, 2, 9], [3, 9], [1, 1, 1, 9]]


def test_empty_doc_skipped(tmp_path):
    arrow = make_arrow(tmp_path, ["", "   ", "a"])
    pt = pretokenize(src(), WordLenTokenizer(), arrow, tmp_path)
    assert pt.docs_total == 1
    assert pt.skipped_empty == 2
    assert [d.tolist() for d in iter_docs(pt)] == [[1, 9]]


def test_null_doc_treated_as_empty(tmp_path):
    arrow = make_arrow(tmp_path, ["a", None])
    pt = pretokenize(src(), WordLenTokenizer(), arrow, tmp_path)
    assert pt.docs_total == 1
    assert pt.skipped_empty == 1


def test_missing_field_names_available_columns(tmp_path):
    arrow = make_arrow(tmp_path, ["a"])
    with pytest.raises(PretokenizeError) as exc_info:
        pretokenize(src(field="absent"), WordLenTokenizer(), arrow, tmp_path)
    assert "absent" in str(exc_info.value)
    assert "text" in str(exc_info.value)  # 可用列名


def test_encode_failure_names_source_and_row(tmp_path):
    class BadTokenizer(WordLenTokenizer):
        def encode(self, text):
            if text == "bad":
                raise ValueError("boom")
            return super().encode(text)

    arrow = make_arrow(tmp_path, ["ok", "bad"])
    with pytest.raises(PretokenizeError) as exc_info:
        pretokenize(src(name="web"), BadTokenizer(), arrow, tmp_path)
    message = str(exc_info.value)
    assert "web" in message
    assert "row 2" in message or "2" in message


def test_all_empty_docs_is_error(tmp_path):
    arrow = make_arrow(tmp_path, ["", "  "])
    with pytest.raises(PretokenizeError, match="web"):
        pretokenize(src(), WordLenTokenizer(), arrow, tmp_path)


def test_pretokenize_streams_in_batches(tmp_path, monkeypatch):
    """GB 级源不得整列载入内存：必须分批迭代（fetch/convert 的内存恒定原则）"""
    rows = ["a bb", "ccc", "d e f"]

    class FakeDataset:
        column_names = ["text"]

        def iter(self, batch_size=1000):
            assert batch_size >= 1
            for i in range(0, len(rows), batch_size):
                yield {"text": rows[i:i + batch_size]}

        def __getitem__(self, key):
            raise AssertionError(
                "pretokenize must stream batches, not load the whole column")

    monkeypatch.setattr("datasets.load_from_disk", lambda p: FakeDataset())
    pt = pretokenize(src(), WordLenTokenizer(), tmp_path / "fake-arrow",
                     tmp_path)
    assert pt.docs_total == 3
    assert pt.tokens_total == 9
