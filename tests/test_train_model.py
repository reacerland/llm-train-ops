import textwrap
from pathlib import Path

import pytest
import torch

from trainops.train.config import ModelRef
from trainops.train.model import ModelLoadError, load_model

FIXTURES_DIR = Path(__file__).parent / "fixtures"

TINY_ARGS = {"vocab_size": 50, "d_model": 32, "n_layer": 2, "n_head": 2,
             "max_seq_len": 16}


def test_load_model_ok():
    model = load_model(ModelRef(file="gpt.py", class_name="GPT",
                                args=dict(TINY_ARGS)),
                       config_dir=FIXTURES_DIR)
    assert isinstance(model, torch.nn.Module)
    logits = model(torch.randint(0, 50, (2, 16)))
    assert logits.shape == (2, 16, 50)


def test_load_model_file_missing(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(ModelRef(file="nope.py", class_name="GPT", args={}),
                   config_dir=tmp_path)


def test_load_model_class_missing():
    with pytest.raises(ModelLoadError, match="NoSuchClass"):
        load_model(ModelRef(file="gpt.py", class_name="NoSuchClass", args={}),
                   config_dir=FIXTURES_DIR)


def test_load_model_not_a_module(tmp_path):
    (tmp_path / "plain.py").write_text(textwrap.dedent("""\
        class Plain:
            def __init__(self, **kwargs):
                pass
    """), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="nn.Module"):
        load_model(ModelRef(file="plain.py", class_name="Plain", args={}),
                   config_dir=tmp_path)


def test_load_model_bad_args():
    with pytest.raises(ModelLoadError, match="instantiate"):
        load_model(ModelRef(file="gpt.py", class_name="GPT", args={}),
                   config_dir=FIXTURES_DIR)


def test_load_model_import_error(tmp_path):
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n",
                                        encoding="utf-8")
    with pytest.raises(ModelLoadError, match="import"):
        load_model(ModelRef(file="broken.py", class_name="GPT", args={}),
                   config_dir=tmp_path)


def test_model_file_resolved_relative_to_config_dir():
    # relative file name resolves against config_dir, not cwd
    model = load_model(ModelRef(file="gpt.py", class_name="GPT",
                                args=dict(TINY_ARGS)),
                       config_dir=FIXTURES_DIR)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.slow
def test_gpt_fixture_30m_params():
    import sys
    sys.path.insert(0, str(FIXTURES_DIR))
    from gpt import GPT

    model = GPT(vocab_size=32000, d_model=320, n_layer=8, n_head=8,
                max_seq_len=4096)
    n = sum(p.numel() for p in model.parameters())
    assert 28_000_000 <= n <= 35_000_000
    # one forward+backward to prove trainability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ids = torch.randint(0, 32000, (2, 128), device=device)
    loss = model(ids).square().mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
