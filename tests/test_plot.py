import json

import pytest

from trainops.train.plot import (
    PlotError,
    load_eval_series,
    parse_y_metrics,
    run_plot,
)


def write_eval_jsonl(tmp_path, entries=None):
    if entries is None:
        entries = []
        for step in (100, 200, 300):
            for split, base in (("valid", 3.0), ("train", 3.2)):
                entries.append({
                    "step": step, "split": split,
                    "loss": base - step * 0.005,
                    "ppl": 30.0 - step * 0.08,
                    "top1": 0.3 + step * 0.001,
                    "top5": 0.6 + step * 0.001,
                    "tokens": 1000,
                })
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                    encoding="utf-8")
    return path


def test_load_eval_series_groups_and_sorts(tmp_path):
    path = write_eval_jsonl(tmp_path)
    series = load_eval_series(path)
    assert set(series) == {"valid", "train"}
    steps = [e["step"] for e in series["valid"]]
    assert steps == [100, 200, 300]


def test_load_eval_series_missing_file(tmp_path):
    with pytest.raises(PlotError, match="not found"):
        load_eval_series(tmp_path / "nope.jsonl")


def test_load_eval_series_empty(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(PlotError, match="no entries"):
        load_eval_series(path)


def test_load_eval_series_bad_line(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"step": 1, "split": "valid"}\nnot json\n',
                    encoding="utf-8")
    with pytest.raises(PlotError, match="line 2"):
        load_eval_series(path)


def test_parse_y_metrics():
    assert parse_y_metrics("ppl") == ["ppl"]
    assert parse_y_metrics("loss, top1 ,top5") == ["loss", "top1", "top5"]
    with pytest.raises(PlotError, match="empty"):
        parse_y_metrics(" , ")


def test_run_plot_single_metric(tmp_path):
    path = write_eval_jsonl(tmp_path)
    out = tmp_path / "curves.png"
    result = run_plot(path, x="step", ys=["ppl"], output=out)
    assert result == out
    assert out.is_file() and out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_run_plot_multi_metric_subplots(tmp_path):
    path = write_eval_jsonl(tmp_path)
    out = tmp_path / "curves.png"
    run_plot(path, x="step", ys=["loss", "ppl", "top1", "top5"], output=out)
    assert out.is_file() and out.stat().st_size > 0


def test_run_plot_default_output_next_to_input(tmp_path):
    path = write_eval_jsonl(tmp_path)
    result = run_plot(path, x="step", ys=["ppl"])
    assert result == tmp_path / "eval_curves.png"
    assert result.is_file()


def test_run_plot_unknown_x_field(tmp_path):
    path = write_eval_jsonl(tmp_path)
    with pytest.raises(PlotError, match="flops"):
        run_plot(path, x="flops", ys=["ppl"], output=tmp_path / "o.png")


def test_run_plot_unknown_y_field(tmp_path):
    path = write_eval_jsonl(tmp_path)
    with pytest.raises(PlotError, match="accuracy"):
        run_plot(path, x="step", ys=["accuracy"], output=tmp_path / "o.png")
