"""Plot training/validation curves from a train run's eval.jsonl.

Reads the per-eval-point entries written by the train loop (fields: step,
split, loss, ppl, top1, top5, tokens) and renders one figure: chosen y
metrics as vertically stacked subplots sharing the x axis, train and valid
splits overlaid on the same axes — the overfitting view.

matplotlib is imported lazily inside run_plot so that parsing helpers and
the rest of the CLI stay importable without it.
"""

import json
from pathlib import Path

DEFAULT_OUTPUT_NAME = "eval_curves.png"


class PlotError(Exception):
    """Raised when the eval.jsonl input is missing or malformed."""


def parse_y_metrics(spec: str) -> list[str]:
    """Parse a comma-separated y-metric list like 'loss,ppl'."""
    metrics = [m.strip() for m in spec.split(",") if m.strip()]
    if not metrics:
        raise PlotError(f"y metrics list is empty, got {spec!r}")
    return metrics


def load_eval_series(path: Path) -> dict[str, list[dict]]:
    """Load eval.jsonl into {split: [entries sorted by step]}."""
    path = Path(path)
    if not path.is_file():
        raise PlotError(f"eval file not found: {path}")
    series: dict[str, list[dict]] = {}
    for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlotError(f"{path}: invalid JSON on line {lineno}: "
                            f"{exc}") from exc
        split = entry.get("split")
        if not isinstance(split, str):
            raise PlotError(f"{path}: line {lineno} missing 'split' field")
        series.setdefault(split, []).append(entry)
    if not series:
        raise PlotError(f"{path}: no entries")
    for entries in series.values():
        entries.sort(key=lambda e: e.get("step", 0))
    return series


def run_plot(input: Path, x: str, ys: list[str],
             output: Path | None = None) -> Path:
    """Render curves figure; returns the output PNG path."""
    import matplotlib
    matplotlib.use("Agg")   # headless: CI has no display
    import matplotlib.pyplot as plt

    input = Path(input)
    series = load_eval_series(input)
    output = Path(output) if output else input.parent / DEFAULT_OUTPUT_NAME

    sample = next(iter(series.values()))[0]
    available = sorted(sample.keys())
    for field in [x, *ys]:
        if field not in available:
            raise PlotError(f"unknown field {field!r} in {input}; available "
                            f"fields: {', '.join(available)}")

    fig, axes = plt.subplots(len(ys), 1, figsize=(9, 3.2 * len(ys)),
                             sharex=True, squeeze=False)
    for ax, y in zip(axes.ravel(), ys):
        for split, entries in sorted(series.items()):
            xs = [e[x] for e in entries if x in e and y in e]
            values = [e[y] for e in entries if x in e and y in e]
            ax.plot(xs, values, marker="o", markersize=3, linewidth=1.2,
                    label=split)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
        ax.legend()
    axes.ravel()[-1].set_xlabel(x)
    fig.suptitle(f"{input.name}: {', '.join(ys)} vs {x}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
