import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import typer

from trainops.convert.command import (
    CacheNotFoundError,
    OutputExistsError,
    run_convert,
)
from trainops.convert.transform import MissingFieldError, parse_fields
from trainops.fetch.cache import CacheCorruptedError
from trainops.fetch.command import run_fetch
from trainops.fetch.sizeparse import parse_size
from trainops.mix.command import MixCacheNotFoundError, run_mix
from trainops.mix.config import ConfigError
from trainops.mix.pretokenize import PretokenizeError
from trainops.mix.tokenizer import TokenizerLoadError
from trainops.train.config import load_task_config
from trainops.train.data import MixtureNotFoundError
from trainops.train.loop import run_training
from trainops.train.model import ModelLoadError
from trainops.train.plot import PlotError, parse_y_metrics, run_plot

app = typer.Typer(help="LLM train CI/CD toolkit")


@app.callback()
def main() -> None:
    """LLM train CI/CD toolkit."""


@app.command()
def fetch(
    dataset: str = typer.Option(..., help="ModelScope dataset id, e.g. 'AI-ModelScope/xxx'"),
    name: str | None = typer.Option(None, help="Cache alias for this dataset (used as the cache key)"),
    subset: str | None = typer.Option(None, help="Dataset subset name (omit for datasets without configs)"),
    split: str = typer.Option("train", help="Dataset split"),
    max_size: str = typer.Option(..., "--max-size", help="Target cache size, e.g. '500MB', '2GB'"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Fetch a ModelScope dataset into the local cache (streaming, size-capped)."""
    logging.getLogger("modelscope").setLevel(logging.WARNING)  # keep progress readable
    try:
        size = parse_size(max_size)
    except ValueError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        result = run_fetch(dataset, subset, split, size, cache_dir, name=name)
    except CacheCorruptedError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(
            f"error: failed to fetch dataset '{dataset}' (subset={subset}, split={split}): {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from exc

    if not result.downloaded:
        typer.echo(f"cache hit: {result.bytes} bytes, {result.records} records already at {result.cache_dir}")
    else:
        typer.echo(f"done: {result.bytes} bytes, {result.records} records cached at {result.cache_dir}")
        if result.dataset_exhausted:
            typer.secho("warning: dataset exhausted before reaching target size",
                        fg=typer.colors.YELLOW)


@app.command()
def convert(
    dataset: str = typer.Option(..., help="ModelScope dataset id (must already be cached by fetch)"),
    name: str | None = typer.Option(None, help="Cache alias used at fetch time"),
    subset: str | None = typer.Option(None, help="Dataset subset name"),
    split: str = typer.Option("train", help="Dataset split"),
    fields: str = typer.Option(..., help="Comma-separated field whitelist, e.g. 'text' or 'text,title'"),
    skip_bad: bool = typer.Option(False, "--skip-bad",
                                  help="Skip records missing fields or with invalid JSON lines"),
    output: Path | None = typer.Option(None, help="Output directory (default: <cache>/.../data.arrow)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output directory"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Convert a cached ModelScope dataset into an HF arrow dataset directory."""
    try:
        field_list = parse_fields(fields)
    except ValueError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        result = run_convert(dataset, subset, split, field_list, cache_dir,
                             output=output, skip_bad=skip_bad, overwrite=overwrite,
                             name=name)
    except CacheNotFoundError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except OutputExistsError as exc:
        typer.secho(f"error: {exc} (use --overwrite to replace it)",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except (MissingFieldError, ValueError) as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"error: failed to convert dataset '{dataset}': {exc}",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"done: {result.records} records -> {result.output_dir} "
               f"({result.bytes} bytes)")
    if result.skipped:
        typer.secho(f"warning: skipped {result.skipped} bad records",
                    fg=typer.colors.YELLOW)


@app.command()
def mix(
    config: Path = typer.Option(..., "--config", help="Path to the mix TOML config file"),
    output: Path | None = typer.Option(None, help="Output directory (default: <cache>/mixtures/<config-stem>)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output directory"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Mix multiple converted datasets into a tokenized, packed training set."""
    try:
        result = run_mix(config, cache_dir, output=output, overwrite=overwrite)
    except ConfigError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except TokenizerLoadError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except MixCacheNotFoundError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except OutputExistsError as exc:
        typer.secho(f"error: {exc} (use --overwrite to replace it)",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except (PretokenizeError, ValueError) as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"error: failed to mix datasets from {config}: {exc}",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"done: {result.num_sequences} sequences, "
               f"{result.total_tokens} tokens -> {result.output_dir} "
               f"({result.bytes} bytes)")
    total = result.total_tokens or 1
    for s in result.sources_meta:
        typer.echo(f"  source {s['name']}: configured "
                   f"{s['configured_ratio']:.2%}, actual "
                   f"{s['tokens'] / total:.2%} "
                   f"({s['docs_sampled']}/{s['docs_total']} docs)")
    if result.valid is not None:
        typer.echo(f"  valid holdout: {result.valid['num_sequences']} sequences, "
                   f"{result.valid['total_tokens']} tokens -> valid_tokens.bin")
        for s in result.valid["sources"]:
            typer.echo(f"    source {s['name']}: {s['docs_sampled']}/"
                       f"{s['docs_total']} docs, {s['tokens']} tokens")
    if result.dropped_tail_tokens:
        typer.secho(f"warning: dropped incomplete tail of "
                    f"{result.dropped_tail_tokens} tokens",
                    fg=typer.colors.YELLOW)
    for name in result.exhausted_sources:
        typer.secho(f"warning: source {name!r} exhausted; packing stopped "
                    f"(output is usable)", fg=typer.colors.YELLOW)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", help="Path to the task TOML config file"),
    output: Path = typer.Option(..., "--output", help="Task output directory (CI artifact dir)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint.pt in the output directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output directory"),
    cache_dir: Path = typer.Option(Path.home() / ".cache" / "trainops", help="Cache root directory"),
) -> None:
    """Train a model from a task config on a mix-produced dataset."""
    if resume and overwrite:
        typer.secho("error: --resume and --overwrite are mutually exclusive",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    try:
        task = load_task_config(config)
    except ConfigError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    mixture_dir = task.data_output or \
        cache_dir / "mixtures" / task.path.stem
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        elif not resume:
            typer.secho(
                f"error: output directory already exists: {output} "
                f"(use --overwrite to replace it, or --resume to continue)",
                err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    try:
        result = run_training(task, mixture_dir, output, resume=resume)
    except MixtureNotFoundError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.secho(f"error: {exc} (cannot --resume without it)",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except (ModelLoadError, TokenizerLoadError, ValueError) as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"error: training from {config} failed: {exc}",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"done: {result.final_step} steps, "
               f"final loss {result.final_loss:.4f} -> {result.output_dir}")
    if result.final_eval is not None:
        e = result.final_eval
        typer.echo(f"final valid: ppl={e['ppl']:.2f} loss={e['loss']:.4f} "
                   f"top1={e['top1']:.3f} top5={e['top5']:.3f} "
                   f"({e['tokens']} tokens)")


@app.command()
def plot(
    input: Path = typer.Option(..., "--input",
                               help="Path to an eval.jsonl produced by train"),
    x: str = typer.Option("step", help="Field on the x axis, e.g. 'step'"),
    y: str = typer.Option("ppl", help="Comma-separated y metrics, "
                          "e.g. 'ppl' or 'loss,ppl,top1,top5'"),
    output: Path | None = typer.Option(
        None, help="Output PNG path (default: eval_curves.png next to input)"),
) -> None:
    """Plot train/valid curves from eval.jsonl (overfitting view)."""
    try:
        metrics = parse_y_metrics(y)
        result = run_plot(input, x, metrics, output)
    except PlotError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"error: failed to plot {input}: {exc}",
                    err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"done: {', '.join(metrics)} vs {x} -> {result}")


def main() -> None:
    """Console entry point.

    The ModelScope/HF stack leaves background HTTP state that can hang
    interpreter shutdown (atexit joins / connection cleanup). All our output
    is flushed and all checkpoints saved by the time we get here, so we skip
    interpreter finalization entirely and exit with the original exit code.
    """
    code = 0
    # The datasets/pyarrow resource-tracker child would otherwise warn about a
    # 'leaked semaphore' when we force-exit; silence it for spawned children.
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    try:
        app()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
