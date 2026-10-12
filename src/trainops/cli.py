import logging
import os
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
