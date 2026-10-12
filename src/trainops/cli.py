import typer

app = typer.Typer(help="LLM train CI/CD toolkit")


@app.command()
def version() -> None:
    """Show version."""
    typer.echo("trainops 0.1.0")
