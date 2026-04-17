"""
ETL CLI

Usage:
    uv run etl ashaar
    uv run etl all
    uv run etl all --dry-run
"""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="الشاعر ETL pipelines")
console = Console()


@app.command()
def run(
    source: str = typer.Argument(help="Pipeline: ashaar | all"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from etl.dedup import dedup
    from etl.load import load

    streams = []

    if source in ("ashaar", "all"):
        from etl.pipelines.ashaar import extract
        console.rule("[bold cyan]ashaar")
        streams.append(extract())

    if not streams:
        console.print(f"[red]Unknown source: {source!r}[/]")
        raise typer.Exit(1)

    console.rule("[bold]Deduplicating")
    deduplicated = dedup(streams)

    console.rule("[bold]Loading → Postgres")
    counts = load(deduplicated, dry_run=dry_run)

    console.rule("[bold green]Done")
    for k, v in counts.items():
        console.print(f"  {k}: {v:,}")


if __name__ == "__main__":
    app()
