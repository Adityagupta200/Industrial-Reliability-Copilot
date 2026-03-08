from __future__ import annotations

import json
import typer

from rag_service.ingestion.pipeline import ingest_all

app = typer.Typer(no_args_is_help=True)


@app.command()
def run() -> None:
    """Run the full ingestion pipeline (manuals + procedures)."""
    stats = ingest_all()
    typer.echo(json.dumps(stats, indent=2))


@app.command("show-config")
def show_config() -> None:
    """Print minimal config hints (sanity check that CLI is wired)."""
    typer.echo("CLI is wired correctly. If you see this, commands are registered.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
