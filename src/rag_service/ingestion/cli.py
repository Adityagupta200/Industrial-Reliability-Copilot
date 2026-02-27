from __future__ import annotations

import json
import typer
from rag_service.ingestion.pipeline import ingest_all

app = typer.Typer(no_args_is_help=True)


@app.command()
def run() -> None:
    stats = ingest_all()
    typer.echo(json.dumps(stats, indent=2))


if __name__ == "__main__":
    app()
