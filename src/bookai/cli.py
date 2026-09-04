from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from .harness import TranslationHarness
from .pipeline import MODE_ALIASES, translate_book

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Translate FB2/EPUB/DOCX/TXT with a selective multi-model literary harness.")


def _print_progress(event: dict) -> None:
    """Emit machine-readable progress so CI does not look hung."""
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


@app.command()
def translate(
    source: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
    mode: str = typer.Option("optimal", help="fast | optimal | literary"),
):
    if mode not in {"fast", "optimal", "literary", "standard", "high"}:
        raise typer.BadParameter("mode must be fast, optimal, or literary")
    mode = MODE_ALIASES.get(mode, mode)
    output = output or source.with_name(f"{source.stem}.ru{source.suffix}")
    harness = TranslationHarness.from_env()
    result = translate_book(source, output, harness, mode=mode, progress=_print_progress)
    typer.echo(f"Done: {result}")
    typer.echo(f"Usage: {harness.usage}")


if __name__ == "__main__":
    app()
