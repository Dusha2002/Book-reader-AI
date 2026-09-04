from __future__ import annotations

from pathlib import Path

import typer

from .llm import OpenAICompatibleProvider
from .pipeline import translate_book

app = typer.Typer(no_args_is_help=True, help="Translate FB2/EPUB books with context-aware literary editing.")


@app.command()
def translate(
    source: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
    mode: str = typer.Option("high", help="fast = translation, standard = + literary edit, high = + bilingual QA"),
):
    if mode not in {"fast", "standard", "high"}:
        raise typer.BadParameter("mode must be fast, standard, or high")
    output = output or source.with_name(f"{source.stem}.ru{source.suffix}")
    provider = OpenAICompatibleProvider()
    result = translate_book(source, output, provider, mode=mode)
    typer.echo(f"Done: {result}")


if __name__ == "__main__":
    app()
