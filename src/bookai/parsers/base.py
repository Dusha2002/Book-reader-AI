from __future__ import annotations

from pathlib import Path

from ..models import BookDocument


class UnsupportedBookFormat(ValueError):
    pass


def load_book(path: Path) -> BookDocument:
    suffix = path.suffix.lower()
    if suffix == ".fb2":
        from .fb2 import load_fb2
        return load_fb2(path)
    if suffix == ".epub":
        from .epub import load_epub
        return load_epub(path)
    raise UnsupportedBookFormat(f"Unsupported format: {suffix}. MVP supports .fb2 and .epub")


def save_book(document: BookDocument, translations: dict[str, str], output: Path) -> None:
    if document.format == "fb2":
        from .fb2 import save_fb2
        save_fb2(document, translations, output)
        return
    if document.format == "epub":
        from .epub import save_epub
        save_epub(document, translations, output)
        return
    raise UnsupportedBookFormat(document.format)
