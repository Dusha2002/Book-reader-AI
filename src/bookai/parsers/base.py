from __future__ import annotations

from pathlib import Path

from ..models import BookDocument


class UnsupportedBookFormat(ValueError):
    pass


SUPPORTED_SUFFIXES = {".fb2", ".epub", ".txt", ".docx"}


def load_book(path: Path) -> BookDocument:
    suffix = path.suffix.lower()
    if suffix == ".fb2":
        from .fb2 import load_fb2
        return load_fb2(path)
    if suffix == ".epub":
        from .epub import load_epub
        return load_epub(path)
    if suffix == ".txt":
        from .text import load_txt
        return load_txt(path)
    if suffix == ".docx":
        from .docx import load_docx
        return load_docx(path)
    raise UnsupportedBookFormat(f"Unsupported format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")


def save_book(document: BookDocument, translations: dict[str, str], output: Path) -> None:
    if document.format == "fb2":
        from .fb2 import save_fb2
        save_fb2(document, translations, output)
        return
    if document.format == "epub":
        from .epub import save_epub
        save_epub(document, translations, output)
        return
    if document.format == "txt":
        from .text import save_txt
        save_txt(document, translations, output)
        return
    if document.format == "docx":
        from .docx import save_docx
        save_docx(document, translations, output)
        return
    raise UnsupportedBookFormat(document.format)
