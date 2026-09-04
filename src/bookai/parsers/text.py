from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import BookDocument, Segment


@dataclass
class TextPayload:
    blocks: list[str]
    segment_index: dict[str, int]
    encoding: str = "utf-8"


def _decode(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace"), "utf-8"


def load_txt(path: Path) -> BookDocument:
    text, encoding = _decode(path.read_bytes())
    blocks = text.splitlines(keepends=True)
    segments: list[Segment] = []
    segment_index: dict[str, int] = {}
    chapter = path.stem
    idx = 0
    for pos, block in enumerate(blocks):
        stripped = block.strip()
        if not stripped:
            continue
        if len(stripped) <= 120 and (stripped.lower().startswith("chapter ") or stripped.lower().startswith("глава ")):
            chapter = stripped
        sid = f"s{idx:06d}"
        segments.append(Segment(sid, stripped, f"line:{pos}", chapter=chapter))
        segment_index[sid] = pos
        idx += 1
    return BookDocument(path, "txt", segments, TextPayload(blocks, segment_index, encoding))


def save_txt(document: BookDocument, translations: dict[str, str], output: Path) -> None:
    payload = document.payload
    assert isinstance(payload, TextPayload)
    blocks = list(payload.blocks)
    for sid, translated in translations.items():
        pos = payload.segment_index.get(sid)
        if pos is None:
            continue
        original = blocks[pos]
        newline = "\r\n" if original.endswith("\r\n") else ("\n" if original.endswith("\n") else "")
        left = original[: len(original) - len(original.lstrip())]
        blocks[pos] = left + translated + newline
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(blocks), encoding="utf-8")
