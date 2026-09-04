from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from ..models import BookDocument, Segment


@dataclass
class EpubPayload:
    files: dict[str, bytes]
    text_files: list[str]
    segment_map: dict[str, tuple[str, int]]


def _visible_nodes(soup: BeautifulSoup) -> list[NavigableString]:
    out: list[NavigableString] = []
    for node in soup.find_all(string=True):
        if isinstance(node, NavigableString) and node.parent is not None and node.parent.name not in {"script", "style"} and str(node).strip():
            out.append(node)
    return out


def load_epub(path: Path) -> BookDocument:
    with zipfile.ZipFile(path, "r") as zf:
        files = {name: zf.read(name) for name in zf.namelist()}
    text_files = [n for n in files if n.lower().endswith((".xhtml", ".html", ".htm"))]
    segments: list[Segment] = []
    segment_map: dict[str, tuple[str, int]] = {}
    idx = 0
    for name in text_files:
        soup = BeautifulSoup(files[name], "lxml-xml")
        visible = _visible_nodes(soup)
        for pos, node in enumerate(visible):
            text = str(node).strip()
            sid = f"s{idx:06d}"
            segments.append(Segment(id=sid, text=text, locator=f"{name}#{pos}"))
            segment_map[sid] = (name, pos)
            idx += 1
    return BookDocument(path, "epub", segments, EpubPayload(files, text_files, segment_map))


def save_epub(document: BookDocument, translations: dict[str, str], output: Path) -> None:
    payload = document.payload
    assert isinstance(payload, EpubPayload)
    files = dict(payload.files)
    grouped: dict[str, list[tuple[int, str]]] = {}
    for sid, translated in translations.items():
        location = payload.segment_map.get(sid)
        if location:
            grouped.setdefault(location[0], []).append((location[1], translated))

    for name, replacements in grouped.items():
        soup = BeautifulSoup(files[name], "lxml-xml")
        nodes = _visible_nodes(soup)
        for pos, translated in replacements:
            if pos < len(nodes):
                original = str(nodes[pos])
                left_ws = original[: len(original) - len(original.lstrip())]
                right_ws = original[len(original.rstrip()) :]
                nodes[pos].replace_with(left_ws + translated + right_ws)
        files[name] = str(soup).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as zf:
        if "mimetype" in files:
            zf.writestr("mimetype", files.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, data in files.items():
            zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
