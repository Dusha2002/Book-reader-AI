from __future__ import annotations

import html as html_lib
import io
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from pypdf import PdfReader

# Public mirror of the same MIT Press edition used for the cross-book diagnostic.
# We still resolve physical PDF pages 35–40, but the smoke test keeps only a small
# representative subset so model-backed iteration stays fast.
SOURCE_URL = "https://raw.githubusercontent.com/janishar/mit-deep-learning-book-pdf/master/complete-book-bookmarked-pdf/deeplearningbook.pdf"
PAGE_FIRST = 35
PAGE_LAST = 40
TARGET_MIN_CHARS = 2200
TARGET_MAX_CHARS = 4800
ANCHORS = ("deep learning", "neural", "LSTM", "CIFAR")

_LIGATURES = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "–": "-", "—": "—", "’": "'", "“": '“', "”": '”',
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
})


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_LIGATURES).replace("\u00ad", "")
    value = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _download_pdf() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Book-reader-AI cross-book benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"Benchmark source is not a PDF: {data[:32]!r}")
    return data


def _page_text(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if len(reader.pages) < PAGE_LAST:
        raise RuntimeError(f"Benchmark PDF has only {len(reader.pages)} pages")
    pages: list[str] = []
    for physical_page in range(PAGE_FIRST, PAGE_LAST + 1):
        raw = reader.pages[physical_page - 1].extract_text() or ""
        text = _norm(raw)
        text = re.sub(r"(?im)^\s*CHAPTER\s+1\.\s+INTRODUCTION\s*$", "", text)
        text = re.sub(r"(?im)^\s*\d{1,2}\s*$", "", text)
        text = _norm(text)
        if len(text) < 200:
            raise RuntimeError(f"Physical page {physical_page} extracted only {len(text)} chars")
        pages.append(text)
    return pages


def _sentences(pages: list[str]) -> list[str]:
    text = " ".join(re.sub(r"\s+", " ", page).strip() for page in pages)
    rows = [row.strip() for row in re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9]|\())", text) if row.strip()]
    return [row for row in rows if len(row) >= 35]


def _short_blocks(pages: list[str]) -> tuple[list[str], dict[str, int]]:
    rows = _sentences(pages)
    anchor_index: dict[str, int] = {}
    for anchor in ANCHORS:
        index = next((i for i, row in enumerate(rows) if anchor.casefold() in row.casefold()), None)
        if index is None:
            raise RuntimeError(f"Goodfellow short benchmark missing anchor {anchor!r}")
        anchor_index[anchor] = index

    selected: set[int] = set(anchor_index.values())
    # Grow context symmetrically around the anchor sentences until the sample is
    # substantial enough for style/terminology checks but still far below six pages.
    radius = 1
    while True:
        candidate = set(selected)
        for index in anchor_index.values():
            for pos in (index - radius, index + radius):
                if 0 <= pos < len(rows):
                    candidate.add(pos)
        chars = sum(len(rows[i]) + 2 for i in sorted(candidate))
        if chars <= TARGET_MAX_CHARS:
            selected = candidate
        if chars >= TARGET_MIN_CHARS or radius >= 4 or chars > TARGET_MAX_CHARS:
            break
        radius += 1

    ordered = sorted(selected)
    # Keep contiguous sentence runs as independent FB2 paragraphs. This preserves
    # local context without merging unrelated page regions into one mega-segment.
    blocks: list[str] = []
    group: list[str] = []
    prev: int | None = None
    for index in ordered:
        if prev is not None and index != prev + 1 and group:
            blocks.append(" ".join(group))
            group = []
        group.append(rows[index])
        prev = index
    if group:
        blocks.append(" ".join(group))

    source_chars = sum(len(block) for block in blocks) + max(0, len(blocks) - 1) * 2
    if source_chars < 1200 or source_chars > TARGET_MAX_CHARS + 600:
        raise RuntimeError(f"Unexpected short Goodfellow benchmark size: {source_chars} chars")
    return blocks, anchor_index


def _fb2(paragraphs: list[str]) -> str:
    body = "\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in paragraphs)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <genre>science</genre>
      <author><first-name>Ian</first-name><last-name>Goodfellow</last-name></author>
      <author><first-name>Yoshua</first-name><last-name>Bengio</last-name></author>
      <author><first-name>Aaron</first-name><last-name>Courville</last-name></author>
      <book-title>Deep Learning — short cross-book benchmark excerpt</book-title>
      <lang>en</lang>
    </title-info>
    <document-info><id>bookai-crossbook-goodfellow-short</id><version>1.0</version></document-info>
  </description>
  <body>
    <section>
      <title><p>Chapter One</p></title>
{body}
    </section>
  </body>
</FictionBook>
'''


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "crossbook-goodfellow")
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _page_text(_download_pdf())
    selected, anchors = _short_blocks(pages)
    source_text = "\n\n".join(selected)

    required = ("deep learning", "neural", "LSTM", "CIFAR")
    missing = [term for term in required if term.casefold() not in source_text.casefold()]
    if missing:
        raise RuntimeError(f"Short benchmark lost required technical coverage; missing={missing}")

    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "physical_pages_scanned": [PAGE_FIRST, PAGE_LAST],
        "source_chars": len(source_text),
        "segments": len(selected),
        "anchor_sentence_indices": anchors,
        "reference_text_embedded": False,
        "selection": "short technical smoke excerpt selected from physical PDF pages 35–40; no gold translation used by pipeline",
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
