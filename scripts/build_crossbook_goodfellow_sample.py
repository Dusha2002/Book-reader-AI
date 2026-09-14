from __future__ import annotations

import html as html_lib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

from pypdf import PdfReader

SOURCE_URL = "https://raw.githubusercontent.com/janishar/mit-deep-learning-book-pdf/master/complete-book-bookmarked-pdf/deeplearningbook.pdf"
PAGE_FIRST = 35
PAGE_LAST = 40
TARGET_MIN_CHARS = 2200
TARGET_MAX_CHARS = 4600
# Prose anchors only: unlike CIFAR in a figure body, these exercise terminology,
# acronyms, cross-references and technical prose without turning the smoke test
# into OCR/layout evaluation.
ANCHORS = ("deep learning", "faster CPUs", "general purpose GPUs", "LSTM")

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


def _page_text_poppler(pdf_bytes: bytes) -> list[str]:
    exe = shutil.which("pdftotext")
    if not exe:
        return []
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(pdf_bytes)
        handle.flush()
        proc = subprocess.run(
            [
                exe,
                "-f", str(PAGE_FIRST),
                "-l", str(PAGE_LAST),
                "-layout",
                "-enc", "UTF-8",
                handle.name,
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    raw_pages = proc.stdout.decode("utf-8", errors="replace").split("\f")
    pages = []
    for raw in raw_pages:
        text = _norm(raw)
        if len(text) >= 200:
            pages.append(text)
    return pages


def _page_text_pypdf(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if len(reader.pages) < PAGE_LAST:
        raise RuntimeError(f"Benchmark PDF has only {len(reader.pages)} pages")
    pages: list[str] = []
    for physical_page in range(PAGE_FIRST, PAGE_LAST + 1):
        raw = reader.pages[physical_page - 1].extract_text() or ""
        text = _norm(raw)
        if len(text) < 200:
            raise RuntimeError(f"Physical page {physical_page} extracted only {len(text)} chars")
        pages.append(text)
    return pages


def _page_text(pdf_bytes: bytes) -> tuple[list[str], str]:
    # Poppler reconstructs spaces from glyph positions substantially better than
    # pypdf on this particular typeset PDF. pypdf remains a portability fallback.
    pages = _page_text_poppler(pdf_bytes)
    extractor = "pdftotext-layout" if pages else "pypdf"
    if not pages:
        pages = _page_text_pypdf(pdf_bytes)

    cleaned: list[str] = []
    for text in pages:
        text = re.sub(r"(?im)^\s*CHAPTER\s+1\.\s+INTRODUCTION\s*$", "", text)
        text = re.sub(r"(?im)^\s*\d{1,2}\s*$", "", text)
        text = _norm(text)
        cleaned.append(text)
    return cleaned, extractor


def _sentences(pages: list[str]) -> list[str]:
    # Layout extraction leaves line breaks at column width. Join them after
    # dehyphenation; real sentence boundaries remain punctuation-based below.
    text = " ".join(re.sub(r"\s+", " ", page).strip() for page in pages)
    rows = [row.strip() for row in re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9]|\())", text) if row.strip()]
    return [row for row in rows if len(row) >= 35]


def _looks_space_corrupt(text: str) -> tuple[bool, list[str]]:
    """Detect layout glue only in the selected smoke excerpt, not all six pages.

    Figure labels and bibliography-like material elsewhere on the physical pages may
    contain long tokens legitimately. The selected prose itself must preserve common
    technical phrase boundaries and must not contain multiple implausibly long words.
    """
    low = str(text or "").casefold()
    glued_sentinels = [
        bad for bad in ("machinelearning", "neuralnetworks", "deeplearning", "generalpurpose")
        if bad in low
    ]
    long_tokens = re.findall(r"\b[a-z]{28,}\b", low)
    bad = glued_sentinels + long_tokens[:6]
    return bool(glued_sentinels or len(long_tokens) >= 2), bad


def _short_blocks(pages: list[str]) -> tuple[list[str], dict[str, int]]:
    rows = _sentences(pages)

    anchor_index: dict[str, int] = {}
    for anchor in ANCHORS:
        index = next((i for i, row in enumerate(rows) if anchor.casefold() in row.casefold()), None)
        if index is None:
            raise RuntimeError(f"Goodfellow short benchmark missing prose anchor {anchor!r}")
        anchor_index[anchor] = index

    selected: set[int] = set(anchor_index.values())
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

    selected_text = "\n\n".join(blocks)
    corrupt, evidence = _looks_space_corrupt(selected_text)
    if corrupt:
        raise RuntimeError(f"Selected Goodfellow prose still contains glued-word corruption: {evidence}")

    source_chars = len(selected_text)
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
    pages, extractor = _page_text(_download_pdf())
    selected, anchors = _short_blocks(pages)
    source_text = "\n\n".join(selected)

    required = ("deep learning", "neural", "LSTM", "CPU", "GPU")
    missing = [term for term in required if term.casefold() not in source_text.casefold()]
    if missing:
        raise RuntimeError(f"Short benchmark lost required technical coverage; missing={missing}")

    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "physical_pages_scanned": [PAGE_FIRST, PAGE_LAST],
        "extractor": extractor,
        "source_chars": len(source_text),
        "segments": len(selected),
        "anchor_sentence_indices": anchors,
        "reference_text_embedded": False,
        "selection": "short clean technical prose smoke excerpt from physical PDF pages 35–40; no gold translation used by pipeline",
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
