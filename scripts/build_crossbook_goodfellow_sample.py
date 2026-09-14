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
# We extract physical PDF pages 35–40, matching the user-provided original range.
SOURCE_URL = "https://raw.githubusercontent.com/janishar/mit-deep-learning-book-pdf/master/complete-book-bookmarked-pdf/deeplearningbook.pdf"
PAGE_FIRST = 35
PAGE_LAST = 40

_LIGATURES = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "–": "-", "—": "—", "’": "'", "“": '“', "”": '”',
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
})


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_LIGATURES).replace("\u00ad", "")
    # Join only typesetting hyphenation at a newline before a lowercase continuation.
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
        # Strip repeated running header and a standalone printed page number, while
        # preserving section numbers, citations, quantities and figure references.
        text = re.sub(r"(?im)^\s*CHAPTER\s+1\.\s+INTRODUCTION\s*$", "", text)
        text = re.sub(r"(?im)^\s*\d{1,2}\s*$", "", text)
        text = _norm(text)
        if len(text) < 200:
            raise RuntimeError(f"Physical page {physical_page} extracted only {len(text)} chars")
        pages.append(text)
    return pages


def _prose_blocks(pages: list[str]) -> list[str]:
    # Keep page boundaries as segment boundaries. Figure/axis fragments are useful to
    # the diagnostic too: a general translator must not hallucinate around numbers,
    # labels, acronyms or cross-references. No gold/reference text is used here.
    return [f"[Physical page {PAGE_FIRST + i}]\n{text}" for i, text in enumerate(pages)]


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
      <book-title>Deep Learning — cross-book benchmark excerpt</book-title>
      <lang>en</lang>
    </title-info>
    <document-info><id>bookai-crossbook-goodfellow</id><version>1.0</version></document-info>
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
    selected = _prose_blocks(pages)
    source_text = "\n\n".join(selected)
    if not (7000 <= len(source_text) <= 30000):
        raise RuntimeError(f"Unexpected pages 35–40 benchmark size: {len(source_text)} chars")
    # Sanity checks describe the domain rather than exact prose and catch wrong-edition downloads.
    required = ("deep learning", "neural", "LSTM", "CIFAR")
    missing = [term for term in required if term.casefold() not in source_text.casefold()]
    if missing:
        raise RuntimeError(f"Benchmark pages do not match expected chapter/domain; missing={missing}")

    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "physical_pages": [PAGE_FIRST, PAGE_LAST],
        "page_count": len(pages),
        "source_chars": len(source_text),
        "reference_text_embedded": False,
        "selection": "physical PDF pages 35–40; no gold translation used by the pipeline",
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
