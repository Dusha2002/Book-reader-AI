from __future__ import annotations

import html as html_lib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

SOURCE_URL = "https://www.deeplearningbook.org/contents/intro.html"

WINDOWS = (
    ("The third wave of neural networks research began", "deep models to leverage large labeled datasets."),
    ("One may wonder why deep learning has only recently become recognized", "unsupervised or semi-supervised learning."),
    ("Another key reason that neural networks are wildly successful today", "An individual neuron or small collection of neurons is not particularly useful."),
    ("Biological neurons are not especially densely connected.", "biological neural networks may be even larger than this plot portrays."),
    ("In retrospect, it is not particularly surprising that neural networks with fewer", "expected to continue well into the future."),
)

_LIGATURES = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "–": "-", "—": "-", "’": "'", "“": '"', "”": '"',
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
})


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_LIGATURES).replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _download() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0 Book-reader-AI benchmark"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_page_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    text = _norm(soup.get_text(" ", strip=True))
    text = re.sub(r"\b\d{1,3}\s+CHAPTER 1\. INTRODUCTION\b", " ", text, flags=re.I)
    text = re.sub(r"\bCHAPTER 1\. INTRODUCTION\b", " ", text, flags=re.I)
    return _norm(text)


def _find_anchor(text: str, marker: str, *, start: int = 0) -> int:
    marker = _norm(marker)
    pos = text.find(marker, start)
    if pos >= 0:
        return pos
    words = re.findall(r"[A-Za-z0-9]+", marker)
    if not words:
        return -1
    pattern = r"\b" + r"[\s\W]{0,24}".join(re.escape(word) for word in words) + r"\b"
    match = re.search(pattern, text[start:], flags=re.I)
    return start + match.start() if match else -1


def _diagnostic(text: str, marker: str) -> str:
    low = text.casefold()
    probes = ["third wave", "neural networks", "deep learning", "introduction"]
    positions = {probe: low.find(probe) for probe in probes}
    snippets = {}
    for probe, pos in positions.items():
        if pos >= 0:
            snippets[probe] = text[max(0, pos - 120):pos + 360]
    return json.dumps({
        "marker": marker,
        "text_chars": len(text),
        "probe_positions": positions,
        "snippets": snippets,
        "head": text[:500],
    }, ensure_ascii=False)


def _extract_window(text: str, start_marker: str, end_marker: str) -> str:
    start_marker = _norm(start_marker)
    end_marker = _norm(end_marker)
    start = _find_anchor(text, start_marker)
    if start < 0:
        raise RuntimeError("Could not resolve benchmark start marker: " + _diagnostic(text, start_marker))
    end = _find_anchor(text, end_marker, start=start)
    if end < 0:
        raise RuntimeError("Could not resolve benchmark end marker: " + _diagnostic(text[start:], end_marker))
    end += len(end_marker)
    value = _norm(text[start:end])
    if len(value) < 100:
        raise RuntimeError(f"Benchmark window unexpectedly short: {len(value)} chars")
    return value


def _select(raw_html: str) -> list[str]:
    text = _clean_page_text(raw_html)
    selected = [_extract_window(text, start, end) for start, end in WINDOWS]
    source_chars = sum(len(row) for row in selected)
    if not (3000 <= source_chars <= 18000):
        raise RuntimeError(f"Unexpected benchmark size: {source_chars} chars")
    return selected


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
    selected = _select(_download())
    source_text = "\n\n".join(selected)
    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "selection": "five continuous prose windows spanning the late third-wave discussion and sections 1.2.2–1.2.3; figures omitted",
        "windows": len(selected),
        "source_chars": len(source_text),
        "reference_text_embedded": False,
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
