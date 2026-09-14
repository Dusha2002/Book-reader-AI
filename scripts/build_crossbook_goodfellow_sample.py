from __future__ import annotations

import html as html_lib
import json
import re
import sys
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

SOURCE_URL = "https://www.deeplearningbook.org/contents/intro.html"
START_MARKER = "One may wonder why deep learning has only recently become recognized"
END_MARKER = "This trend is generally expected to continue well into the future."


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _download() -> str:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Book-reader-AI cross-book benchmark/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _paragraphs(raw_html: str) -> list[str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    rows: list[str] = []
    for p in soup.find_all("p"):
        text = _norm(p.get_text(" ", strip=True))
        if len(text) < 80:
            continue
        if re.match(r"^Figure\s+\d", text, re.I):
            continue
        rows.append(text)
    return rows


def _select(rows: list[str]) -> list[str]:
    start = next((i for i, row in enumerate(rows) if START_MARKER in row), None)
    end = next((i for i, row in enumerate(rows) if END_MARKER in row), None)
    if start is None or end is None or end < start:
        raise RuntimeError(
            f"Could not resolve benchmark markers: start={start}, end={end}, paragraphs={len(rows)}"
        )
    selected = rows[start : end + 1]
    if not (5 <= len(selected) <= 40):
        raise RuntimeError(f"Unexpected benchmark paragraph count: {len(selected)}")
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
    raw = _download()
    rows = _paragraphs(raw)
    selected = _select(rows)
    source_text = "\n\n".join(selected)
    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "selection": "continuous prose from sections 1.2.2–1.2.3; representative of the user-requested pp. 35–40 region",
        "start_marker": START_MARKER,
        "end_marker": END_MARKER,
        "paragraphs": len(selected),
        "source_chars": len(source_text),
        "reference_text_embedded": False,
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
