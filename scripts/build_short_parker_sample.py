from __future__ import annotations

import html as html_lib
import json
import sys
from pathlib import Path

from bookai.parsers.base import load_book


# Five dense regression paragraphs are enough for the fast inner loop: literary
# idiom + an uncommon defined term, two material contrasts, coupled dimensions,
# and recurring/composite names. Broader semantic cases stay for larger audits.
# The selectors are source-only; no Russian gold/reference text is embedded.
ANCHORS = (
    "last lesson but one",
    "brass bushing",
    "not brass but bronze",
    "half-inch steel rods",
    "Miel Ducas",
)
MAX_SEGMENTS = 5
MAX_SOURCE_CHARS = 4800


def _fb2(paragraphs: list[str]) -> str:
    body = "\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in paragraphs)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <genre>sf_fantasy</genre>
      <author><first-name>K. J.</first-name><last-name>Parker</last-name></author>
      <book-title>Devices and Desires — short regression excerpt</book-title>
      <lang>en</lang>
    </title-info>
    <document-info><id>bookai-short-parker</id><version>1.0</version></document-info>
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
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "Devices_and_Desires.fb2")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "short-parker")
    out_dir.mkdir(parents=True, exist_ok=True)

    document = load_book(source)
    segments = [s for s in document.segments if str(s.text or "").strip()]
    chosen = []
    seen: set[str] = set()
    matched: dict[str, str] = {}

    for anchor in ANCHORS:
        low_anchor = anchor.casefold()
        match = next((s for s in segments if low_anchor in str(s.text or "").casefold()), None)
        if match is None or match.id in seen:
            continue
        text = str(match.text or "").strip()
        if not text:
            continue
        if sum(len(str(s.text or "")) for s in chosen) + len(text) > MAX_SOURCE_CHARS and len(chosen) >= 4:
            continue
        chosen.append(match)
        seen.add(match.id)
        matched[anchor] = match.id
        if len(chosen) >= MAX_SEGMENTS:
            break

    if len(chosen) != len(ANCHORS):
        missing = [anchor for anchor in ANCHORS if anchor not in matched]
        raise RuntimeError(f"Short Parker benchmark resolved {len(chosen)}/{len(ANCHORS)} anchors; missing={missing}")

    paragraphs = [str(s.text or "").strip() for s in chosen]
    source_text = "\n\n".join(paragraphs)
    (out_dir / "sample.fb2").write_text(_fb2(paragraphs), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source": str(source),
        "segments": len(chosen),
        "source_chars": len(source_text),
        "source_segment_ids": [s.id for s in chosen],
        "source_chapters": [s.chapter for s in chosen],
        "matched_anchors": matched,
        "reference_text_embedded": False,
        "selection": "five-paragraph fast regression covering idiom, terminology, materials, dimensions and name consistency",
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
