from __future__ import annotations

import html as html_lib
import json
import os
import sys
from pathlib import Path

from bookai.parsers.base import load_book


# Variant A is the original targeted regression. Variant B deliberately samples
# unrelated, distributed prose so short-cycle work cannot overfit known anchors.
ANCHORS = (
    "last lesson but one",
    "brass bushing",
    "not brass but bronze",
    "half-inch steel rods",
    "Miel Ducas",
)
MAX_SEGMENTS = 5
MAX_SOURCE_CHARS = 4800
VARIANT = (os.getenv("BOOKAI_SHORT_VARIANT") or "a").strip().casefold()


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


def _targeted_sample(segments):
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
    return chosen, {"matched_anchors": matched, "selection_mode": "targeted-regression-a"}


def _unseen_spread_sample(segments):
    old = tuple(anchor.casefold() for anchor in ANCHORS)
    eligible = []
    for segment in segments:
        text = str(segment.text or "").strip()
        low = text.casefold()
        if not (320 <= len(text) <= 1250):
            continue
        if any(anchor in low for anchor in old):
            continue
        # Reject headings/list-like fragments; retain ordinary continuous prose.
        if text.count(" ") < 45 or sum(ch.isalpha() for ch in text) / max(1, len(text)) < 0.62:
            continue
        eligible.append(segment)
    if len(eligible) < 20:
        raise RuntimeError(f"Not enough Parker prose for unseen spread sample: {len(eligible)}")

    fractions = (0.13, 0.31, 0.49, 0.67, 0.85)
    chosen = []
    seen: set[str] = set()
    chars = 0
    for fraction in fractions:
        center = round(fraction * (len(eligible) - 1))
        offsets = [0]
        for delta in range(1, 24):
            offsets.extend((delta, -delta))
        match = None
        for offset in offsets:
            pos = center + offset
            if not (0 <= pos < len(eligible)):
                continue
            candidate = eligible[pos]
            text = str(candidate.text or "").strip()
            if candidate.id in seen:
                continue
            if chars + len(text) > MAX_SOURCE_CHARS:
                continue
            match = candidate
            break
        if match is None:
            continue
        chosen.append(match)
        seen.add(match.id)
        chars += len(str(match.text or "").strip())

    if len(chosen) < 4 or chars < 2200:
        raise RuntimeError(f"Unseen Parker sample too small: segments={len(chosen)} chars={chars}")
    return chosen, {
        "selection_mode": "unseen-distributed-b",
        "fractions": list(fractions),
        "eligible_segments": len(eligible),
        "excluded_known_anchors": True,
    }


def main() -> None:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "Devices_and_Desires.fb2")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "short-parker")
    out_dir.mkdir(parents=True, exist_ok=True)

    document = load_book(source)
    segments = [s for s in document.segments if str(s.text or "").strip()]
    if VARIANT in {"b", "alt", "unseen"}:
        chosen, selection_meta = _unseen_spread_sample(segments)
    else:
        chosen, selection_meta = _targeted_sample(segments)

    paragraphs = [str(s.text or "").strip() for s in chosen]
    source_text = "\n\n".join(paragraphs)
    (out_dir / "sample.fb2").write_text(_fb2(paragraphs), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "variant": VARIANT,
        "source": str(source),
        "segments": len(chosen),
        "source_chars": len(source_text),
        "source_segment_ids": [s.id for s in chosen],
        "source_chapters": [s.chapter for s in chosen],
        "reference_text_embedded": False,
        **selection_meta,
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
