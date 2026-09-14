from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path


SOURCE_URL = "https://www.gutenberg.org/cache/epub/20203/pg20203.txt"
START_ANCHOR = "Dear son: I have ever had pleasure in obtaining any little anecdotes"
END_ANCHOR = "The notes one of my uncles"
TARGET_MIN_CHARS = 2200
TARGET_MAX_CHARS = 4800
VARIANT = (os.getenv("BOOKAI_SHORT_VARIANT") or "a").strip().casefold()


def _norm(text: str) -> str:
    value = str(text or "").replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _download_text() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Book-reader-AI cross-domain benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    if "AUTOBIOGRAPHY" not in text or START_ANCHOR[:32] not in text:
        raise RuntimeError("Franklin Gutenberg source changed or expected autobiography text is missing")
    return text


def _clean_blocks(text: str) -> list[str]:
    blocks = [_norm(block) for block in re.split(r"\n\s*\n", text) if _norm(block)]
    start = next((i for i, block in enumerate(blocks) if START_ANCHOR.casefold() in block.casefold()), None)
    if start is None:
        raise RuntimeError("Franklin benchmark start anchor not found")
    cleaned: list[str] = []
    for block in blocks[start:]:
        low = block.casefold()
        if "end of the project gutenberg" in low:
            break
        if re.match(r"^\[\d+\]", block):
            continue
        if block.startswith("[Illustration:") or block.startswith("[Transcriber's note:"):
            continue
        if block.startswith("Gibbon and Hume, the great British historians"):
            continue
        if block.count(" ") < 20:
            continue
        if sum(ch.isalpha() for ch in block) / max(1, len(block)) < 0.58:
            continue
        cleaned.append(block)
    return cleaned


def _opening_sample(blocks: list[str]):
    selected: list[str] = []
    selected_indices: list[int] = []
    for i, block in enumerate(blocks):
        if END_ANCHOR.casefold() in block.casefold():
            break
        selected.append(block)
        selected_indices.append(i)
    joined = "\n\n".join(selected)
    if not selected or not (TARGET_MIN_CHARS <= len(joined) <= TARGET_MAX_CHARS):
        raise RuntimeError(f"Unexpected Franklin benchmark size: {len(joined)} chars")
    required = ("poverty and obscurity", "second edition", "Without vanity I may say", "thank God for his vanity", "kind providence")
    missing = [phrase for phrase in required if phrase.casefold() not in joined.casefold()]
    if missing:
        raise RuntimeError(f"Franklin benchmark lost required narrative coverage: {missing}")
    return selected, {"selection_mode": "opening-regression-a"}, set(selected_indices)


def _window_at(blocks: list[str], fraction: float, mode: str, excluded: set[int] | None = None):
    if len(blocks) < 30:
        raise RuntimeError(f"Not enough Franklin narrative blocks: {len(blocks)}")
    excluded = set(excluded or ())
    center = round(fraction * (len(blocks) - 1))
    selected_indices: list[int] = []
    chars = 0
    radius = 0
    while chars < 3000 and radius < 28:
        candidates = (center,) if radius == 0 else (center - radius, center + radius)
        for idx in candidates:
            if not (0 <= idx < len(blocks)) or idx in excluded or idx in selected_indices:
                continue
            size = len(blocks[idx]) + 2
            if chars + size > TARGET_MAX_CHARS:
                continue
            selected_indices.append(idx)
            chars += size
        radius += 1
        if chars >= TARGET_MIN_CHARS and len(selected_indices) >= 4 and radius > 4:
            break
    selected_indices.sort()
    selected = [blocks[idx] for idx in selected_indices]
    joined = "\n\n".join(selected)
    if not (TARGET_MIN_CHARS <= len(joined) <= TARGET_MAX_CHARS):
        raise RuntimeError(f"Unexpected Franklin benchmark size: {len(joined)} chars")
    return selected, {
        "selection_mode": mode,
        "center_clean_block": center,
        "selected_clean_block_indices": selected_indices,
        "excluded_prior_variant_indices": len(excluded),
    }, set(selected_indices)


def _unseen_window(blocks: list[str]):
    return _window_at(blocks, 0.38, "unseen-mid-memoir-window-b")


def _unseen_window_c(blocks: list[str]):
    _, _, prior = _unseen_window(blocks)
    return _window_at(blocks, 0.72, "unseen-late-memoir-window-c", prior)


def _fb2(paragraphs: list[str], *, title: str, document_id: str) -> str:
    body = "\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in paragraphs)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <genre>biography</genre>
      <author><first-name>Benjamin</first-name><last-name>Franklin</last-name></author>
      <book-title>{html_lib.escape(title)}</book-title>
      <lang>en</lang>
    </title-info>
    <document-info><id>{html_lib.escape(document_id)}</id><version>1.0</version></document-info>
  </description>
  <body>
    <section><title><p>Chapter One</p></title>
{body}
    </section>
  </body>
</FictionBook>
'''


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "short-franklin")
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = _clean_blocks(_download_text())
    if VARIANT in {"c", "fresh", "unseen-c"}:
        selected, selection_meta, _selected_indices = _unseen_window_c(blocks)
    elif VARIANT in {"b", "alt", "unseen"}:
        selected, selection_meta, _selected_indices = _unseen_window(blocks)
    else:
        selected, selection_meta, _selected_indices = _opening_sample(blocks)
    source_text = "\n\n".join(selected)

    (out_dir / "sample.fb2").write_text(
        _fb2(selected, title="Autobiography of Benjamin Franklin — short anti-overfit benchmark", document_id="bookai-crossbook-franklin-short"),
        "utf-8",
    )
    # The memory source is intentionally independent of the chosen target window.
    (out_dir / "memory-source.fb2").write_text(
        _fb2(blocks, title="Autobiography of Benjamin Franklin — stable whole-book memory source", document_id="bookai-crossbook-franklin-memory"),
        "utf-8",
    )
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "variant": VARIANT,
        "source_url": SOURCE_URL,
        "source_format": "project-gutenberg-plain-text",
        "source_chars": len(source_text),
        "segments": len(selected),
        "memory_scope": "stable-whole-book-source",
        "memory_segments": len(blocks),
        "memory_chars": sum(len(row) for row in blocks),
        "memory_source": "memory-source.fb2",
        "reference_text_embedded": False,
        **selection_meta,
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
