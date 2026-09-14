from __future__ import annotations

import html as html_lib
import json
import re
import sys
import urllib.request
from pathlib import Path


SOURCE_URL = "https://www.gutenberg.org/cache/epub/20203/pg20203.txt"
START_ANCHOR = "Dear son: I have ever had pleasure in obtaining any little anecdotes"
END_ANCHOR = "The notes one of my uncles"
TARGET_MIN_CHARS = 2200
TARGET_MAX_CHARS = 4800


def _norm(text: str) -> str:
    value = str(text or "").replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _download_text() -> str:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Book-reader-AI cross-domain benchmark/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    if "AUTOBIOGRAPHY" not in text or START_ANCHOR[:32] not in text:
        raise RuntimeError("Franklin Gutenberg source changed or expected autobiography text is missing")
    return text


def _paragraphs(text: str) -> list[str]:
    # Gutenberg plain text uses blank lines as paragraph boundaries and wraps prose
    # across physical lines. Normalize only inside each paragraph; do not OCR or
    # heuristically rewrite source wording.
    blocks = [_norm(block) for block in re.split(r"\n\s*\n", text) if _norm(block)]
    start = next((i for i, block in enumerate(blocks) if START_ANCHOR.casefold() in block.casefold()), None)
    if start is None:
        raise RuntimeError("Franklin benchmark start anchor not found")

    selected: list[str] = []
    for block in blocks[start:]:
        if END_ANCHOR.casefold() in block.casefold():
            break
        # Skip editor footnotes interleaved with Franklin's prose. They are not part
        # of the author's narrative and would distort voice/domain evaluation.
        if re.match(r"^\[\d+\]", block):
            continue
        if block.startswith("[Illustration:") or block.startswith("[Transcriber's note:"):
            continue
        selected.append(block)

    if not selected:
        raise RuntimeError("Franklin benchmark selected no narrative paragraphs")
    joined = "\n\n".join(selected)
    if not (TARGET_MIN_CHARS <= len(joined) <= TARGET_MAX_CHARS):
        raise RuntimeError(f"Unexpected Franklin benchmark size: {len(joined)} chars")

    required = (
        "poverty and obscurity",
        "second edition",
        "Without vanity I may say",
        "thank God for his vanity",
        "kind providence",
    )
    missing = [phrase for phrase in required if phrase.casefold() not in joined.casefold()]
    if missing:
        raise RuntimeError(f"Franklin benchmark lost required narrative coverage: {missing}")
    return selected


def _fb2(paragraphs: list[str]) -> str:
    body = "\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in paragraphs)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <genre>biography</genre>
      <author><first-name>Benjamin</first-name><last-name>Franklin</last-name></author>
      <book-title>Autobiography of Benjamin Franklin — short narrative nonfiction benchmark</book-title>
      <lang>en</lang>
    </title-info>
    <document-info><id>bookai-crossbook-franklin-short</id><version>1.0</version></document-info>
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
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "short-franklin")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _paragraphs(_download_text())
    source_text = "\n\n".join(selected)

    (out_dir / "sample.fb2").write_text(_fb2(selected), "utf-8")
    (out_dir / "sample-source.txt").write_text(source_text, "utf-8")
    meta = {
        "source_url": SOURCE_URL,
        "source_format": "project-gutenberg-plain-text",
        "source_chars": len(source_text),
        "segments": len(selected),
        "reference_text_embedded": False,
        "selection": (
            "short opening memoir excerpt covering long periodic syntax, autobiographical voice, "
            "self-irony, abstract reasoning and period register; no Russian gold used by pipeline"
        ),
    }
    (out_dir / "sample-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
