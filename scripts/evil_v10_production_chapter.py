from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from bookai.pipeline import _chapter_groups, _should_translate
import bookai.v10_release as v10_release


_BASE_SELECT = v10_release.select_numbered_chapter
_BARE_NUMBER = re.compile(r"^\d+$")


def _select_numeric_or_named_chapter(document, chapter_name: str):
    """Production selector that also accepts FB2 chapter labels like '1', '2', ...

    Clean v10 originally accepted only 'Chapter One/Chapter 1'. Parker's Evil for Evil
    uses bare numeric section labels, so treating only English prose headings as chapter
    boundaries is unsafe. Bare numbers are accepted only when the whole label is numeric;
    labels such as '1. Minutes of previous meeting' remain ordinary subheadings.
    """
    wanted = " ".join(str(chapter_name or "").casefold().split())
    if not _BARE_NUMBER.fullmatch(wanted):
        return _BASE_SELECT(document, chapter_name)

    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    exact = [i for i, (name, _rows) in enumerate(groups) if " ".join(str(name or "").casefold().split()) == wanted]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected one bare numeric chapter matching {chapter_name!r}; found {len(exact)}; "
            f"available={[name for name, _ in groups]}"
        )

    start = exact[0]
    end = len(groups)
    next_chapter = ""
    for i in range(start + 1, len(groups)):
        label = " ".join(str(groups[i][0] or "").casefold().split())
        if _BARE_NUMBER.fullmatch(label):
            end = i
            next_chapter = groups[i][0]
            break

    selected_groups = groups[start:end]
    targets = [segment for _name, rows in selected_groups for segment in rows]
    if not targets:
        raise RuntimeError(f"Chapter {chapter_name!r} selected no translatable segments")

    position = {segment.id: i for i, segment in enumerate(all_targets)}
    boundary_complete = True
    if end < len(groups):
        next_rows = groups[end][1]
        boundary_complete = bool(next_rows) and position[targets[-1].id] + 1 == position[next_rows[0].id]
    if not boundary_complete:
        raise RuntimeError(f"Chapter {chapter_name!r} is not contiguous with next numeric boundary")

    return all_targets, groups[start][0], targets, {
        "requested": chapter_name,
        "matched": groups[start][0],
        "included_groups": [name for name, _rows in selected_groups],
        "next_numbered_chapter": next_chapter or None,
        "boundary_complete": True,
        "segments": len(targets),
        "selector": "production-bare-numeric-v1",
    }


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def main() -> None:
    # Patch only chapter-boundary recognition before release9 freezes its imported symbol.
    v10_release.select_numbered_chapter = _select_numeric_or_named_chapter
    import chapter_translation_v10_release9 as release9

    source = Path(os.getenv("BOOKAI_SOURCE") or "Evil for evil.fb2")
    if not source.exists() or source.stat().st_size < 10_000:
        raise RuntimeError(f"invalid/missing source: {source}")
    chapter = str(os.getenv("BOOKAI_CHAPTER_NAME") or "1")
    fp = _fingerprint(source)
    slug = re.sub(r"[^a-z0-9]+", "-", chapter.casefold()).strip("-") or "chapter"

    # Source isolation: book memory and cache are always tied to the exact source bytes.
    release9.SOURCE = source
    release9.MEMORY_SOURCE = source
    release9.CHAPTER_NAME = chapter
    release9.SLUG = slug
    release9.OUTPUT = Path(os.getenv("BOOKAI_V10_OUTPUT") or f"Evil_for_Evil_RU_V10_{slug}.fb2")
    release9.REPORT = Path(os.getenv("BOOKAI_V10_REPORT") or f"v10-evil-chapter-{slug}-report.json")
    release9.MAP = Path(os.getenv("BOOKAI_V10_MAP") or f"v10-evil-chapter-{slug}-map.json")
    release9.SOURCE_TXT = Path(os.getenv("BOOKAI_V10_SOURCE_TXT") or f"v10-evil-chapter-{slug}-source.txt")
    release9.TRANSLATED_TXT = Path(os.getenv("BOOKAI_V10_TRANSLATED_TXT") or f"v10-evil-chapter-{slug}-translated.txt")
    release9.BIBLE = Path(os.getenv("BOOKAI_V10_BIBLE_CACHE") or f".bookai-cache-v10-{fp}/book-bible.json")
    release9.BIBLE.parent.mkdir(parents=True, exist_ok=True)

    print(f"[v10-production-wrapper] source={source} fingerprint={fp} chapter={chapter} bible={release9.BIBLE}", flush=True)
    release9.main()


if __name__ == "__main__":
    main()
