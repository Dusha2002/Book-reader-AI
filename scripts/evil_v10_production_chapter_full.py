from __future__ import annotations

import os
import re

import bookai.v10 as v10_core
from bookai.pipeline import _chapter_groups, _should_translate
from bookai.v10_production_release_guard import (
    clean_numeric_result,
    clean_quantity_result,
    localized_gender_issues,
)

import evil_v10_production_chapter as production


_BASE_SELECT = production._select_numeric_or_named_chapter
_BASE_NUMERIC_COMPARE = v10_core.compare_numeric_fidelity
_BASE_CLEAN_QUANTITY = production._clean_quantity_with_compounds
_BARE_NUMBER = re.compile(r"^\d+$")


def _select_with_optional_prelude(document, chapter_name: str):
    """Full-book-only selector that attaches leading translatable front matter to chapter 1.

    The normal production selector remains unchanged, so the chapter-one A/B benchmark
    still targets the 135 literary chapter segments. For a complete reading build we
    additionally need the two translatable segments that precede the first numbered
    chapter in this FB2. They are translated under the same Book Bible/QA contract and
    checkpointed together with the first chapter instead of becoming a fake chapter.
    """
    include_prelude = str(os.getenv("BOOKAI_V10_INCLUDE_PRELUDE") or "").casefold() in {
        "1", "true", "yes", "on"
    }
    wanted = " ".join(str(chapter_name or "").casefold().split())
    if not include_prelude or not _BARE_NUMBER.fullmatch(wanted):
        return _BASE_SELECT(document, chapter_name)

    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    numeric_indexes = [
        i for i, (name, _rows) in enumerate(groups)
        if _BARE_NUMBER.fullmatch(" ".join(str(name or "").casefold().split()))
    ]
    exact = [
        i for i in numeric_indexes
        if " ".join(str(groups[i][0] or "").casefold().split()) == wanted
    ]
    if len(exact) != 1:
        raise RuntimeError(f"Expected one numeric chapter {chapter_name!r}; found {len(exact)}")
    chapter_index = exact[0]
    if not numeric_indexes or chapter_index != numeric_indexes[0]:
        return _BASE_SELECT(document, chapter_name)

    end = len(groups)
    next_chapter = ""
    for i in numeric_indexes:
        if i > chapter_index:
            end = i
            next_chapter = str(groups[i][0])
            break

    selected_groups = groups[:end]
    targets = [segment for _name, rows in selected_groups for segment in rows]
    if not targets:
        raise RuntimeError(f"Chapter {chapter_name!r} + prelude selected no translatable segments")

    position = {segment.id: i for i, segment in enumerate(all_targets)}
    if end < len(groups):
        next_rows = groups[end][1]
        boundary_complete = bool(next_rows) and position[targets[-1].id] + 1 == position[next_rows[0].id]
    else:
        boundary_complete = True
    if not boundary_complete:
        raise RuntimeError(f"Prelude + chapter {chapter_name!r} is not contiguous with next numeric boundary")

    prelude_groups = groups[:chapter_index]
    prelude_segments = sum(len(rows) for _name, rows in prelude_groups)
    return all_targets, str(groups[chapter_index][0]), targets, {
        "requested": chapter_name,
        "matched": str(groups[chapter_index][0]),
        "included_groups": [name for name, _rows in selected_groups],
        "next_numbered_chapter": next_chapter or None,
        "boundary_complete": True,
        "segments": len(targets),
        "selector": "production-bare-numeric-v4+prelude",
        "included_prelude": True,
        "prelude_segments": prelude_segments,
    }


def _range_safe_numeric(source: str, target: str):
    return clean_numeric_result(source, _BASE_NUMERIC_COMPARE(source, target))


def _range_safe_quantity(cls, source: str, target: str):
    return clean_quantity_result(
        source,
        _BASE_CLEAN_QUANTITY(cls, source, target),
    )


# These patches are intentionally full-book-only. The standalone chapter benchmark
# remains frozen, while production reading builds get false-positive suppression for
# coordinated numeric ranges and speaker-local gender attribution.
production._select_numeric_or_named_chapter = _select_with_optional_prelude
production._safe_gender_issues = localized_gender_issues
production._clean_quantity_with_compounds = _range_safe_quantity
v10_core.compare_numeric_fidelity = _range_safe_numeric


if __name__ == "__main__":
    production.main()
