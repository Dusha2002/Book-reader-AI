from __future__ import annotations

from .models import Segment


def _usable_translations(state: dict, source_ids: set[str] | None = None) -> dict[str, str]:
    original = dict(state.get("translations") or {})
    return {
        str(sid): text
        for sid, text in original.items()
        if (source_ids is None or str(sid) in source_ids)
        and isinstance(text, str)
        and text.strip()
    }


def sanitize_resume_state(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> tuple[dict, dict]:
    """Keep every usable cached translation, including unfinished chapters."""
    original = dict(state.get("translations") or {})
    source_ids = {segment.id for _, chapter in chapters for segment in chapter}
    translations = _usable_translations(state, source_ids)

    chapter_ids = {name: {segment.id for segment in chapter} for name, chapter in chapters}
    known_names = set(chapter_ids)

    def valid_claims(key: str) -> set[str]:
        claimed = set(state.get(key) or []) & known_names
        return {
            name
            for name in claimed
            if chapter_ids[name] and chapter_ids[name].issubset(translations)
        }

    completed = valid_claims("completed_chapters")
    polished = valid_claims("polished_chapters")
    qa_passed = valid_claims("qa_passed_chapters")

    cleaned = dict(state)
    cleaned["translations"] = translations
    cleaned["completed_chapters"] = sorted(completed)
    cleaned["polished_chapters"] = sorted(polished)
    cleaned["qa_passed_chapters"] = sorted(qa_passed)

    all_ids = set(translations)
    if all_ids != source_ids or qa_passed != known_names:
        cleaned.pop("final_quality", None)

    partial_chapters = [
        name
        for name, ids in chapter_ids.items()
        if ids.intersection(translations) and not ids.issubset(translations)
    ]
    report = {
        "preserved_translations": len(translations),
        "removed_invalid_or_stale": len(original) - len(translations),
        "partial_chapters": partial_chapters,
        "qa_passed_chapters": len(qa_passed),
    }
    return cleaned, report


def first_complete_unchecked_chapter(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> str | None:
    """Find the earliest chapter safe to waive after an unexpected late-stage error.

    A chapter is safe to waive only when every one of its source ids already has a
    non-empty cached translation and it has not already been marked QA-passed.
    Partial chapters are never returned, so real translation gaps remain blockers.
    """
    translations = _usable_translations(state)
    qa_passed = set(state.get("qa_passed_chapters") or [])
    for name, chapter in chapters:
        ids = {segment.id for segment in chapter}
        if name not in qa_passed and ids and ids.issubset(translations):
            return name
    return None
