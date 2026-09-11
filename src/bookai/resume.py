from __future__ import annotations

from .models import Segment


def sanitize_resume_state(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> tuple[dict, dict]:
    """Keep every usable cached translation, including unfinished chapters.

    Invalid/stale ids and empty translations are discarded, but partial chapter
    progress is preserved. Chapter-level completion claims survive only when all
    source ids for that chapter have non-empty cached translations.
    """
    original = dict(state.get("translations") or {})
    source_ids = {segment.id for _, chapter in chapters for segment in chapter}
    translations = {
        str(sid): text
        for sid, text in original.items()
        if str(sid) in source_ids and isinstance(text, str) and text.strip()
    }

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
