from __future__ import annotations

from .models import Segment
from .release_guards import source_fingerprint


def _usable_translations(state: dict, source_ids: set[str] | None = None) -> dict[str, str]:
    original = dict(state.get("translations") or {})
    return {
        str(sid): text
        for sid, text in original.items()
        if (source_ids is None or str(sid) in source_ids)
        and isinstance(text, str)
        and text.strip()
    }


def _review_required(state: dict) -> dict[str, dict]:
    raw = state.get("review_required_chapters") or {}
    if isinstance(raw, dict):
        return {str(name): dict(value or {}) for name, value in raw.items()}
    if isinstance(raw, list):
        return {str(name): {"reason": "legacy review-required marker"} for name in raw}
    return {}


def sanitize_resume_state(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> tuple[dict, dict]:
    """Keep usable cached work without ever promoting waived chapters to QA-passed."""
    original = dict(state.get("translations") or {})
    source_segments = {segment.id: segment for _, chapter in chapters for segment in chapter}
    source_ids = set(source_segments)
    translations = _usable_translations(state, source_ids)

    provenance = {
        str(sid): dict(row)
        for sid, row in dict(state.get("translation_provenance") or {}).items()
        if str(sid) in source_ids and isinstance(row, dict)
    }
    stale_by_hash = {
        sid
        for sid, row in provenance.items()
        if row.get("source_hash")
        and row.get("source_hash") != source_fingerprint(source_segments[sid].text)
    }
    for sid in stale_by_hash:
        translations.pop(sid, None)
        provenance.pop(sid, None)

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

    review_required = {
        name: payload
        for name, payload in _review_required(state).items()
        if name in known_names
        and chapter_ids[name]
        and chapter_ids[name].issubset(translations)
    }
    # A chapter cannot simultaneously be certified and explicitly require review.
    qa_passed.difference_update(review_required)

    cleaned = dict(state)
    cleaned["translations"] = translations
    cleaned["translation_provenance"] = provenance
    cleaned["completed_chapters"] = sorted(completed)
    cleaned["polished_chapters"] = sorted(polished)
    cleaned["qa_passed_chapters"] = sorted(qa_passed)
    cleaned["review_required_chapters"] = review_required

    all_ids = set(translations)
    if all_ids != source_ids or qa_passed != known_names or review_required:
        cleaned.pop("final_quality", None)

    partial_chapters = [
        name
        for name, ids in chapter_ids.items()
        if ids.intersection(translations) and not ids.issubset(translations)
    ]
    report = {
        "preserved_translations": len(translations),
        "removed_invalid_or_stale": len(original) - len(translations),
        "stale_source_hashes": len(stale_by_hash),
        "partial_chapters": partial_chapters,
        "qa_passed_chapters": len(qa_passed),
        "review_required_chapters": sorted(review_required),
        "provenance_coverage": len(provenance),
    }
    return cleaned, report


def first_complete_unchecked_chapter(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> str | None:
    """Find a fully translated chapter that is neither QA-passed nor already waived."""
    translations = _usable_translations(state)
    qa_passed = set(state.get("qa_passed_chapters") or [])
    review_required = set(_review_required(state))
    for name, chapter in chapters:
        ids = {segment.id for segment in chapter}
        if name not in qa_passed and name not in review_required and ids and ids.issubset(translations):
            return name
    return None
