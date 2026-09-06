from __future__ import annotations

import json
from pathlib import Path

from bookai.models import Segment
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import PIPELINE_VERSION, _cache_path, _chapter_groups, _should_translate, translate_book
from bookai.reference_harness import build_reference_harness


SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU_REFERENCE.fb2")
CACHE = Path(".bookai-cache-reference-v10")
PROGRESS_REPORT = Path("reference-progress.json")


def progress(event: dict) -> None:
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _sanitize_resume_state(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> tuple[dict, dict]:
    """Keep every usable cached translation, including unfinished chapters.

    The old strict resume policy deleted a whole unfinished chapter unless it had
    already passed every QA gate. That made transient model/JSON failures extremely
    expensive because the next run translated the same material again. Best-effort
    mode keeps all non-empty source translations and only revokes chapter-level
    completion claims that are impossible because ids are missing.
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

    state = dict(state)
    state["translations"] = translations
    state["completed_chapters"] = sorted(completed)
    state["polished_chapters"] = sorted(polished)
    state["qa_passed_chapters"] = sorted(qa_passed)

    all_ids = set(translations)
    if all_ids != source_ids or qa_passed != known_names:
        state.pop("final_quality", None)

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
    return state, report


def _sanitize_resume_cache(source: Path, cache_dir: Path, mode: str = "optimal") -> dict:
    state_path = _cache_path(source, cache_dir, mode)
    empty = {
        "preserved_translations": 0,
        "removed_invalid_or_stale": 0,
        "partial_chapters": [],
        "qa_passed_chapters": 0,
    }
    if not state_path.exists():
        return empty

    try:
        state = json.loads(state_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    if state.get("pipeline_version") != PIPELINE_VERSION:
        return empty

    document = load_book(source)
    source_segments = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(source_segments)
    cleaned, report = _sanitize_resume_state(state, chapters)
    if cleaned != state:
        state_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")
    return report


def _cached_state(source: Path, cache_dir: Path, mode: str = "optimal") -> dict:
    state_path = _cache_path(source, cache_dir, mode)
    try:
        return json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_latest_artifact(error: BaseException | None = None) -> dict:
    """Build the newest usable FB2 even when the strict pipeline aborted midway."""
    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    target_ids = {segment.id for segment in targets}
    state = _cached_state(SOURCE, CACHE)
    translations = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if str(sid) in target_ids and isinstance(text, str) and text.strip()
    }

    translated = len(translations)
    total = len(targets)
    if translated:
        save_book(document, translations, OUTPUT)

    status = "complete" if error is None and translated == total else "partial"
    report = {
        "status": status,
        "translated_segments": translated,
        "total_segments": total,
        "completion_percent": round((translated / total * 100.0) if total else 0.0, 2),
        "remaining_segments": max(0, total - translated),
        "qa_passed_chapters": list(state.get("qa_passed_chapters") or []),
        "output_exists": OUTPUT.exists(),
        "output_bytes": OUTPUT.stat().st_size if OUTPUT.exists() else 0,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error)[:2000] if error is not None else None,
    }
    PROGRESS_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("[full-reference] progress=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    resume = _sanitize_resume_cache(SOURCE, CACHE)
    print("[full-reference] resume=" + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)

    harness = build_reference_harness()
    try:
        result = translate_book(
            SOURCE,
            OUTPUT,
            harness,
            mode="optimal",
            cache_dir=CACHE,
            progress=progress,
            memory_updates=True,
        )
        print(f"[full-reference] output={result} bytes={result.stat().st_size}", flush=True)
        print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)
        _write_latest_artifact(None)
    except Exception as exc:
        report = _write_latest_artifact(exc)
        if report["translated_segments"] <= 0:
            raise
        print(
            "[full-reference] best_effort_recovered=true; strict pipeline stopped, "
            "but cached translation was exported and the workflow may continue",
            flush=True,
        )
        print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
