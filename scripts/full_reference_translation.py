from __future__ import annotations

import json
from pathlib import Path

from bookai.models import Segment
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import PIPELINE_VERSION, _cache_path, _chapter_groups, _should_translate, translate_book
from bookai.reference_harness import build_reference_harness
from bookai.resume import sanitize_resume_state


SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU_REFERENCE.fb2")
CACHE = Path(".bookai-cache-reference-v11-gigachat")
PROGRESS_REPORT = Path("reference-progress.json")


def progress(event: dict) -> None:
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _sanitize_resume_cache(source: Path, cache_dir: Path, mode: str = "optimal") -> dict:
    state_path = _cache_path(source, cache_dir, mode)
    empty = {
        "preserved_translations": 0,
        "removed_invalid_or_stale": 0,
        "partial_chapters": [],
        "qa_passed_chapters": 0,
        "review_required_chapters": [],
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
    cleaned, report = sanitize_resume_state(state, chapters)
    if cleaned != state:
        state_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")
    return report


def _cached_state(source: Path, cache_dir: Path, mode: str = "optimal") -> dict:
    state_path = _cache_path(source, cache_dir, mode)
    try:
        return json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _chapter_items() -> list[tuple[str, list[Segment]]]:
    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    return _chapter_groups(targets)


def _waivable_failure(error: BaseException) -> tuple[str, str] | None:
    text = str(error)
    polish_prefix = "Literary polish failed strict acceptance in chapter "
    if text.startswith(polish_prefix):
        return "polish", text[len(polish_prefix):].strip()

    qa_prefix = "Chapter "
    qa_marker = " failed final literary QA;"
    if text.startswith(qa_prefix) and qa_marker in text:
        chapter = text[len(qa_prefix):].split(qa_marker, 1)[0].strip()
        return "qa", chapter
    return None


def _record_best_effort_waiver(error: BaseException) -> dict | None:
    """Preserve a complete chapter without ever certifying it as QA-passed.

    A waived late-stage failure becomes review_required. The pipeline knows how to
    skip that chapter on the next resumable iteration so the rest of the book can
    progress, but final quality remains needs_review until the marker is cleared by
    a successful re-check.
    """
    failure = _waivable_failure(error)
    if failure is None:
        return None
    phase, chapter_name = failure
    chapters = dict(_chapter_items())
    chapter = chapters.get(chapter_name)
    if not chapter:
        return None

    state_path = _cache_path(SOURCE, CACHE, "optimal")
    state = _cached_state(SOURCE, CACHE)
    translations = dict(state.get("translations") or {})
    if not all(str(translations.get(segment.id) or "").strip() for segment in chapter):
        return None

    entry = {
        "phase": phase,
        "chapter": chapter_name,
        "reason": str(error)[:1200],
    }
    waivers = list(state.get("best_effort_waivers") or [])
    waivers.append(entry)
    state["best_effort_waivers"] = waivers[-64:]

    review_required = dict(state.get("review_required_chapters") or {})
    review_required[chapter_name] = entry
    state["review_required_chapters"] = review_required

    completed = set(state.get("completed_chapters") or [])
    completed.add(chapter_name)
    state["completed_chapters"] = sorted(completed)

    qa_passed = set(state.get("qa_passed_chapters") or [])
    qa_passed.discard(chapter_name)
    state["qa_passed_chapters"] = sorted(qa_passed)
    state.pop("final_quality", None)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    return entry


def _write_latest_artifact(error: BaseException | None) -> None:
    """Materialize the latest checkpoint even after timeout/failure."""
    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    state = _cached_state(SOURCE, CACHE)
    translations = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if isinstance(text, str) and text.strip()
    }
    save_book(document, translations, OUTPUT)

    completed = sum(segment.id in translations for segment in targets)
    remaining = max(0, len(targets) - completed)
    first_pending = next((segment.id for segment in targets if segment.id not in translations), None)
    review_required = dict(state.get("review_required_chapters") or {})
    if remaining:
        status = "partial"
    elif review_required:
        status = "needs_review"
    else:
        status = "complete"
    report = {
        "status": status,
        "completed": completed,
        "total": len(targets),
        "remaining": remaining,
        "progress_percent": round((completed / max(1, len(targets))) * 100, 2),
        "first_pending": first_pending,
        "output": str(OUTPUT),
        "cache": str(CACHE),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "best_effort_waivers": state.get("best_effort_waivers") or [],
        "review_required_chapters": sorted(review_required),
        "final_quality": state.get("final_quality") or {},
    }
    PROGRESS_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    progress({"phase": "rapid_exit", **report})


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    harness = build_reference_harness()
    resume = _sanitize_resume_cache(SOURCE, CACHE)
    print("[full-reference] resume=" + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)
    try:
        while True:
            try:
                translate_book(
                    SOURCE,
                    OUTPUT,
                    harness,
                    mode="optimal",
                    cache_dir=CACHE,
                    progress=progress,
                )
                break
            except BaseException as exc:
                waiver = _record_best_effort_waiver(exc)
                if waiver is None:
                    raise
                print("[full-reference] needs_review=" + json.dumps(waiver, ensure_ascii=False), flush=True)
        _write_latest_artifact(None)
    except BaseException as exc:
        _write_latest_artifact(exc)
        raise


if __name__ == "__main__":
    main()
