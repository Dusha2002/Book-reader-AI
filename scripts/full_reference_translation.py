from __future__ import annotations

import json
from pathlib import Path

from bookai.models import Segment
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import PIPELINE_VERSION, _cache_path, _chapter_groups, _should_translate, translate_book
from bookai.reference_harness import build_reference_harness
from bookai.resume import first_complete_unchecked_chapter, sanitize_resume_state


SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU_REFERENCE.fb2")
CACHE = Path(".bookai-cache-reference-v10")
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
    """Continue past any late-stage error once the whole chapter has usable text.

    Explicit polish/QA errors retain their stage. Other errors are inferred only if
    the earliest unchecked chapter is already 100% translated. We never waive a
    partial chapter, so a genuine missing translation remains a real blocker.
    """
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    state = _cached_state(SOURCE, CACHE)
    if not state or state.get("pipeline_version") != PIPELINE_VERSION:
        return None

    chapters = _chapter_items()
    chapter_map = {name: chapter for name, chapter in chapters}
    classified = _waivable_failure(error)
    if classified is None:
        chapter_name = first_complete_unchecked_chapter(state, chapters)
        if chapter_name is None:
            return None
        stage = "chapter_exception"
    else:
        stage, chapter_name = classified

    chapter = chapter_map.get(chapter_name)
    if not chapter:
        return None
    chapter_ids = {segment.id for segment in chapter}
    translations = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if isinstance(text, str) and text.strip()
    }
    if not chapter_ids or not chapter_ids.issubset(translations):
        return None

    polished = set(state.get("polished_chapters") or [])
    completed = set(state.get("completed_chapters") or [])
    qa_passed = set(state.get("qa_passed_chapters") or [])

    polished.add(chapter_name)
    if stage != "polish":
        completed.add(chapter_name)
        qa_passed.add(chapter_name)

    state["polished_chapters"] = sorted(polished)
    state["completed_chapters"] = sorted(completed)
    state["qa_passed_chapters"] = sorted(qa_passed)
    state.pop("final_quality", None)

    waivers = dict(state.get("best_effort_waivers") or {})
    history = list(waivers.get(chapter_name) or [])
    entry = {"stage": stage, "error": str(error)[:2000]}
    if entry not in history:
        history.append(entry)
    waivers[chapter_name] = history
    state["best_effort_waivers"] = waivers
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    result = {
        "stage": stage,
        "chapter": chapter_name,
        "chapter_segments": len(chapter_ids),
        "waived_chapters": len(waivers),
    }
    print("[full-reference] best_effort_waiver=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


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

    if translated == total:
        status = "complete" if error is None else "complete_with_warnings"
    else:
        status = "partial"
    waivers = dict(state.get("best_effort_waivers") or {})
    report = {
        "status": status,
        "translated_segments": translated,
        "total_segments": total,
        "completion_percent": round((translated / total * 100.0) if total else 0.0, 2),
        "remaining_segments": max(0, total - translated),
        "qa_passed_chapters": list(state.get("qa_passed_chapters") or []),
        "best_effort_waived_chapters": sorted(waivers),
        "best_effort_waiver_count": len(waivers),
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
    max_waivers = 128
    for _ in range(max_waivers + 1):
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
            return
        except Exception as exc:
            report = _write_latest_artifact(exc)
            waiver = _record_best_effort_waiver(exc)
            if waiver is not None:
                continue
            if report["translated_segments"] <= 0:
                raise
            print(
                "[full-reference] best_effort_recovered=true; strict pipeline stopped, "
                "but cached translation was exported and the workflow may continue",
                flush=True,
            )
            print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)
            return

    report = _write_latest_artifact(RuntimeError("best-effort waiver limit reached"))
    print("[full-reference] waiver_limit_reached=true report=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
