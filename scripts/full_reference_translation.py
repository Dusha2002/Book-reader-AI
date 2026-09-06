from __future__ import annotations

import json
from pathlib import Path

from bookai.models import Segment
from bookai.parsers.base import load_book
from bookai.pipeline import PIPELINE_VERSION, _cache_path, _chapter_groups, _should_translate, translate_book
from bookai.reference_harness import build_reference_harness


SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU_REFERENCE.fb2")
CACHE = Path(".bookai-cache-reference-v10")


def progress(event: dict) -> None:
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _sanitize_resume_state(
    state: dict,
    chapters: list[tuple[str, list[Segment]]],
) -> tuple[dict, dict]:
    """Trust cached prose only for chapters that already passed the full QA gate.

    A failed/interrupted chapter may contain polished or repeatedly edited text from
    an older harness revision. Reusing that prose would skip the current translator
    (including sentence decomposition). Keep completed QA-passed chapters, but make
    every other chapter start again from the faithful Flash translation pass.
    """
    translations = dict(state.get("translations") or {})
    claimed_qa = set(state.get("qa_passed_chapters") or [])

    trusted_chapters: set[str] = set()
    trusted_ids: set[str] = set()
    source_ids: set[str] = set()

    for name, chapter in chapters:
        ids = {segment.id for segment in chapter}
        source_ids.update(ids)
        if name in claimed_qa and ids and ids.issubset(translations):
            trusted_chapters.add(name)
            trusted_ids.update(ids)

    reset_chapters: list[str] = []
    for name, chapter in chapters:
        ids = {segment.id for segment in chapter}
        if name not in trusted_chapters and ids.intersection(translations):
            reset_chapters.append(name)

    removed_ids = sorted(sid for sid in translations if sid in source_ids and sid not in trusted_ids)
    # The cache is keyed by the source digest, so non-source ids are stale/corrupt
    # and should not survive a resume either.
    stale_ids = sorted(sid for sid in translations if sid not in source_ids)
    cleaned_translations = {sid: text for sid, text in translations.items() if sid in trusted_ids}

    state = dict(state)
    state["translations"] = cleaned_translations
    state["completed_chapters"] = sorted(trusted_chapters)
    state["polished_chapters"] = sorted(trusted_chapters)
    state["qa_passed_chapters"] = sorted(trusted_chapters)
    state.pop("final_quality", None)

    report = {
        "trusted_chapters": len(trusted_chapters),
        "trusted_translations": len(cleaned_translations),
        "reset_chapters": reset_chapters,
        "removed_translations": len(removed_ids) + len(stale_ids),
    }
    return state, report


def _sanitize_resume_cache(source: Path, cache_dir: Path, mode: str = "optimal") -> dict:
    state_path = _cache_path(source, cache_dir, mode)
    if not state_path.exists():
        return {
            "trusted_chapters": 0,
            "trusted_translations": 0,
            "reset_chapters": [],
            "removed_translations": 0,
        }

    try:
        state = json.loads(state_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "trusted_chapters": 0,
            "trusted_translations": 0,
            "reset_chapters": [],
            "removed_translations": 0,
        }

    if state.get("pipeline_version") != PIPELINE_VERSION:
        return {
            "trusted_chapters": 0,
            "trusted_translations": 0,
            "reset_chapters": [],
            "removed_translations": 0,
        }

    document = load_book(source)
    source_segments = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(source_segments)
    cleaned, report = _sanitize_resume_state(state, chapters)

    if cleaned != state:
        state_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")
    return report


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    resume = _sanitize_resume_cache(SOURCE, CACHE)
    print("[full-reference] resume_sanitize=" + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)

    harness = build_reference_harness()
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


if __name__ == "__main__":
    main()
