from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from bookai.parsers.base import load_book, save_book
from bookai.pipeline import (
    PIPELINE_VERSION,
    _analysis_sample,
    _batches,
    _cache_path,
    _chapter_groups,
    _context_for,
    _memory_from_dict,
    _persist,
    _should_translate,
)
from bookai.quality import batch_issues, hard_ids
from bookai.reference_harness import build_reference_harness
from full_reference_translation import (
    CACHE,
    OUTPUT,
    SOURCE,
    _cached_state,
    _sanitize_resume_cache,
    _write_latest_artifact,
)


def progress(event: dict) -> None:
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _split_by_chars(batch):
    if len(batch) < 2:
        return batch, []
    total = sum(len(segment.text) for segment in batch)
    target = total / 2
    seen = 0
    split_at = 1
    for index, segment in enumerate(batch[:-1], 1):
        seen += len(segment.text)
        if seen >= target:
            split_at = index
            break
    return batch[:split_at], batch[split_at:]


def _translate_resilient(harness, source_segments, batch, memory, *, depth: int = 0):
    """Translate one batch; retry locally and split only the failing batch."""
    before, after = _context_for(source_segments, batch, radius=3)
    last_error: BaseException | None = None
    for attempt in range(2):
        try:
            candidate = harness.translate(
                batch,
                memory,
                context_before=before,
                context_after=after,
            )
            hard = hard_ids(batch_issues(batch, candidate, memory))
            if hard:
                raise ValueError("deterministic hard QA failed for: " + ", ".join(sorted(hard)))
            return candidate
        except BaseException as exc:
            last_error = exc
            print(
                "[bookai-rapid] retry="
                + json.dumps(
                    {
                        "attempt": attempt + 1,
                        "depth": depth,
                        "segments": len(batch),
                        "first": batch[0].id,
                        "last": batch[-1].id,
                        "error": type(exc).__name__,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    if len(batch) <= 1 or depth >= 7:
        assert last_error is not None
        raise last_error

    left, right = _split_by_chars(batch)
    print(
        "[bookai-rapid] split="
        + json.dumps(
            {
                "depth": depth,
                "segments": len(batch),
                "left": len(left),
                "right": len(right),
                "first": batch[0].id,
                "last": batch[-1].id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    out = _translate_resilient(harness, source_segments, left, memory, depth=depth + 1)
    if right:
        out.update(_translate_resilient(harness, source_segments, right, memory, depth=depth + 1))
    return out


def _load_or_build_memory(harness, state: dict, chapters):
    memory_data = state.get("memory")
    if memory_data:
        return _memory_from_dict(memory_data)
    progress({"phase": "analyzing_book", "progress": 2, "chapters": len(chapters)})
    analysis_chars = max(30000, int(os.getenv("BOOKAI_ANALYSIS_CHARS") or "70000"))
    return harness.analyze(_analysis_sample(chapters, analysis_chars))


def _parallel_translate(harness, targets, state, memory) -> dict[str, str]:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    translated: dict[str, str] = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if isinstance(text, str) and text.strip()
    }
    total = len(targets)
    batch_chars = max(12000, int(os.getenv("BOOKAI_RAPID_BATCH_CHARS") or "24000"))
    workers = max(2, min(8, int(os.getenv("BOOKAI_RAPID_WORKERS") or "6")))

    batches = []
    for batch in _batches(targets, batch_chars):
        pending = [segment for segment in batch if segment.id not in translated]
        if pending:
            batches.append(pending)

    progress(
        {
            "phase": "rapid_translate",
            "completed": sum(segment.id in translated for segment in targets),
            "total": total,
            "batches": len(batches),
            "workers": workers,
            "batch_chars": batch_chars,
            "strategy": "parallel-draft→deterministic-QA→local-split-retry→selective-repair",
        }
    )

    failures: list[tuple[list, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v41") as pool:
        future_to_batch = {
            pool.submit(_translate_resilient, harness, targets, batch, memory): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                accepted = future.result()
            except BaseException as exc:
                failures.append((batch, exc))
                print(
                    "[bookai-rapid] batch_failed="
                    + json.dumps(
                        {
                            "first": batch[0].id,
                            "last": batch[-1].id,
                            "segments": len(batch),
                            "error": type(exc).__name__,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            translated.update(accepted)
            _persist(state_path, state, translated, memory)
            completed = sum(segment.id in translated for segment in targets)
            progress(
                {
                    "phase": "rapid_batch_done",
                    "completed": completed,
                    "total": total,
                    "progress": round(completed / max(total, 1) * 100, 2),
                    "first": batch[0].id,
                    "last": batch[-1].id,
                }
            )

    for batch, original_error in failures:
        try:
            accepted = _translate_resilient(harness, targets, batch, memory, depth=1)
        except BaseException as exc:
            _persist(state_path, state, translated, memory)
            raise RuntimeError(
                f"Rapid translation could not recover batch {batch[0].id}..{batch[-1].id}"
            ) from exc
        translated.update(accepted)
        _persist(state_path, state, translated, memory)
        completed = sum(segment.id in translated for segment in targets)
        progress(
            {
                "phase": "rapid_salvage_done",
                "completed": completed,
                "total": total,
                "progress": round(completed / max(total, 1) * 100, 2),
                "first": batch[0].id,
                "last": batch[-1].id,
                "previous_error": type(original_error).__name__,
            }
        )

    return translated


def _selective_repair(harness, targets, translated, memory, state) -> dict[str, str]:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    issues = batch_issues(targets, translated, memory)
    hard = [issue for issue in issues if issue.severity == "hard"]
    if not hard:
        progress({"phase": "deterministic_qa", "hard": 0})
        return translated

    hard_ids_set = {issue.id for issue in hard}
    repair_targets = [segment for segment in targets if segment.id in hard_ids_set]
    progress({"phase": "selective_repair", "hard": len(repair_targets)})
    for batch in _batches(repair_targets, 12000):
        accepted = _translate_resilient(harness, targets, batch, memory, depth=1)
        translated.update(accepted)
        _persist(state_path, state, translated, memory)

    remaining = [issue for issue in batch_issues(targets, translated, memory) if issue.severity == "hard"]
    progress({"phase": "deterministic_recheck", "hard": len(remaining)})
    if remaining:
        raise RuntimeError(
            "Rapid final deterministic QA still has hard ids: "
            + ", ".join(sorted({issue.id for issue in remaining})[:20])
        )
    return translated


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    resume = _sanitize_resume_cache(SOURCE, CACHE)
    print("[full-reference] resume=" + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)

    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(targets)
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    state = _cached_state(SOURCE, CACHE)
    if not state or state.get("pipeline_version") != PIPELINE_VERSION:
        state = {
            "pipeline_version": PIPELINE_VERSION,
            "translations": {},
            "completed_chapters": [],
            "chapter_briefs": {},
            "polished_chapters": [],
            "qa_passed_chapters": [],
        }

    harness = build_reference_harness()
    memory = _load_or_build_memory(harness, state, chapters)
    _persist(state_path, state, dict(state.get("translations") or {}), memory)

    try:
        translated = _parallel_translate(harness, targets, state, memory)
        missing = [segment.id for segment in targets if segment.id not in translated]
        if missing:
            raise RuntimeError(
                f"Rapid translation finished with {len(missing)} missing ids: "
                + ", ".join(missing[:20])
            )

        translated = _selective_repair(harness, targets, translated, memory, state)

        save_book(document, translated, OUTPUT)
        completed_chapters = [
            name for name, chapter in chapters
            if all(segment.id in translated for segment in chapter)
        ]
        state["completed_chapters"] = completed_chapters
        state["final_quality"] = {
            "hard_issues": 0,
            "segments": len(targets),
            "model_ceiling": "deepseek/deepseek-v4.1-flash",
            "pipeline_version": PIPELINE_VERSION,
            "strategy": "rapid-parallel-v1",
        }
        _persist(state_path, state, translated, memory)
        progress({"phase": "done", "progress": 100, "completed": len(targets), "total": len(targets)})
        print(f"[full-reference] output={OUTPUT} bytes={OUTPUT.stat().st_size}", flush=True)
        print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)
        _write_latest_artifact(None)
    except Exception as exc:
        report = _write_latest_artifact(exc)
        print(
            "[full-reference] rapid_best_effort="
            + json.dumps(report, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)
        if report.get("translated_segments", 0) <= 0:
            raise


if __name__ == "__main__":
    main()
