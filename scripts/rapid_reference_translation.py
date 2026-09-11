from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from bookai.literary_context import (
    SourceContextIndex,
    apply_style_card,
    atomic_persist,
    build_book_synopsis,
    ensure_chapter_digests,
    extract_style_card,
    locked_glossary_violations,
    memory_with_context,
    select_refinement_targets,
)
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import (
    PIPELINE_VERSION,
    _analysis_sample,
    _batches,
    _cache_path,
    _chapter_groups,
    _context_for,
    _memory_from_dict,
    _should_translate,
    _translated_context,
)
from bookai.quality import batch_issues, hard_ids
from bookai.reference_harness import build_reference_harness
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED
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
            glossary_failures = locked_glossary_violations(batch, candidate, REFERENCE_GLOSSARY_SEED)
            if glossary_failures:
                detail = "; ".join(
                    f"{sid}:{','.join(terms)}" for sid, terms in sorted(glossary_failures.items())
                )
                raise ValueError("locked glossary violation: " + detail)
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


def _prepare_literary_context(harness, chapters, state, memory):
    style_card = state.get("style_card")
    if not isinstance(style_card, dict) or not style_card.get("rules"):
        style_sample_chars = max(6000, min(16000, int(os.getenv("BOOKAI_STYLE_SAMPLE_CHARS") or "12000")))
        style_card = extract_style_card(
            harness.analyzer,
            _analysis_sample(chapters, style_sample_chars),
            memory,
        )
        state["style_card"] = style_card
    memory = apply_style_card(memory, style_card)

    existing_digests = dict(state.get("chapter_briefs") or {})
    chapter_digests = ensure_chapter_digests(harness, chapters, memory, existing_digests)
    state["chapter_briefs"] = chapter_digests

    book_synopsis = build_book_synopsis(
        harness.analyzer,
        chapter_digests,
        str(state.get("book_synopsis") or ""),
    )
    state["book_synopsis"] = book_synopsis
    state["glossary_status"] = {
        source: {"translation": target, "status": "locked"}
        for source, target in REFERENCE_GLOSSARY_SEED.items()
    }
    state["context_strategy"] = {
        "style_card": "abstract-source-style",
        "chapter_digests": "cached-parallel-prescan",
        "book_synopsis": "compact-continuity",
        "retrieval": "idf-distant-source-k2",
        "locked_glossary": True,
        "refinement": "bounded-risk-selected-segment-level",
    }
    progress(
        {
            "phase": "context_ready",
            "style_rules": len(style_card.get("rules") or []),
            "style_rules_rejected": len(style_card.get("rejected") or []),
            "chapter_digests": len(chapter_digests),
            "synopsis_chars": len(book_synopsis),
            "locked_terms": len(REFERENCE_GLOSSARY_SEED),
        }
    )
    return memory, chapter_digests, book_synopsis


def _batch_memory(memory, batch, chapter_digests, book_synopsis, context_index):
    distant = context_index.retrieve(batch)
    chapter_name = batch[0].chapter if batch else ""
    return memory_with_context(
        memory,
        book_synopsis=book_synopsis,
        chapter_digest=chapter_digests.get(chapter_name, ""),
        distant=distant,
    )


def _parallel_translate(
    harness,
    targets,
    state,
    memory,
    *,
    chapter_digests,
    book_synopsis,
    context_index,
) -> dict[str, str]:
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
            "strategy": "contextual-parallel-draft→deterministic-QA→locked-glossary→local-split-retry",
        }
    )

    failures: list[tuple[list, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v41") as pool:
        future_to_batch = {}
        for batch in batches:
            batch_memory = _batch_memory(
                memory,
                batch,
                chapter_digests,
                book_synopsis,
                context_index,
            )
            future = pool.submit(_translate_resilient, harness, targets, batch, batch_memory)
            future_to_batch[future] = batch

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
            atomic_persist(state_path, state, translated, memory)
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
        batch_memory = _batch_memory(memory, batch, chapter_digests, book_synopsis, context_index)
        try:
            accepted = _translate_resilient(harness, targets, batch, batch_memory, depth=1)
        except BaseException as exc:
            atomic_persist(state_path, state, translated, memory)
            raise RuntimeError(
                f"Rapid translation could not recover batch {batch[0].id}..{batch[-1].id}"
            ) from exc
        translated.update(accepted)
        atomic_persist(state_path, state, translated, memory)
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


def _repair_hard_failures(
    harness,
    targets,
    translated,
    memory,
    state,
    *,
    chapter_digests,
    book_synopsis,
    context_index,
) -> dict[str, str]:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    issues = batch_issues(targets, translated, memory)
    glossary = locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
    hard_ids_set = {issue.id for issue in issues if issue.severity == "hard"} | set(glossary)
    if not hard_ids_set:
        progress({"phase": "deterministic_qa", "hard": 0, "locked_glossary": 0})
        return translated

    repair_targets = [segment for segment in targets if segment.id in hard_ids_set]
    progress(
        {
            "phase": "selective_repair",
            "hard": len(repair_targets),
            "locked_glossary": len(glossary),
        }
    )
    for batch in _batches(repair_targets, 12000):
        batch_memory = _batch_memory(memory, batch, chapter_digests, book_synopsis, context_index)
        accepted = _translate_resilient(harness, targets, batch, batch_memory, depth=1)
        translated.update(accepted)
        atomic_persist(state_path, state, translated, memory)

    remaining_hard = {
        issue.id for issue in batch_issues(targets, translated, memory) if issue.severity == "hard"
    }
    remaining_glossary = set(
        locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
    )
    remaining = remaining_hard | remaining_glossary
    progress(
        {
            "phase": "deterministic_recheck",
            "hard": len(remaining_hard),
            "locked_glossary": len(remaining_glossary),
        }
    )
    if remaining:
        raise RuntimeError(
            "Rapid final deterministic QA still has hard ids: "
            + ", ".join(sorted(remaining)[:20])
        )
    return translated


def _refine_batch(
    harness,
    targets,
    batch,
    translated_snapshot,
    memory,
    chapter_digests,
    book_synopsis,
    context_index,
):
    batch_memory = _batch_memory(memory, batch, chapter_digests, book_synopsis, context_index)
    context = _translated_context(targets, batch, translated_snapshot, radius=2)
    polished = harness.polish(batch, translated_snapshot, batch_memory, context=context)
    hard = hard_ids(batch_issues(batch, polished, batch_memory))
    glossary = locked_glossary_violations(batch, polished, REFERENCE_GLOSSARY_SEED)
    if hard or glossary:
        raise ValueError(
            "refinement rejected: "
            + ",".join(sorted(set(hard) | set(glossary))[:20])
        )
    return polished


def _selective_literary_refinement(
    harness,
    targets,
    translated,
    memory,
    state,
    *,
    chapter_digests,
    book_synopsis,
    context_index,
) -> dict[str, str]:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    glossary = locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
    selected = select_refinement_targets(
        targets,
        translated,
        locked_violations=glossary,
    )
    if not selected:
        progress({"phase": "literary_refinement", "selected": 0, "accepted": 0})
        return translated

    batch_chars = max(4000, int(os.getenv("BOOKAI_RAPID_REFINE_BATCH_CHARS") or "8000"))
    workers = max(2, min(6, int(os.getenv("BOOKAI_RAPID_REFINE_WORKERS") or "4")))
    batches = list(_batches(selected, batch_chars))
    snapshot = dict(translated)
    progress(
        {
            "phase": "literary_refinement",
            "selected": len(selected),
            "batches": len(batches),
            "workers": workers,
            "strategy": "simple-segment-level-refinement-on-risk-selected-prose",
        }
    )

    accepted_total = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-refine") as pool:
        future_to_batch = {
            pool.submit(
                _refine_batch,
                harness,
                targets,
                batch,
                snapshot,
                memory,
                chapter_digests,
                book_synopsis,
                context_index,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                refined = future.result()
            except BaseException as exc:
                print(
                    "[bookai-refine] skipped="
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
            translated.update(refined)
            accepted_total += len(refined)
            atomic_persist(state_path, state, translated, memory)
            progress(
                {
                    "phase": "literary_refine_batch_done",
                    "accepted_total": accepted_total,
                    "selected": len(selected),
                    "first": batch[0].id,
                    "last": batch[-1].id,
                }
            )

    progress(
        {
            "phase": "literary_refinement_done",
            "selected": len(selected),
            "accepted": accepted_total,
        }
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
    memory, chapter_digests, book_synopsis = _prepare_literary_context(
        harness,
        chapters,
        state,
        memory,
    )
    atomic_persist(state_path, state, dict(state.get("translations") or {}), memory)
    context_index = SourceContextIndex(targets)

    try:
        translated = _parallel_translate(
            harness,
            targets,
            state,
            memory,
            chapter_digests=chapter_digests,
            book_synopsis=book_synopsis,
            context_index=context_index,
        )
        missing = [segment.id for segment in targets if segment.id not in translated]
        if missing:
            raise RuntimeError(
                f"Rapid translation finished with {len(missing)} missing ids: "
                + ", ".join(missing[:20])
            )

        translated = _repair_hard_failures(
            harness,
            targets,
            translated,
            memory,
            state,
            chapter_digests=chapter_digests,
            book_synopsis=book_synopsis,
            context_index=context_index,
        )
        translated = _selective_literary_refinement(
            harness,
            targets,
            translated,
            memory,
            state,
            chapter_digests=chapter_digests,
            book_synopsis=book_synopsis,
            context_index=context_index,
        )

        final_hard = {
            issue.id for issue in batch_issues(targets, translated, memory)
            if issue.severity == "hard"
        }
        final_glossary = set(
            locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
        )
        if final_hard or final_glossary:
            raise RuntimeError(
                "Final rapid-v2 validation failed for: "
                + ", ".join(sorted(final_hard | final_glossary)[:20])
            )

        save_book(document, translated, OUTPUT)
        completed_chapters = [
            name for name, chapter in chapters
            if all(segment.id in translated for segment in chapter)
        ]
        state["completed_chapters"] = completed_chapters
        state["qa_passed_chapters"] = completed_chapters
        state["final_quality"] = {
            "hard_issues": 0,
            "locked_glossary_issues": 0,
            "segments": len(targets),
            "model_ceiling": "deepseek/deepseek-v4.1-flash",
            "pipeline_version": PIPELINE_VERSION,
            "strategy": "rapid-contextual-literary-v2",
            "style_rules": len((state.get("style_card") or {}).get("rules") or []),
            "chapter_digests": len(state.get("chapter_briefs") or {}),
            "retrieval": state.get("context_strategy", {}).get("retrieval"),
        }
        atomic_persist(state_path, state, translated, memory)
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
