from __future__ import annotations

import json
import math
import os
from pathlib import Path

from bookai.bulk_mt import HybridStats
from bookai.gigachat_mt import GigaChatLightningBackend, benchmark_gigachat
from bookai.hybrid_mt import choose_probe_segments, decide_route, dumps_report, routing_summary
from bookai.literary_context import SourceContextIndex, atomic_persist, locked_glossary_violations
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import PIPELINE_VERSION, _batches, _cache_path, _chapter_groups, _context_for, _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.reference_harness import build_reference_harness
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

from full_reference_translation import (
    CACHE,
    OUTPUT,
    SOURCE,
    _cached_state,
    _sanitize_resume_cache,
    _write_latest_artifact,
)
from rapid_reference_translation import (
    _batch_memory,
    _load_or_build_memory,
    _prepare_literary_context,
    _repair_hard_failures,
    _selective_literary_refinement,
    progress,
)

PROBE_REPORT = Path("hybrid-probe.json")
ROUTING_REPORT = Path("hybrid-routing.json")
SAMPLE_REPORT = Path("hybrid-sample.json")


def _base_memory() -> BookMemory:
    return apply_reference_profile(BookMemory(glossary=dict(REFERENCE_GLOSSARY_SEED)))


def _is_bulk(segment: Segment) -> bool:
    return decide_route(segment).route != "deepseek"


def _candidate_bad(segment: Segment, candidate: str, memory: BookMemory) -> bool:
    mapping = {segment.id: candidate}
    if hard_ids(batch_issues([segment], mapping, memory)):
        return True
    return bool(locked_glossary_violations([segment], mapping, REFERENCE_GLOSSARY_SEED))


def _readable_through(targets: list[Segment], translated: dict[str, str]) -> int:
    count = 0
    for segment in targets:
        if segment.id not in translated:
            break
        count += 1
    return count


def _publish_partial(
    document,
    targets: list[Segment],
    translated: dict[str, str],
    state: dict,
    memory: BookMemory,
    stats: HybridStats,
    bulk: GigaChatLightningBackend,
    *,
    phase: str = "hybrid_batch_done",
    unresolved: int = 0,
) -> None:
    """Persist state and a readable mixed-language snapshot after every accepted batch."""
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    state["hybrid_stats"] = stats.as_dict()
    state["bulk_backend"] = bulk.backend_name
    state["gigachat_usage"] = bulk.usage.as_dict()
    state["unresolved_segments"] = unresolved
    atomic_persist(state_path, state, translated, memory)
    save_book(document, translated, OUTPUT)
    completed = sum(segment.id in translated for segment in targets)
    readable = _readable_through(targets, translated)
    progress(
        {
            "phase": phase,
            "completed": completed,
            "total": len(targets),
            "progress": round(completed / max(1, len(targets)) * 100, 2),
            "readable_through": readable,
            "readable_percent": round(readable / max(1, len(targets)) * 100, 2),
            "unresolved": unresolved,
            "partial_output": str(OUTPUT),
            "gigachat_api_calls": bulk.usage.api_calls,
            "gigachat_tokens": bulk.usage.total_tokens,
            **stats.as_dict(),
        }
    )


def _translate_hybrid_batch(
    harness,
    bulk: GigaChatLightningBackend,
    source_segments: list[Segment],
    batch: list[Segment],
    memory: BookMemory,
    stats: HybridStats,
) -> tuple[dict[str, str], set[str]]:
    """Return every usable translation now; failed GigaChat ids are deferred, never fatal."""
    decisions = {segment.id: decide_route(segment) for segment in batch}
    bulk_targets = [segment for segment in batch if decisions[segment.id].route != "deepseek"]
    direct_deep = [segment for segment in batch if decisions[segment.id].route == "deepseek"]

    bulk_map, bulk_errors = bulk.translate_many(
        bulk_targets,
        memory,
        source_segments=source_segments,
    )
    qa_bad_ids = {
        segment.id
        for segment in bulk_targets
        if segment.id in bulk_map and _candidate_bad(segment, bulk_map[segment.id], memory)
    }
    error_ids = set(bulk_errors)
    accepted_bulk = {
        segment.id: bulk_map[segment.id]
        for segment in bulk_targets
        if segment.id in bulk_map and segment.id not in qa_bad_ids
    }

    stats.add(
        bulk_accepted=len(accepted_bulk),
        bulk_qa_escalated=len(qa_bad_ids),
        bulk_errors=len(error_ids),
        source_chars_bulk=sum(len(segment.text) for segment in bulk_targets),
        deep_direct=len(direct_deep),
    )

    # GigaChat transport/JSON failures are NOT paid-model escalations. They remain
    # in a retry queue and are retried as single segments later. DeepSeek is used
    # only for structurally hard prose or a translation that actually failed QA.
    deep_targets = direct_deep + [segment for segment in bulk_targets if segment.id in qa_bad_ids]
    cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "300"))
    projected = stats.deep_direct + stats.deep_escalated + max(0, len(deep_targets) - len(direct_deep))
    if cap and projected > cap:
        # Preserve the already good GigaChat work instead of killing the whole run.
        overflow = projected - cap
        if overflow > 0:
            qa_slots = max(0, len(qa_bad_ids) - overflow)
            allowed_qa = [segment for segment in deep_targets if segment not in direct_deep][:qa_slots]
            deep_targets = direct_deep + allowed_qa

    result = dict(accepted_bulk)
    if deep_targets:
        before, after = _context_for(source_segments, batch, radius=3)
        deep_map = harness.translate(
            deep_targets,
            memory,
            context_before=before,
            context_after=after,
        )
        result.update(deep_map)
        stats.add(
            deep_escalated=max(0, len(deep_targets) - len(direct_deep)),
            source_chars_deep=sum(len(segment.text) for segment in deep_targets),
        )

    # Reject only the problematic ids. Good ids from the same batch are published
    # immediately instead of being discarded because one neighbour failed.
    resolved_segments = [segment for segment in batch if segment.id in result]
    hard = set(hard_ids(batch_issues(resolved_segments, result, memory))) if resolved_segments else set()
    glossary = set(locked_glossary_violations(resolved_segments, result, REFERENCE_GLOSSARY_SEED)) if resolved_segments else set()
    rejected = hard | glossary
    for sid in rejected:
        result.pop(sid, None)

    unresolved = error_ids | rejected
    # A QA-bad GigaChat id that could not fit under the DeepSeek cap remains pending.
    unresolved.update(sid for sid in qa_bad_ids if sid not in result)
    return result, unresolved


def _run_cost_safe_sample(
    bulk: GigaChatLightningBackend,
    targets: list[Segment],
    memory: BookMemory,
    probe: dict,
) -> dict:
    candidates = [segment for segment in targets if _is_bulk(segment)]
    count = min(30, len(candidates))
    if count <= 0:
        report = {"mode": "bulk-only-diagnostic", "backend": bulk.backend_name, "segments": 0}
        SAMPLE_REPORT.write_text(dumps_report(report), "utf-8")
        return report
    step = (len(candidates) - 1) / max(1, count - 1)
    sample = [candidates[round(i * step)] for i in range(count)]
    translated, errors = bulk.translate_many(sample, memory, source_segments=targets)
    bad = [
        segment.id
        for segment in sample
        if segment.id in translated and _candidate_bad(segment, translated[segment.id], memory)
    ]
    report = {
        "mode": "bulk-only-diagnostic",
        "backend": bulk.backend_name,
        "probe": probe,
        "segments": len(sample),
        "translated": len(translated),
        "errors": errors,
        "qa_bad_count": len(bad),
        "qa_bad_ids": bad,
        "deepseek_calls": 0,
    }
    SAMPLE_REPORT.write_text(dumps_report(report), "utf-8")
    progress(
        {
            "phase": "bulk_diagnostic_done",
            "segments": len(sample),
            "translated": len(translated),
            "qa_bad": len(bad),
            "deepseek_calls": 0,
            "backend": bulk.backend_name,
        }
    )
    return report


def _retry_gigachat_failures(
    bulk: GigaChatLightningBackend,
    targets: list[Segment],
    translated: dict[str, str],
    pending_ids: set[str],
    memory: BookMemory,
    document,
    state: dict,
    stats: HybridStats,
) -> set[str]:
    by_id = {segment.id: segment for segment in targets}
    rounds = max(1, min(4, int(os.getenv("BOOKAI_GIGACHAT_RECOVERY_ROUNDS") or "3")))
    remaining = set(pending_ids)
    for round_index in range(1, rounds + 1):
        if not remaining:
            break
        next_remaining: set[str] = set()
        # Single-segment retries avoid the JSON batching failures that caused the
        # old workflow to abort at 2.47%.
        for sid in sorted(remaining, key=lambda value: int(value[1:]) if value[1:].isdigit() else value):
            segment = by_id.get(sid)
            if segment is None:
                continue
            rows, errors = bulk.translate_many([segment], memory, source_segments=targets)
            candidate = rows.get(sid)
            if candidate and not _candidate_bad(segment, candidate, memory):
                translated[sid] = candidate
                stats.add(bulk_accepted=1, source_chars_bulk=len(segment.text))
            else:
                next_remaining.add(sid)
        remaining = next_remaining
        _publish_partial(
            document,
            targets,
            translated,
            state,
            memory,
            stats,
            bulk,
            phase="gigachat_recovery_round",
            unresolved=len(remaining),
        )
        progress({"phase": "gigachat_recovery", "round": round_index, "remaining": len(remaining)})
    return remaining


def _deepseek_final_recovery(
    harness,
    targets: list[Segment],
    translated: dict[str, str],
    pending_ids: set[str],
    memory: BookMemory,
    stats: HybridStats,
) -> set[str]:
    """Rare bounded recovery after repeated single-segment GigaChat failures."""
    if not pending_ids:
        return set()
    by_id = {segment.id: segment for segment in targets}
    cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "300"))
    used = stats.deep_direct + stats.deep_escalated
    slots = max(0, cap - used) if cap else len(pending_ids)
    recover_ids = list(sorted(pending_ids))[:slots]
    still = set(pending_ids) - set(recover_ids)
    for batch_ids in [recover_ids[i:i + 8] for i in range(0, len(recover_ids), 8)]:
        batch = [by_id[sid] for sid in batch_ids if sid in by_id]
        if not batch:
            continue
        before, after = _context_for(targets, batch, radius=3)
        try:
            rows = harness.translate(batch, memory, context_before=before, context_after=after)
        except Exception:
            still.update(segment.id for segment in batch)
            continue
        for segment in batch:
            candidate = rows.get(segment.id)
            if candidate and not _candidate_bad(segment, candidate, memory):
                translated[segment.id] = candidate
                stats.add(deep_escalated=1, source_chars_deep=len(segment.text))
            else:
                still.add(segment.id)
    return still


def _run_full(
    harness,
    bulk: GigaChatLightningBackend,
    document,
    targets: list[Segment],
    chapters,
    state: dict,
) -> dict:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    memory = _load_or_build_memory(harness, state, chapters)
    memory, chapter_digests, book_synopsis = _prepare_literary_context(
        harness,
        chapters,
        state,
        memory,
    )
    context_index = SourceContextIndex(targets)
    translated: dict[str, str] = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if isinstance(text, str) and text.strip()
    }
    stats = HybridStats()
    pending_retry: set[str] = set()

    # Publish restored work immediately so a restarted translation remains readable.
    if translated:
        save_book(document, translated, OUTPUT)
        _publish_partial(document, targets, translated, state, memory, stats, bulk, phase="resume_published")

    batch_chars = max(4000, int(os.getenv("BOOKAI_HYBRID_BATCH_CHARS") or "9000"))
    batches = [
        [segment for segment in batch if segment.id not in translated]
        for batch in _batches(targets, batch_chars)
    ]
    batches = [batch for batch in batches if batch]
    progress(
        {
            "phase": "hybrid_translate",
            "completed": sum(segment.id in translated for segment in targets),
            "total": len(targets),
            "batches": len(batches),
            "bulk_backend": bulk.backend_name,
            "strategy": "progressive:GigaChat-Lightning→publish-now→retry-failures→DeepSeek-hard/reasoning-only",
        }
    )

    for batch in batches:
        batch_memory = _batch_memory(
            memory,
            batch,
            chapter_digests,
            book_synopsis,
            context_index,
        )
        accepted, unresolved = _translate_hybrid_batch(
            harness,
            bulk,
            targets,
            batch,
            batch_memory,
            stats,
        )
        translated.update(accepted)
        pending_retry.update(unresolved)
        pending_retry.difference_update(translated)
        _publish_partial(
            document,
            targets,
            translated,
            state,
            memory,
            stats,
            bulk,
            unresolved=len(pending_retry),
        )

    pending_retry = _retry_gigachat_failures(
        bulk,
        targets,
        translated,
        pending_retry,
        memory,
        document,
        state,
        stats,
    )
    if pending_retry:
        progress({"phase": "deepseek_bounded_recovery", "segments": len(pending_retry)})
        pending_retry = _deepseek_final_recovery(
            harness,
            targets,
            translated,
            pending_retry,
            memory,
            stats,
        )
        _publish_partial(
            document,
            targets,
            translated,
            state,
            memory,
            stats,
            bulk,
            phase="deepseek_recovery_done",
            unresolved=len(pending_retry),
        )

    missing = [segment.id for segment in targets if segment.id not in translated]
    if missing:
        # Keep the partial FB2 and resumable cache instead of deleting useful work.
        state["status"] = "partial"
        state["missing_ids"] = missing
        atomic_persist(state_path, state, translated, memory)
        save_book(document, translated, OUTPUT)
        raise RuntimeError(
            f"Progressive translation paused with {len(missing)} unresolved segments; "
            "partial FB2 and cache were preserved: " + ", ".join(missing[:20])
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
    save_book(document, translated, OUTPUT)

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
    save_book(document, translated, OUTPUT)

    state["completed_chapters"] = [
        name for name, chapter in chapters
        if all(segment.id in translated for segment in chapter)
    ]
    state["status"] = "complete"
    state["final_quality"] = {
        "hard_issues": 0,
        "segments": len(targets),
        "strategy": "progressive-gigachat-lightning-deepseek-v41-v2",
        "primary_mt": bulk.backend_name,
        "deepseek_role": "analysis+style+context+hard-prose+bounded-recovery+selective-refinement",
        "gigachat_role": "primary-progressive-literary-translation",
        "external_pro_judge": False,
        "hybrid_stats": stats.as_dict(),
        "gigachat_usage": bulk.usage.as_dict(),
    }
    atomic_persist(state_path, state, translated, memory)
    _write_latest_artifact(None)
    progress(
        {
            "phase": "done",
            "progress": 100,
            "completed": len(targets),
            "total": len(targets),
            "readable_through": len(targets),
            "readable_percent": 100,
            "bulk_backend": bulk.backend_name,
            "gigachat_api_calls": bulk.usage.api_calls,
            "gigachat_tokens": bulk.usage.total_tokens,
            **stats.as_dict(),
        }
    )
    return {"mode": "full", "stats": stats.as_dict(), "output": str(OUTPUT)}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(targets)
    total_chars = sum(len(segment.text) for segment in targets)

    routing = routing_summary(targets)
    if "hymt" in routing:
        routing["bulk"] = routing.pop("hymt")
    ROUTING_REPORT.write_text(dumps_report(routing), "utf-8")
    progress({"phase": "routing_ready", **routing})

    bulk = GigaChatLightningBackend()
    if not bulk.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing; refusing whole-book DeepSeek fallback")

    base_memory = _base_memory()
    probe_segments = choose_probe_segments(
        targets,
        count=max(4, int(os.getenv("BOOKAI_BULK_PROBE_SEGMENTS") or "12")),
    )
    probe = benchmark_gigachat(
        bulk,
        probe_segments,
        base_memory,
        source_segments=targets,
        total_source_chars=total_chars,
    )

    qa_translated, qa_errors = bulk.translate_many(probe_segments, base_memory, source_segments=targets)
    qa_bad = [
        segment.id
        for segment in probe_segments
        if segment.id in qa_translated and _candidate_bad(segment, qa_translated[segment.id], base_memory)
    ]
    probe["qa_bad"] = len(qa_bad)
    probe["qa_bad_ids"] = qa_bad
    probe["qa_errors"] = qa_errors
    probe["token_usage_after_qa_probe"] = bulk.usage.as_dict()
    PROBE_REPORT.write_text(dumps_report(probe), "utf-8")
    progress({"phase": "bulk_probe_done", **{k: v for k, v in probe.items() if k not in {"probe_errors", "qa_errors", "qa_bad_ids"}}})

    estimate = probe.get("estimated_full_seconds")
    max_seconds = float(os.getenv("BOOKAI_BULK_MAX_ESTIMATE_SECONDS") or "9000")
    successes = int(probe.get("probe_success") or 0)
    min_success = max(2, len(probe_segments) - 1)
    max_qa_bad = max(1, math.floor(len(probe_segments) * 0.25))
    full_allowed = (
        isinstance(estimate, (int, float))
        and float(estimate) <= max_seconds
        and successes >= min_success
        and len(qa_bad) <= max_qa_bad
    )

    if not full_allowed:
        print(
            "[bookai-hybrid] GigaChat Lightning failed health/quality gate; running zero-DeepSeek diagnostic only. "
            f"estimate={estimate!r}s qa_bad={len(qa_bad)} limit={max_qa_bad}",
            flush=True,
        )
        _run_cost_safe_sample(bulk, targets, base_memory, probe)
        return

    harness = build_reference_harness()
    resume = _sanitize_resume_cache(SOURCE, CACHE)
    print("[full-reference] resume=" + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)
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
    state["hybrid_probe"] = probe
    state["routing_summary"] = routing
    state["bulk_backend"] = bulk.backend_name
    state["architecture"] = {
        "primary_translation": "GigaChat-3-Lightning",
        "reasoning_analysis": "deepseek/deepseek-v4.1-flash",
        "hard_translation": "deepseek/deepseek-v4.1-flash",
        "literary_refinement": "deepseek/deepseek-v4.1-flash",
        "external_pro_judge": None,
        "progressive_publish": True,
        "full_paid_fallback": False,
    }
    _run_full(harness, bulk, document, targets, chapters, state)


if __name__ == "__main__":
    main()
