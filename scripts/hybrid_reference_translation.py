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


def _translate_hybrid_batch(
    harness,
    bulk: GigaChatLightningBackend,
    source_segments: list[Segment],
    batch: list[Segment],
    memory: BookMemory,
    stats: HybridStats,
) -> dict[str, str]:
    decisions = {segment.id: decide_route(segment) for segment in batch}
    bulk_targets = [segment for segment in batch if decisions[segment.id].route != "deepseek"]
    direct_deep = [segment for segment in batch if decisions[segment.id].route == "deepseek"]

    bulk_map, bulk_errors = bulk.translate_many(
        bulk_targets,
        memory,
        source_segments=source_segments,
    )
    max_errors = max(2, math.ceil(len(bulk_targets) * 0.15)) if bulk_targets else 0
    if len(bulk_errors) > max_errors:
        raise RuntimeError(
            f"GigaChat bulk translation unhealthy: {len(bulk_errors)}/{len(bulk_targets)} segments failed; "
            "refusing silent whole-book DeepSeek fallback"
        )

    qa_bad_ids = {
        segment.id
        for segment in bulk_targets
        if segment.id in bulk_map and _candidate_bad(segment, bulk_map[segment.id], memory)
    }
    error_ids = set(bulk_errors)
    escalated_ids = qa_bad_ids | error_ids
    accepted_bulk = {
        segment.id: bulk_map[segment.id]
        for segment in bulk_targets
        if segment.id in bulk_map and segment.id not in escalated_ids
    }

    stats.add(
        bulk_accepted=len(accepted_bulk),
        bulk_qa_escalated=len(qa_bad_ids),
        bulk_errors=len(error_ids),
        source_chars_bulk=sum(len(segment.text) for segment in bulk_targets),
        deep_direct=len(direct_deep),
    )

    deep_targets = direct_deep + [
        segment for segment in bulk_targets if segment.id in escalated_ids
    ]
    cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "300"))
    projected = stats.deep_direct + stats.deep_escalated + max(0, len(deep_targets) - len(direct_deep))
    if cap and projected > cap:
        raise RuntimeError(
            f"DeepSeek segment cap would be exceeded: projected={projected} cap={cap}; "
            "refusing silent whole-book paid fallback"
        )

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
            deep_escalated=len(deep_targets) - len(direct_deep),
            source_chars_deep=sum(len(segment.text) for segment in deep_targets),
        )

    hard = hard_ids(batch_issues(batch, result, memory))
    glossary = locked_glossary_violations(batch, result, REFERENCE_GLOSSARY_SEED)
    missing = [segment.id for segment in batch if segment.id not in result]
    if hard or glossary or missing:
        problem = sorted(set(hard) | set(glossary) | set(missing))
        raise ValueError("hybrid batch final QA failed: " + ", ".join(problem[:20]))
    return result


def _run_cost_safe_sample(
    bulk: GigaChatLightningBackend,
    targets: list[Segment],
    memory: BookMemory,
    probe: dict,
) -> dict:
    """Diagnostic fallback with zero DeepSeek calls."""
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


def _run_full(
    harness,
    bulk: GigaChatLightningBackend,
    document,
    targets: list[Segment],
    chapters,
    state: dict,
) -> dict:
    state_path = _cache_path(SOURCE, CACHE, "optimal")

    # DeepSeek V4.1 Flash is deliberately used for book-level reasoning before
    # GigaChat sees the bulk prose: analysis, abstract style card, chapter digests,
    # compact synopsis, glossary/context preparation and later selective refinement.
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
    batch_chars = max(6000, int(os.getenv("BOOKAI_HYBRID_BATCH_CHARS") or "16000"))
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
            "strategy": "GigaChat-3-Lightning→deterministic-QA→DeepSeek-V4.1-Flash-hard-or-failed-only",
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
        accepted = _translate_hybrid_batch(
            harness,
            bulk,
            targets,
            batch,
            batch_memory,
            stats,
        )
        translated.update(accepted)
        state["hybrid_stats"] = stats.as_dict()
        state["bulk_backend"] = bulk.backend_name
        state["gigachat_usage"] = bulk.usage.as_dict()
        atomic_persist(state_path, state, translated, memory)
        completed = sum(segment.id in translated for segment in targets)
        progress(
            {
                "phase": "hybrid_batch_done",
                "completed": completed,
                "total": len(targets),
                "progress": round(completed / max(1, len(targets)) * 100, 2),
                "gigachat_api_calls": bulk.usage.api_calls,
                "gigachat_tokens": bulk.usage.total_tokens,
                **stats.as_dict(),
            }
        )

    missing = [segment.id for segment in targets if segment.id not in translated]
    if missing:
        raise RuntimeError(f"hybrid translation left {len(missing)} missing ids: " + ", ".join(missing[:20]))

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

    save_book(document, translated, OUTPUT)
    state["completed_chapters"] = [
        name for name, chapter in chapters
        if all(segment.id in translated for segment in chapter)
    ]
    state["final_quality"] = {
        "hard_issues": 0,
        "segments": len(targets),
        "strategy": "gigachat-lightning-deepseek-v41-hybrid-v1",
        "primary_mt": bulk.backend_name,
        "deepseek_role": "book-analysis+style-card+synopsis+chapter-digests+hard-routing+qa-escalation+bounded-literary-refinement",
        "gigachat_role": "primary-literary-draft",
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
        "full_paid_fallback": False,
    }
    _run_full(harness, bulk, document, targets, chapters, state)


if __name__ == "__main__":
    main()
