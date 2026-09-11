from __future__ import annotations

import json
import math
import os
from pathlib import Path

from bookai.hybrid_mt import (
    HYMTClient,
    HybridStats,
    benchmark_hymt,
    choose_probe_segments,
    decide_route,
    dumps_report,
    routing_summary,
)
from bookai.literary_context import (
    SourceContextIndex,
    atomic_persist,
    locked_glossary_violations,
)
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


def _representative_sample(targets: list[Segment], hymt_count: int = 24, deep_count: int = 6) -> list[Segment]:
    hymt = [segment for segment in targets if decide_route(segment).route == "hymt"]
    deep = [segment for segment in targets if decide_route(segment).route == "deepseek"]

    def spread(rows: list[Segment], count: int) -> list[Segment]:
        if len(rows) <= count:
            return rows
        step = (len(rows) - 1) / max(1, count - 1)
        return [rows[round(i * step)] for i in range(count)]

    selected = spread(hymt, hymt_count) + spread(deep, deep_count)
    positions = {segment.id: index for index, segment in enumerate(targets)}
    selected.sort(key=lambda segment: positions.get(segment.id, 0))
    return selected


def _candidate_bad(segment: Segment, candidate: str, memory: BookMemory) -> bool:
    mapping = {segment.id: candidate}
    if hard_ids(batch_issues([segment], mapping, memory)):
        return True
    return bool(locked_glossary_violations([segment], mapping, REFERENCE_GLOSSARY_SEED))


def _translate_hybrid_batch(
    harness,
    hymt: HYMTClient,
    source_segments: list[Segment],
    batch: list[Segment],
    memory: BookMemory,
    stats: HybridStats,
) -> dict[str, str]:
    decisions = {segment.id: decide_route(segment) for segment in batch}
    hymt_targets = [segment for segment in batch if decisions[segment.id].route == "hymt"]
    direct_deep = [segment for segment in batch if decisions[segment.id].route == "deepseek"]

    hymt_map, hymt_errors = hymt.translate_many(
        hymt_targets,
        memory,
        source_segments=source_segments,
    )
    max_errors = max(2, math.ceil(len(hymt_targets) * 0.15)) if hymt_targets else 0
    if len(hymt_errors) > max_errors:
        # Cost-safe failure mode: a broken local MT backend must never silently
        # turn the whole book back into a paid DeepSeek translation.
        raise RuntimeError(
            f"HY-MT unhealthy: {len(hymt_errors)}/{len(hymt_targets)} local requests failed"
        )

    qa_bad_ids = {
        segment.id
        for segment in hymt_targets
        if segment.id in hymt_map and _candidate_bad(segment, hymt_map[segment.id], memory)
    }
    error_ids = set(hymt_errors)
    escalated_ids = qa_bad_ids | error_ids
    accepted_hymt = {
        segment.id: hymt_map[segment.id]
        for segment in hymt_targets
        if segment.id in hymt_map and segment.id not in escalated_ids
    }

    stats.add(
        hymt_accepted=len(accepted_hymt),
        hymt_qa_escalated=len(qa_bad_ids),
        hymt_errors=len(error_ids),
        source_chars_hymt=sum(len(segment.text) for segment in hymt_targets),
        deep_direct=len(direct_deep),
    )

    deep_targets = direct_deep + [
        segment for segment in hymt_targets if segment.id in escalated_ids
    ]
    result = dict(accepted_hymt)
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


def _run_sample(
    harness,
    hymt: HYMTClient,
    targets: list[Segment],
    memory: BookMemory,
    probe: dict,
) -> dict:
    sample = _representative_sample(targets)
    stats = HybridStats()
    translated: dict[str, str] = {}
    for batch in _batches(sample, 9000):
        translated.update(_translate_hybrid_batch(harness, hymt, targets, batch, memory, stats))
    rows = []
    for segment in sample:
        decision = decide_route(segment)
        rows.append({
            "id": segment.id,
            "chapter": segment.chapter,
            "route": decision.route,
            "score": decision.score,
            "reasons": decision.reasons,
            "source_chars": len(segment.text),
            "translated": segment.id in translated,
        })
    report = {
        "mode": "cost-safe-sample",
        "probe": probe,
        "stats": stats.as_dict(),
        "sample_segments": len(sample),
        "rows": rows,
    }
    SAMPLE_REPORT.write_text(dumps_report(report), "utf-8")
    progress({"phase": "hybrid_sample_done", "segments": len(sample), **stats.as_dict()})
    return report


def _run_full(
    harness,
    hymt: HYMTClient,
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
    batch_chars = max(6000, int(os.getenv("BOOKAI_HYBRID_BATCH_CHARS") or "16000"))
    batches = [
        [segment for segment in batch if segment.id not in translated]
        for batch in _batches(targets, batch_chars)
    ]
    batches = [batch for batch in batches if batch]
    progress({
        "phase": "hybrid_translate",
        "completed": sum(segment.id in translated for segment in targets),
        "total": len(targets),
        "batches": len(batches),
        "strategy": "HY-MT-local→deterministic-QA→DeepSeek-hard-or-failed-only",
    })

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
            hymt,
            targets,
            batch,
            batch_memory,
            stats,
        )
        translated.update(accepted)
        state["hybrid_stats"] = stats.as_dict()
        atomic_persist(state_path, state, translated, memory)
        completed = sum(segment.id in translated for segment in targets)
        progress({
            "phase": "hybrid_batch_done",
            "completed": completed,
            "total": len(targets),
            "progress": round(completed / max(1, len(targets)) * 100, 2),
            **stats.as_dict(),
        })

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
        "strategy": "hybrid-hymt2bit-deepseek-v1",
        "primary_mt": os.getenv("BOOKAI_HYMT_REPO") or "tencent/Hy-MT1.5-1.8B-2bit-GGUF",
        "deepseek_role": "analysis+hard-routing+qa-escalation+selective-literary-refinement",
        "hybrid_stats": stats.as_dict(),
    }
    atomic_persist(state_path, state, translated, memory)
    _write_latest_artifact(None)
    progress({"phase": "done", "progress": 100, "completed": len(targets), "total": len(targets), **stats.as_dict()})
    return {"mode": "full", "stats": stats.as_dict(), "output": str(OUTPUT)}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(targets)
    total_chars = sum(len(segment.text) for segment in targets)
    routing = routing_summary(targets)
    ROUTING_REPORT.write_text(dumps_report(routing), "utf-8")
    progress({"phase": "routing_ready", **routing})

    hymt = HYMTClient()
    if not hymt.healthy():
        raise RuntimeError("HY-MT local server is unavailable; refusing paid full-book fallback")

    base_memory = _base_memory()
    probe_segments = choose_probe_segments(
        targets,
        count=max(4, int(os.getenv("BOOKAI_HYMT_PROBE_SEGMENTS") or "10")),
    )
    probe = benchmark_hymt(
        hymt,
        probe_segments,
        base_memory,
        source_segments=targets,
        total_source_chars=total_chars,
    )
    PROBE_REPORT.write_text(dumps_report(probe), "utf-8")
    progress({"phase": "hymt_probe_done", **{k: v for k, v in probe.items() if k != "probe_errors"}})

    estimate = probe.get("estimated_full_seconds")
    max_seconds = float(os.getenv("BOOKAI_HYMT_MAX_ESTIMATE_SECONDS") or "1500")
    force_full = os.getenv("BOOKAI_HYMT_FORCE_FULL", "false").lower() in {"1", "true", "yes"}
    full_allowed = force_full or (
        isinstance(estimate, (int, float))
        and probe.get("probe_success", 0) >= max(2, len(probe_segments) - 1)
        and float(estimate) <= max_seconds
    )

    harness = build_reference_harness()
    if not full_allowed:
        print(
            "[bookai-hybrid] CPU/GPU probe predicts a slow full run; executing cost-safe sample only. "
            f"estimate={estimate!r}s limit={max_seconds}s",
            flush=True,
        )
        _run_sample(harness, hymt, targets, base_memory, probe)
        return

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
    _run_full(harness, hymt, document, targets, chapters, state)


if __name__ == "__main__":
    main()
