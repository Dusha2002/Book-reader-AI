from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import full_reference_translation as fullref
import hybrid_reference_translation as hybrid
from bookai.bulk_mt import HybridStats
from bookai.gigachat_v3 import GigaChatLightningV3Backend
from bookai.hybrid_mt import choose_probe_segments, dumps_report, routing_summary
from bookai.literary_context import SourceContextIndex, atomic_persist, locked_glossary_violations
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import PIPELINE_VERSION, _batches, _cache_path, _chapter_groups, _context_for, _should_translate
from bookai.quality import hard_ids
from bookai.quality_v3 import (
    enhanced_batch_issues,
    enhanced_candidate_issues,
    infer_active_speaker,
    is_short_semantic_risk,
)
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED

SOURCE = Path("Devices_and_Desires.fb2")
CHAPTER_NAME = os.getenv("BOOKAI_CHAPTER_NAME") or "Chapter Twelve"
CHAPTER_SLUG = re.sub(r"[^a-z0-9]+", "-", CHAPTER_NAME.casefold()).strip("-") or "chapter"
OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V3.fb2")
CACHE = Path(f".bookai-cache-chapter-eval-v3-{CHAPTER_SLUG}")
PROGRESS = Path("chapter-v3-progress.json")
PROBE = Path("chapter-v3-probe.json")
ROUTING = Path("chapter-v3-routing.json")
REPORT = Path("chapter-v3.json")
SOURCE_TXT = Path("chapter-v3-source.txt")
TRANSLATED_TXT = Path("chapter-v3-translated.txt")
MAP_JSON = Path("chapter-v3-translation-map.json")


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _configure_modules() -> None:
    hybrid.SOURCE = SOURCE
    hybrid.OUTPUT = OUTPUT
    hybrid.CACHE = CACHE
    hybrid.PROBE_REPORT = PROBE
    hybrid.ROUTING_REPORT = ROUTING
    fullref.SOURCE = SOURCE
    fullref.OUTPUT = OUTPUT
    fullref.CACHE = CACHE
    fullref.PROGRESS_REPORT = PROGRESS


def _select_chapter(document) -> tuple[str, list]:
    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    wanted = _norm(CHAPTER_NAME)
    exact = [(name, rows) for name, rows in groups if _norm(name) == wanted]
    if not exact:
        exact = [(name, rows) for name, rows in groups if wanted in _norm(name)]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected one chapter matching {wanted!r}; found {len(exact)}. "
            f"Available={[name for name, _ in groups]}"
        )
    return exact[0]


def _candidate_bad_v3(segment, candidate: str, memory, source_segments: list) -> bool:
    issues = enhanced_candidate_issues(
        segment,
        candidate,
        memory,
        source_segments=source_segments,
    )
    if hard_ids(issues):
        return True
    return bool(
        locked_glossary_violations(
            [segment],
            {segment.id: candidate},
            REFERENCE_GLOSSARY_SEED,
        )
    )


def _analysis_task(harness, chapters, state_snapshot: dict):
    started = time.perf_counter()
    memory = hybrid._load_or_build_memory(harness, state_snapshot, chapters)
    memory, chapter_digests, book_synopsis = hybrid._prepare_literary_context(
        harness,
        chapters,
        state_snapshot,
        memory,
    )
    elapsed = time.perf_counter() - started
    print(
        "[v3-analysis] done "
        + json.dumps(
            {
                "elapsed_seconds": round(elapsed, 2),
                "synopsis_chars": len(book_synopsis),
                "chapter_digests": len(chapter_digests),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return memory, chapter_digests, book_synopsis, state_snapshot, elapsed


def _merge_analysis_state(state: dict, analysis_state: dict) -> None:
    protected = {"translations", "hybrid_stats", "gigachat_usage", "unresolved_segments", "status"}
    for key, value in analysis_state.items():
        if key not in protected:
            state[key] = value


def _speaker_memory(memory, segment, targets):
    speaker = infer_active_speaker(segment, targets, memory)
    if not speaker:
        return memory
    source_name, gender, ru_name = speaker
    form = "женский" if gender == "female" else "мужской"
    extra = (
        f"\nCURRENT FIRST-PERSON SPEAKER: {source_name}/{ru_name}; gender={gender}. "
        f"When Russian first-person past tense/adjectives mark gender, use {form} forms."
    )
    return replace(memory, rolling_summary=(memory.rolling_summary + extra)[-7000:])


def _semantic_short_repair(harness, targets, translated, memory) -> dict:
    by_id = {row.id: row for row in targets}
    hard_now = {
        issue.id
        for issue in enhanced_batch_issues(
            targets,
            translated,
            memory,
            source_segments=targets,
        )
        if issue.severity == "hard"
    }
    short_ids = [row.id for row in targets if is_short_semantic_risk(row)]
    max_short = max(0, int(os.getenv("BOOKAI_SHORT_SEMANTIC_MAX") or "48"))
    selected_ids = list(dict.fromkeys([*sorted(hard_now), *short_ids[:max_short]]))
    cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "80"))
    selected_ids = selected_ids[:cap or None]
    if not selected_ids:
        return {"selected": 0, "accepted": 0, "remaining_hard": 0}

    batches = [selected_ids[i : i + 8] for i in range(0, len(selected_ids), 8)]
    workers = max(1, min(4, int(os.getenv("BOOKAI_SHORT_SEMANTIC_WORKERS") or "4")))
    accepted = 0

    def run(batch_ids: list[str]):
        batch = [by_id[sid] for sid in batch_ids]
        local_memory = memory
        # If the whole batch belongs to a known letter writer, preserve that first-person gender.
        if batch:
            local_memory = _speaker_memory(local_memory, batch[0], targets)
        before, after = _context_for(targets, batch, radius=5)
        rows = harness.translate(batch, local_memory, context_before=before, context_after=after)
        return batch, rows

    print(
        "[v3-semantic] start "
        + json.dumps({"selected": len(selected_ids), "hard": len(hard_now), "short": len(short_ids), "workers": workers}),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, ids) for ids in batches]
        for future in as_completed(futures):
            batch, rows = future.result()
            for segment in batch:
                candidate = rows.get(segment.id)
                if not candidate:
                    continue
                issues = enhanced_candidate_issues(
                    segment,
                    candidate,
                    _speaker_memory(memory, segment, targets),
                    source_segments=targets,
                )
                if not any(issue.severity == "hard" for issue in issues):
                    translated[segment.id] = candidate
                    accepted += 1

    remaining_hard = {
        issue.id
        for issue in enhanced_batch_issues(
            targets,
            translated,
            memory,
            source_segments=targets,
        )
        if issue.severity == "hard"
    }
    print(
        "[v3-semantic] done "
        + json.dumps({"selected": len(selected_ids), "accepted": accepted, "remaining_hard": len(remaining_hard)}),
        flush=True,
    )
    return {"selected": len(selected_ids), "accepted": accepted, "remaining_hard": len(remaining_hard)}


def _run_full_v3(harness, bulk, document, targets, chapters, state: dict) -> dict:
    state_path = _cache_path(SOURCE, CACHE, "optimal")
    base_memory = hybrid._base_memory()
    context_index = SourceContextIndex(targets)
    translated = {
        str(sid): text
        for sid, text in dict(state.get("translations") or {}).items()
        if isinstance(text, str) and text.strip()
    }
    stats = HybridStats()
    pending_retry: set[str] = set()
    analysis_state = copy.deepcopy(state)

    # Make v3 QA authoritative for the immediate GigaChat acceptance gate.
    original_candidate_bad = hybrid._candidate_bad
    hybrid._candidate_bad = lambda segment, candidate, memory: _candidate_bad_v3(
        segment, candidate, memory, targets
    )

    analysis_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as analysis_pool:
        analysis_future = analysis_pool.submit(_analysis_task, harness, chapters, analysis_state)
        rich_memory = None
        chapter_digests: dict = {}
        book_synopsis = ""
        analysis_elapsed = None

        if translated:
            save_book(document, translated, OUTPUT)
            hybrid._publish_partial(document, targets, translated, state, base_memory, stats, bulk, phase="resume_published")

        batch_chars = max(4000, int(os.getenv("BOOKAI_HYBRID_BATCH_CHARS") or "9000"))
        batches = [[s for s in batch if s.id not in translated] for batch in _batches(targets, batch_chars)]
        batches = [batch for batch in batches if batch]
        hybrid.progress(
            {
                "phase": "v3_progressive_translate",
                "completed": len(translated),
                "total": len(targets),
                "batches": len(batches),
                "strategy": "GigaChat-now || DeepSeek-analysis-in-parallel → semantic-short-repair → selective-refinement",
            }
        )

        for index, batch in enumerate(batches, 1):
            if rich_memory is None and analysis_future.done():
                rich_memory, chapter_digests, book_synopsis, completed_state, analysis_elapsed = analysis_future.result()
                _merge_analysis_state(state, completed_state)
                print(f"[v3-analysis] activated_at_batch={index}/{len(batches)}", flush=True)

            active_memory = rich_memory or base_memory
            if rich_memory is not None:
                batch_memory = hybrid._batch_memory(
                    active_memory,
                    batch,
                    chapter_digests,
                    book_synopsis,
                    context_index,
                )
            else:
                batch_memory = active_memory

            accepted, unresolved = hybrid._translate_hybrid_batch(
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
            hybrid._publish_partial(
                document,
                targets,
                translated,
                state,
                active_memory,
                stats,
                bulk,
                phase="v3_batch_done",
                unresolved=len(pending_retry),
            )

        draft_done = time.perf_counter()
        print(
            "[v3-draft] "
            + json.dumps(
                {
                    "elapsed_seconds": round(draft_done - analysis_started, 2),
                    "completed": len(translated),
                    "total": len(targets),
                    "analysis_ready": analysis_future.done(),
                }
            ),
            flush=True,
        )

        if rich_memory is None:
            rich_memory, chapter_digests, book_synopsis, completed_state, analysis_elapsed = analysis_future.result()
            _merge_analysis_state(state, completed_state)
        memory = rich_memory

    try:
        pending_retry = hybrid._retry_gigachat_failures(
            bulk, targets, translated, pending_retry, memory, document, state, stats
        )
        if pending_retry:
            pending_retry = hybrid._deepseek_final_recovery(
                harness, targets, translated, pending_retry, memory, stats
            )

        missing = [segment.id for segment in targets if segment.id not in translated]
        if missing:
            state["status"] = "partial"
            state["missing_ids"] = missing
            atomic_persist(state_path, state, translated, memory)
            save_book(document, translated, OUTPUT)
            raise RuntimeError(f"v3 chapter has {len(missing)} unresolved segments: {missing[:12]}")

        semantic_stats = _semantic_short_repair(harness, targets, translated, memory)
        save_book(document, translated, OUTPUT)

        # Keep the existing proven hard-failure repair as a second safety net.
        translated = hybrid._repair_hard_failures(
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

        # v3 intentionally keeps this bounded: semantic repair already touched the
        # riskiest short lines, so don't spend minutes rewriting clean prose.
        translated = hybrid._selective_literary_refinement(
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

        final_issues = enhanced_batch_issues(
            targets,
            translated,
            memory,
            source_segments=targets,
        )
        final_hard = [issue for issue in final_issues if issue.severity == "hard"]
        state["status"] = "complete" if not final_hard else "needs_review"
        state["v3_semantic_stats"] = semantic_stats
        state["v3_final_issues"] = [issue.__dict__ for issue in final_issues]
        state["completed_chapters"] = [CHAPTER_NAME]
        state["final_quality"] = {
            "hard_issues": len(final_hard),
            "segments": len(targets),
            "strategy": "progressive-gigachat-lightning-v3-parallel-analysis",
            "primary_mt": bulk.backend_name,
            "deepseek_role": "parallel-analysis+semantic-short-repair+hard-recovery+bounded-refinement",
            "gigachat_usage": bulk.usage.as_dict(),
            "hybrid_stats": stats.as_dict(),
            "semantic_stats": semantic_stats,
            "analysis_elapsed_seconds": round(float(analysis_elapsed or 0), 2),
        }
        atomic_persist(state_path, state, translated, memory)
        hybrid.progress(
            {
                "phase": "done",
                "progress": 100,
                "completed": len(targets),
                "total": len(targets),
                "hard_issues": len(final_hard),
                "gigachat_api_calls": bulk.usage.api_calls,
                "gigachat_tokens": bulk.usage.total_tokens,
                **stats.as_dict(),
            }
        )
        return {
            "mode": "v3",
            "stats": stats.as_dict(),
            "semantic": semantic_stats,
            "hard_issues": len(final_hard),
            "output": str(OUTPUT),
        }
    finally:
        hybrid._candidate_bad = original_candidate_bad


def _write_exports(chapter_name: str, targets: list, state: dict, memory, *, status: str, extra: dict | None = None) -> None:
    translations = {
        str(k): str(v)
        for k, v in dict(state.get("translations") or {}).items()
        if isinstance(v, str) and v.strip()
    }
    SOURCE_TXT.write_text("\n\n".join(segment.text for segment in targets), "utf-8")
    TRANSLATED_TXT.write_text(
        "\n\n".join(translations.get(segment.id, f"[UNTRANSLATED {segment.id}]\n{segment.text}") for segment in targets),
        "utf-8",
    )
    mapping = [
        {
            "id": segment.id,
            "chapter": segment.chapter,
            "source": segment.text,
            "translation": translations.get(segment.id),
            "length_ratio": round(len(translations.get(segment.id, "")) / max(1, len(segment.text)), 3),
        }
        for segment in targets
    ]
    MAP_JSON.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), "utf-8")
    completed = sum(segment.id in translations for segment in targets)
    issues = enhanced_batch_issues(targets, translations, memory, source_segments=targets)
    issue_counts = Counter(f"{issue.severity}:{issue.code}" for issue in issues)
    ratios = [row for row in mapping if row["translation"]]
    ratios.sort(key=lambda row: abs(row["length_ratio"] - 0.78), reverse=True)
    report = {
        "chapter": chapter_name,
        "status": status,
        "segments": len(targets),
        "completed_segments": completed,
        "completion_percent": round(completed / max(1, len(targets)) * 100, 2),
        "source_chars": sum(len(segment.text) for segment in targets),
        "translated_chars": sum(len(translations.get(segment.id, "")) for segment in targets),
        "overall_length_ratio": round(sum(len(translations.get(s.id, "")) for s in targets) / max(1, sum(len(s.text) for s in targets)), 3),
        "architecture": {
            "primary": "GigaChat-3-Lightning-v3 strict-json-schema",
            "parallel_reasoning_analysis": "deepseek/deepseek-v4.1-flash",
            "semantic_short_repair": "deepseek/deepseek-v4.1-flash",
            "external_pro_judge": False,
            "full_paid_fallback": False,
        },
        "state_status": state.get("status"),
        "hybrid_stats": state.get("hybrid_stats") or {},
        "gigachat_usage": state.get("gigachat_usage") or {},
        "final_quality": state.get("final_quality") or {},
        "v3_semantic_stats": state.get("v3_semantic_stats") or {},
        "post_export_issue_counts": dict(issue_counts),
        "largest_length_outliers": [
            {"id": row["id"], "ratio": row["length_ratio"], "source": row["source"][:160], "translation": (row["translation"] or "")[:220]}
            for row in ratios[:12]
        ],
        "extra": extra or {},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_modules()
    document = load_book(SOURCE)
    chapter_name, targets = _select_chapter(document)
    chapters = [(chapter_name, targets)]
    total_chars = sum(len(segment.text) for segment in targets)
    print(
        "[chapter-v3] "
        + json.dumps(
            {
                "chapter": chapter_name,
                "segments": len(targets),
                "source_chars": total_chars,
                "first_id": targets[0].id if targets else None,
                "last_id": targets[-1].id if targets else None,
                "cache": str(CACHE),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    routing = routing_summary(targets)
    ROUTING.write_text(dumps_report(routing), "utf-8")
    bulk = GigaChatLightningV3Backend()
    if not bulk.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing")

    base_memory = hybrid._base_memory()
    probe_segments = choose_probe_segments(
        targets,
        count=max(4, min(len(targets), int(os.getenv("BOOKAI_BULK_PROBE_SEGMENTS") or "8"))),
    )
    started = time.perf_counter()
    probe_map, probe_errors = bulk.translate_many(probe_segments, base_memory, source_segments=targets)
    probe_elapsed = time.perf_counter() - started
    qa_bad = [
        segment.id for segment in probe_segments
        if segment.id in probe_map and _candidate_bad_v3(segment, probe_map[segment.id], base_memory, targets)
    ]
    probe = {
        "segments": len(probe_segments),
        "success": len(probe_map),
        "errors": probe_errors,
        "qa_bad": qa_bad,
        "elapsed_seconds": round(probe_elapsed, 3),
        "usage": bulk.usage.as_dict(),
    }
    PROBE.write_text(dumps_report(probe), "utf-8")
    print("[chapter-v3-probe] " + json.dumps(probe, ensure_ascii=False), flush=True)
    if len(probe_map) < max(2, len(probe_segments) - 1) or len(qa_bad) > max(2, math.floor(len(probe_segments) * 0.5)):
        raise RuntimeError(f"v3 probe failed: {probe}")

    harness = hybrid.build_reference_harness()
    resume = hybrid._sanitize_resume_cache(SOURCE, CACHE)
    print("[chapter-v3-resume] " + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)
    state = hybrid._cached_state(SOURCE, CACHE)
    if not state or state.get("pipeline_version") != PIPELINE_VERSION:
        state = {
            "pipeline_version": PIPELINE_VERSION,
            "translations": {},
            "completed_chapters": [],
            "chapter_briefs": {},
            "polished_chapters": [],
            "qa_passed_chapters": [],
        }
    state["chapter_evaluation"] = chapter_name
    state["v3_probe"] = probe
    state["bulk_backend"] = bulk.backend_name

    memory = base_memory
    try:
        result = _run_full_v3(harness, bulk, document, targets, chapters, state)
        latest = hybrid._cached_state(SOURCE, CACHE)
        _write_exports(
            chapter_name,
            targets,
            latest,
            memory,
            status=str(latest.get("status") or "complete"),
            extra={"run_result": result, "deepseek_usage": harness.usage},
        )
        print(
            "[chapter-v3-done] "
            + json.dumps(
                {
                    "chapter": chapter_name,
                    "segments": len(targets),
                    "gigachat_usage": bulk.usage.as_dict(),
                    "deepseek_usage": harness.usage.get("total", {}),
                    "result": result,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except BaseException as exc:
        latest = hybrid._cached_state(SOURCE, CACHE)
        _write_exports(
            chapter_name,
            targets,
            latest,
            memory,
            status="partial",
            extra={"error": f"{type(exc).__name__}: {exc}", "deepseek_usage": getattr(harness, "usage", {})},
        )
        raise


if __name__ == "__main__":
    main()
