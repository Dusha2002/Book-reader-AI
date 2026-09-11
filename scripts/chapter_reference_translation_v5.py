from __future__ import annotations

import json
import os
import time
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v4 as v4
import chapter_reference_translation_v4b as v4b
import hybrid_reference_translation as hybrid
from bookai.literary_context import atomic_persist
from bookai.pipeline import _cache_path, _chapter_groups, _should_translate
from bookai.quality import candidate_issues
from bookai.quality_v3 import QualityIssueV3


# v5 is intentionally a simplification, not another stack of translators:
#   GigaChat bulk draft (no paid model in the hot path)
#   -> one all-segment semantic QE pass
#   -> at most two reason-aware minimal edit rounds
#   -> a tiny bounded direct DeepSeek candidate fallback (no v10 micro-splitting)
#   -> bounded literary QE/edit only for risky passages
# Clean segments are never rewritten. A stubborn segment marks needs_review instead
# of aborting the whole chapter.

_LARGEST_ALIASES = {"largest", "largest-chapter", "__largest__"}
_CATASTROPHIC_DRAFT_CODES = {
    "empty",
    "unchanged",
    "unexpected_script",
    "english_leftover",
    "service_leak",
    "repetition_loop",
    "heading_multiline",
    "heading_expanded",
    "chapter_heading",
}

_V5_REMAINING: set[str] = set()
_V5_REASONS: dict[str, str] = {}
_V5_SEMANTIC_STATS: dict = {}
_ORIGINAL_ENHANCED_BATCH_ISSUES = v3.enhanced_batch_issues
_ORIGINAL_SELECT_CHAPTER = v3._select_chapter


def _is_largest_request() -> bool:
    return str(v3.CHAPTER_NAME or "").strip().casefold() in _LARGEST_ALIASES


def _select_chapter(document):
    if not _is_largest_request():
        return _ORIGINAL_SELECT_CHAPTER(document)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(targets)
    if not groups:
        raise RuntimeError("No translatable chapters found")
    chapter_name, chapter = max(
        groups,
        key=lambda pair: (sum(len(segment.text) for segment in pair[1]), len(pair[1])),
    )
    print(
        "[v5-largest-chapter] "
        + json.dumps(
            {
                "chapter": chapter_name,
                "segments": len(chapter),
                "source_chars": sum(len(segment.text) for segment in chapter),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return chapter_name, chapter


def _draft_bad(segment, candidate: str, memory, source_segments=None) -> bool:
    """Only reject catastrophic draft failures before the rich QE stage.

    Numbers, terminology, names, isolated Latin residue and semantic nuance are
    deliberately diagnosed after the full draft, where the richer book memory is
    available. This prevents an early local defect from triggering an expensive
    DeepSeek paragraph retranslation in the bulk hot path.
    """
    for issue in candidate_issues(segment, candidate, memory):
        if issue.severity == "hard" and issue.code in _CATASTROPHIC_DRAFT_CODES:
            return True
    return False


def _translate_gigachat_only_batch(
    harness,
    bulk,
    source_segments,
    batch,
    memory,
    stats,
):
    """Pure bulk draft: zero DeepSeek routing/escalation in normal translation."""
    rows, errors = bulk.translate_many(batch, memory, source_segments=source_segments)
    result: dict[str, str] = {}
    unresolved = set(errors)
    for segment in batch:
        candidate = rows.get(segment.id)
        if candidate and not _draft_bad(segment, candidate, memory, source_segments):
            result[segment.id] = candidate
        else:
            unresolved.add(segment.id)
    stats.add(
        bulk_accepted=len(result),
        bulk_errors=len(set(errors)),
        source_chars_bulk=sum(len(segment.text) for segment in batch),
    )
    return result, unresolved


def _direct_deepseek_recovery(harness, targets, translated, pending_ids, memory, stats):
    """Rare transport/catastrophic recovery without ReferenceHarness micro-splitting."""
    if not pending_ids:
        return set()
    by_id = {segment.id: segment for segment in targets}
    cap = max(0, int(os.getenv("BOOKAI_DRAFT_RECOVERY_MAX") or "6"))
    ids = sorted(pending_ids)[:cap] if cap else []
    still = set(pending_ids) - set(ids)
    for start in range(0, len(ids), 3):
        batch = [by_id[sid] for sid in ids[start : start + 3] if sid in by_id]
        if not batch:
            continue
        before, after = v3._context_for(targets, batch, radius=4)
        try:
            # Call the underlying translator directly. This intentionally bypasses
            # ReferenceTranslationHarness.translate(), whose v10 sentence micro-path
            # caused the v4 request explosion.
            rows = harness.translator.translate(
                batch,
                memory,
                context_before=before,
                context_after=after,
            )
        except Exception as exc:
            print(f"[v5-draft-recovery] error={type(exc).__name__} ids={[s.id for s in batch]}", flush=True)
            still.update(segment.id for segment in batch)
            continue
        for segment in batch:
            candidate = rows.get(segment.id)
            if candidate and not _draft_bad(segment, candidate, memory, targets):
                translated[segment.id] = candidate
                stats.add(deep_escalated=1, source_chars_deep=len(segment.text))
            else:
                still.add(segment.id)
    return still


def _fast_analysis_task(harness, chapters, state_snapshot: dict):
    """One control-plane call for a chapter evaluation.

    Full-book production can keep cached chapter digests/synopsis. For a one-chapter
    benchmark, asking DeepSeek to independently build analyzer memory + style card +
    chapter brief + synopsis duplicates the same evidence and dominated v4 latency.
    The reference profile already supplies the stable target style contract, so one
    analyzer pass is sufficient here.
    """
    started = time.perf_counter()
    memory = hybrid._load_or_build_memory(harness, state_snapshot, chapters)
    compact = " ".join(str(memory.rolling_summary or "").split())[:5000]
    chapter_digests = {name: compact for name, _ in chapters if compact}
    book_synopsis = compact
    state_snapshot["chapter_briefs"] = chapter_digests
    state_snapshot["book_synopsis"] = book_synopsis
    state_snapshot["context_strategy"] = {
        "analysis": "single-pass-chapter-eval",
        "style": "reference-profile+analyzer",
        "retrieval": "idf-distant-source-k2",
        "semantic_qe": "all-segment-reference-free",
        "repair": "diagnose-then-minimal-edit",
    }
    elapsed = time.perf_counter() - started
    hybrid.progress(
        {
            "phase": "context_ready",
            "analysis_calls": 1,
            "chapter_digests": len(chapter_digests),
            "synopsis_chars": len(book_synopsis),
            "strategy": "single-pass-chapter-eval",
        }
    )
    print(
        "[v5-analysis] "
        + json.dumps(
            {
                "elapsed_seconds": round(elapsed, 2),
                "chapter_digests": len(chapter_digests),
                "synopsis_chars": len(book_synopsis),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return memory, chapter_digests, book_synopsis, state_snapshot, elapsed


def _semantic_batches(targets, max_chars: int) -> list[list]:
    batches: list[list] = []
    current: list = []
    chars = 0
    for segment in targets:
        if current and chars + len(segment.text) > max_chars:
            batches.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += len(segment.text)
    if current:
        batches.append(current)
    return batches


def _direct_fresh_candidates(harness, targets, selected, translated, memory, *, limit: int):
    """Generate one independent candidate per stubborn target, without micro-translation."""
    if limit <= 0 or not selected:
        return {}
    snapshot = dict(translated)
    chosen = list(selected)[:limit]
    out: dict[str, str] = {}
    for start in range(0, len(chosen), 3):
        batch = chosen[start : start + 3]
        before, after = v3._context_for(targets, batch, radius=5)
        try:
            rows = harness.translator.translate(
                batch,
                memory,
                context_before=before,
                context_after=after,
            )
        except Exception as exc:
            print(f"[v5-fresh-candidate] error={type(exc).__name__} ids={[s.id for s in batch]}", flush=True)
            continue
        for segment in batch:
            candidate = rows.get(segment.id)
            if candidate and candidate.strip() and candidate.strip() != snapshot.get(segment.id, "").strip():
                out[segment.id] = candidate
    return out


def _semantic_qe_repair(harness, targets, translated, memory) -> dict:
    """QE-informed APE: audit once, edit only defects, bounded independent fallback."""
    global _V5_REMAINING, _V5_REASONS, _V5_SEMANTIC_STATS

    qe_chars = max(5000, int(os.getenv("BOOKAI_SEMANTIC_QE_BATCH_CHARS") or "11000"))
    qe_workers = max(1, min(6, int(os.getenv("BOOKAI_SEMANTIC_QE_WORKERS") or "5")))
    edit_workers = max(1, min(4, int(os.getenv("BOOKAI_TARGETED_EDIT_WORKERS") or "4")))
    rounds = max(1, min(2, int(os.getenv("BOOKAI_SEMANTIC_REPAIR_ROUNDS") or "2")))
    repair_cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "80"))
    fresh_cap = max(0, int(os.getenv("BOOKAI_FRESH_RETRANSLATE_MAX") or "8"))

    batches = _semantic_batches(targets, qe_chars)
    hybrid.progress(
        {
            "phase": "v5_semantic_qe",
            "audited": len(targets),
            "batches": len(batches),
            "workers": qe_workers,
            "strategy": "audit-once→minimal-edit×2→bounded-independent-candidate→needs-review",
        }
    )
    findings = v4._parallel_qe(harness, batches, translated, memory, workers=qe_workers)
    critic_reasons = v4b._findings_reason_map(findings)
    deterministic_ids, deterministic_reasons = v4b._deterministic_hard(targets, translated, memory)
    reasons = v4b._merge_reasons(critic_reasons, deterministic_reasons)

    position = {segment.id: index for index, segment in enumerate(targets)}
    by_id = {segment.id: segment for segment in targets}
    flagged = sorted(
        set(critic_reasons) | deterministic_ids,
        key=lambda sid: position.get(sid, 10**9),
    )
    if repair_cap:
        repair_ids = flagged[:repair_cap]
        overflow = set(flagged[repair_cap:])
    else:
        repair_ids = []
        overflow = set(flagged)
    for sid in overflow:
        reasons[sid] = (reasons.get(sid, "") + "; repair cap exceeded").strip("; ")

    pending = set(repair_ids)
    accepted_edits: set[str] = set()
    rounds_used = 0

    for round_index in range(1, rounds + 1):
        if not pending:
            break
        rounds_used = round_index
        selected = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        edits = v4b._targeted_edit_batched(
            harness,
            targets,
            selected,
            translated,
            memory,
            reasons,
            workers=edit_workers,
        )
        candidate_map = dict(translated)
        candidate_map.update(edits)
        edited_targets = [by_id[sid] for sid in edits if sid in by_id]
        qe_bad = {
            finding.id
            for finding in v4b._semantic_recheck(
                harness,
                targets,
                edited_targets,
                candidate_map,
                memory,
                workers=qe_workers,
            )
        }
        det_bad, det_reasons = v4b._deterministic_hard(edited_targets, candidate_map, memory)
        bad = qe_bad | det_bad
        accepted = set(edits) - bad
        for sid in accepted:
            translated[sid] = edits[sid]
        accepted_edits.update(accepted)
        pending -= accepted
        reasons = v4b._merge_reasons(reasons, det_reasons)
        hybrid.progress(
            {
                "phase": "v5_semantic_edit_round",
                "round": round_index,
                "attempted": len(selected),
                "accepted": len(accepted),
                "remaining": len(pending) + len(overflow),
            }
        )

    fresh_attempted = 0
    fresh_accepted = 0
    if pending and fresh_cap:
        stubborn = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        fresh = _direct_fresh_candidates(
            harness,
            targets,
            stubborn,
            translated,
            memory,
            limit=fresh_cap,
        )
        fresh_attempted = len(fresh)
        candidate_map = dict(translated)
        candidate_map.update(fresh)
        fresh_targets = [by_id[sid] for sid in fresh if sid in by_id]
        qe_bad = {
            finding.id
            for finding in v4b._semantic_recheck(
                harness,
                targets,
                fresh_targets,
                candidate_map,
                memory,
                workers=qe_workers,
            )
        }
        det_bad, det_reasons = v4b._deterministic_hard(fresh_targets, candidate_map, memory)
        bad = qe_bad | det_bad
        accepted = set(fresh) - bad
        for sid in accepted:
            translated[sid] = fresh[sid]
        fresh_accepted = len(accepted)
        pending -= accepted
        reasons = v4b._merge_reasons(reasons, det_reasons)

    remaining = set(pending) | overflow
    _V5_REMAINING = remaining
    _V5_REASONS = {sid: reasons.get(sid, "semantic QE unresolved") for sid in remaining}
    _V5_SEMANTIC_STATS = {
        "audited": len(targets),
        "flagged": len(flagged),
        "repair_cap": repair_cap,
        "edited_accepted": len(accepted_edits),
        "repair_rounds": rounds_used,
        "fresh_attempted": fresh_attempted,
        "fresh_accepted": fresh_accepted,
        "remaining": len(remaining),
        "remaining_ids": sorted(remaining),
    }
    hybrid.progress({"phase": "v5_semantic_qe_done", **_V5_SEMANTIC_STATS})
    # Deliberately do not raise. A usable chapter with explicit unresolved ids is
    # more valuable than losing all output because one critic/editor disagrees.
    return dict(_V5_SEMANTIC_STATS)


def _skip_duplicate_hard_repair(
    harness,
    targets,
    translated,
    memory,
    state,
    *,
    chapter_digests,
    book_synopsis,
    context_index,
):
    hybrid.progress(
        {
            "phase": "v5_duplicate_repair_skipped",
            "reason": "all-segment semantic QE already owns hard semantic/deterministic repair",
        }
    )
    return translated


def _final_issues(segments, translations, memory=None, *, source_segments=None):
    issues = list(
        _ORIGINAL_ENHANCED_BATCH_ISSUES(
            segments,
            translations,
            memory,
            source_segments=source_segments,
        )
    )
    existing = {(issue.id, issue.code) for issue in issues}
    valid_ids = {segment.id for segment in segments}
    for sid in sorted(_V5_REMAINING):
        if sid in valid_ids and (sid, "semantic_qe_unresolved") not in existing:
            issues.append(
                QualityIssueV3(
                    sid,
                    "hard",
                    "semantic_qe_unresolved",
                    _V5_REASONS.get(sid, "semantic QE still reports a defect after bounded repair"),
                )
            )
    return issues


def _configure_v5() -> None:
    slug = "largest" if _is_largest_request() else v3.CHAPTER_SLUG
    v3.OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V5.fb2")
    v3.CACHE = Path(f".bookai-cache-chapter-eval-v5-{slug}")
    v3.PROGRESS = Path("chapter-v5-progress.json")
    v3.PROBE = Path("chapter-v5-probe.json")
    v3.ROUTING = Path("chapter-v5-routing.json")
    v3.REPORT = Path("chapter-v5.json")
    v3.SOURCE_TXT = Path("chapter-v5-source.txt")
    v3.TRANSLATED_TXT = Path("chapter-v5-translated.txt")
    v3.MAP_JSON = Path("chapter-v5-translation-map.json")

    # Hot path: GigaChat only.
    v3._select_chapter = _select_chapter
    v3._candidate_bad_v3 = _draft_bad
    v3._analysis_task = _fast_analysis_task
    hybrid._translate_hybrid_batch = _translate_gigachat_only_batch
    hybrid._deepseek_final_recovery = _direct_deepseek_recovery

    # Post-draft quality path.
    v3._semantic_short_repair = _semantic_qe_repair
    hybrid._repair_hard_failures = _skip_duplicate_hard_repair
    hybrid._selective_literary_refinement = v4b._literary_qe_refinement
    v3.enhanced_batch_issues = _final_issues


def _annotate_report() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        "version": "quality-v5",
        "primary": "GigaChat-3-Lightning bulk-only hot path",
        "analysis": "single DeepSeek V4.1 Flash chapter-eval pass",
        "semantic_qe": "all segments once + bounded rechecks only for edits",
        "semantic_repair": "reason-aware minimal edits, max 2 rounds",
        "fallback": "bounded direct DeepSeek candidate; no sentence-micro cascade",
        "literary_refinement": "risk-selected QE then edit only flagged",
        "failure_policy": "needs_review instead of chapter abort",
        "deepseek_v4_pro": False,
    }
    report["v5_semantic_stats"] = _V5_SEMANTIC_STATS
    report["v5_remaining"] = [
        {"id": sid, "reason": _V5_REASONS.get(sid, "semantic QE unresolved")}
        for sid in sorted(_V5_REMAINING)
    ]
    if _V5_REMAINING:
        report["status"] = "needs_review"
        report["state_status"] = "needs_review"
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v5()
    try:
        v3.main()
    finally:
        _annotate_report()


if __name__ == "__main__":
    main()
