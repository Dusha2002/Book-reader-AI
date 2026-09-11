from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chapter_reference_translation_v3 as v3
import hybrid_reference_translation as hybrid
from bookai.critics import literary_gate_batch
from bookai.literary_context import (
    atomic_persist,
    locked_glossary_violations,
    select_refinement_targets,
)
from bookai.pipeline import _cache_path, _translated_context
from bookai.quality_v3 import enhanced_candidate_issues
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED
from bookai.resilience import resilient_findings


# Quality-v4 deliberately reuses the proven v3 draft/context/checkpoint machinery.
# The architectural change is post-draft: reference-free QE first, targeted editing
# second. Clean prose is never rewritten merely because it looks statistically risky.


def _local_batches(selected: list, targets: list, *, max_segments: int = 6, max_chars: int = 6000, max_gap: int = 2):
    if not selected:
        return []
    position = {segment.id: index for index, segment in enumerate(targets)}
    rows = sorted(selected, key=lambda segment: position.get(segment.id, 10**9))
    batches: list[list] = []
    current: list = []
    chars = 0
    previous_index: int | None = None
    for segment in rows:
        index = position.get(segment.id, 10**9)
        gap = 0 if previous_index is None else index - previous_index
        split = bool(
            current
            and (
                len(current) >= max_segments
                or chars + len(segment.text) > max_chars
                or gap > max_gap
            )
        )
        if split:
            batches.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += len(segment.text)
        previous_index = index
    if current:
        batches.append(current)
    return batches


def _semantic_findings(harness, batch, translated, memory):
    checker = getattr(harness, "_semantic_confirmation_batch", None)
    if checker is None:
        from bookai.critics import semantic_gate_batch

        return resilient_findings(
            list(batch),
            lambda part: semantic_gate_batch(harness.gate, part, translated, memory),
            label="v4_semantic_qe",
            attempts=2,
        )
    return resilient_findings(
        list(batch),
        lambda part: checker(part, translated, memory),
        label="v4_semantic_confirmation",
        attempts=2,
    )


def _deterministically_safe(segment, candidate: str, memory, targets) -> bool:
    issues = enhanced_candidate_issues(
        segment,
        candidate,
        memory,
        source_segments=targets,
    )
    if any(issue.severity == "hard" for issue in issues):
        return False
    return not bool(
        locked_glossary_violations(
            [segment],
            {segment.id: candidate},
            REFERENCE_GLOSSARY_SEED,
        )
    )


def _parallel_qe(harness, batches, translated, memory, *, workers: int):
    findings = []
    if not batches:
        return findings
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bookai-v4-qe") as pool:
        futures = {
            pool.submit(_semantic_findings, harness, batch, translated, memory): batch
            for batch in batches
        }
        for future in as_completed(futures):
            findings.extend(future.result())
    return findings


def _reason_map(findings) -> dict[str, str]:
    rows: dict[str, list[str]] = {}
    for finding in findings:
        rows.setdefault(finding.id, []).append(finding.reason)
    return {sid: "; ".join(dict.fromkeys(reasons)) for sid, reasons in rows.items()}


def _targeted_edit(
    harness,
    targets,
    selected,
    translated,
    memory,
    reasons,
    *,
    workers: int,
):
    """Reason-aware APE: edit only flagged passages, preserving clean draft text."""
    batches = _local_batches(selected, targets)
    snapshot = dict(translated)
    accepted: dict[str, str] = {}

    def run(batch):
        context = _translated_context(targets, batch, snapshot, radius=3)
        rows = harness.edit(
            batch,
            snapshot,
            memory,
            reasons={segment.id: reasons.get(segment.id, "targeted QE repair") for segment in batch},
            context=context,
        )
        return batch, rows

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bookai-v4-edit") as pool:
        futures = {pool.submit(run, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch, rows = future.result()
            for segment in batch:
                candidate = rows.get(segment.id)
                if candidate and _deterministically_safe(segment, candidate, memory, targets):
                    accepted[segment.id] = candidate
    return accepted


def _semantic_qe_repair(harness, targets, translated, memory) -> dict:
    """Audit every segment, then edit only true semantic failures.

    This replaces v3's heuristic short-line retranslations. The critic must return
    an explicit semantic verdict for every audited id, so a long fluent paragraph
    cannot silently lose a proposition and still escape simply because its length
    ratio looks normal.
    """
    qe_chars = max(3000, int(os.getenv("BOOKAI_SEMANTIC_QE_BATCH_CHARS") or "9000"))
    qe_workers = max(1, min(6, int(os.getenv("BOOKAI_SEMANTIC_QE_WORKERS") or "4")))
    edit_workers = max(1, min(4, int(os.getenv("BOOKAI_TARGETED_EDIT_WORKERS") or "4")))
    cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "80"))

    batches: list[list] = []
    current: list = []
    chars = 0
    for segment in targets:
        if current and chars + len(segment.text) > qe_chars:
            batches.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += len(segment.text)
    if current:
        batches.append(current)

    hybrid.progress(
        {
            "phase": "v4_semantic_qe",
            "audited": len(targets),
            "batches": len(batches),
            "workers": qe_workers,
            "strategy": "all-segment-reference-free-QE→reason-aware-targeted-edit→semantic-recheck",
        }
    )
    findings = _parallel_qe(harness, batches, translated, memory, workers=qe_workers)
    reasons = _reason_map(findings)
    flagged_ids = list(dict.fromkeys(finding.id for finding in findings))
    if cap and len(flagged_ids) > cap:
        raise RuntimeError(
            f"v4 semantic QE flagged {len(flagged_ids)} segments, exceeding paid repair cap {cap}; "
            "fail closed instead of silently accepting unaudited corruption"
        )

    by_id = {segment.id: segment for segment in targets}
    repair_targets = [by_id[sid] for sid in flagged_ids if sid in by_id]
    if not repair_targets:
        hybrid.progress({"phase": "v4_semantic_qe_done", "audited": len(targets), "flagged": 0, "repaired": 0})
        return {
            "audited": len(targets),
            "flagged": 0,
            "edited": 0,
            "fallback_retranslated": 0,
            "remaining": 0,
        }

    edited = _targeted_edit(
        harness,
        targets,
        repair_targets,
        translated,
        memory,
        reasons,
        workers=edit_workers,
    )
    candidate_map = dict(translated)
    candidate_map.update(edited)

    # Never trust an editor merely because it produced valid Russian. Re-run the
    # proposition-level semantic checker on every changed segment.
    recheck_batches = _local_batches(
        [by_id[sid] for sid in edited if sid in by_id],
        targets,
        max_segments=8,
        max_chars=8000,
        max_gap=3,
    )
    recheck = _parallel_qe(harness, recheck_batches, candidate_map, memory, workers=qe_workers)
    still_bad = {finding.id for finding in recheck}
    accepted_edit_ids = set(edited) - still_bad
    for sid in accepted_edit_ids:
        translated[sid] = edited[sid]

    fallback_ids = set(flagged_ids) - accepted_edit_ids
    fallback_retranslated = 0
    if fallback_ids:
        fallback_targets = [by_id[sid] for sid in fallback_ids if sid in by_id]
        # ReferenceTranslationHarness automatically sentence-decomposes genuinely
        # hard long paragraphs, giving the last-resort path smaller semantic tasks.
        for batch in _local_batches(fallback_targets, targets, max_segments=4, max_chars=5000, max_gap=1):
            before, after = v3._context_for(targets, batch, radius=5)
            rows = harness.translate(batch, memory, context_before=before, context_after=after)
            for segment in batch:
                candidate = rows.get(segment.id)
                if candidate and _deterministically_safe(segment, candidate, memory, targets):
                    translated[segment.id] = candidate
                    fallback_retranslated += 1

        final_targets = [by_id[sid] for sid in fallback_ids if sid in by_id]
        final_batches = _local_batches(final_targets, targets, max_segments=8, max_chars=8000, max_gap=3)
        final_findings = _parallel_qe(harness, final_batches, translated, memory, workers=qe_workers)
        remaining = {finding.id for finding in final_findings} | {
            segment.id
            for segment in final_targets
            if not _deterministically_safe(segment, translated.get(segment.id, ""), memory, targets)
        }
        if remaining:
            raise RuntimeError(
                "v4 semantic QE still rejects repaired segments: " + ", ".join(sorted(remaining)[:20])
            )
    else:
        remaining = set()

    stats = {
        "audited": len(targets),
        "flagged": len(flagged_ids),
        "edited": len(accepted_edit_ids),
        "fallback_retranslated": fallback_retranslated,
        "remaining": len(remaining),
    }
    hybrid.progress({"phase": "v4_semantic_qe_done", **stats})
    return stats


def _literary_qe_refinement(
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
    """Risk-select candidates, but rewrite only passages the literary critic flags."""
    limit = max(0, int(os.getenv("BOOKAI_RAPID_REFINE_MAX") or "20"))
    glossary = locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
    selected = select_refinement_targets(
        targets,
        translated,
        limit=limit,
        locked_violations=glossary,
    )
    if not selected:
        state["v4_literary_qe"] = {"selected": 0, "flagged": 0, "accepted": 0}
        hybrid.progress({"phase": "v4_literary_qe_done", **state["v4_literary_qe"]})
        return translated

    gate_workers = max(1, min(4, int(os.getenv("BOOKAI_LITERARY_QE_WORKERS") or "4")))
    batches = _local_batches(selected, targets, max_segments=8, max_chars=7000, max_gap=3)

    def audit(batch):
        return resilient_findings(
            list(batch),
            lambda part: literary_gate_batch(harness.gate, part, translated, memory),
            label="v4_literary_qe",
            attempts=2,
        )

    findings = []
    with ThreadPoolExecutor(max_workers=gate_workers, thread_name_prefix="bookai-v4-literary-qe") as pool:
        futures = {pool.submit(audit, batch): batch for batch in batches}
        for future in as_completed(futures):
            findings.extend(future.result())

    reasons = _reason_map(findings)
    flagged_ids = list(dict.fromkeys(finding.id for finding in findings))
    by_id = {segment.id: segment for segment in targets}
    flagged = [by_id[sid] for sid in flagged_ids if sid in by_id]
    if not flagged:
        state["v4_literary_qe"] = {"selected": len(selected), "flagged": 0, "accepted": 0}
        hybrid.progress({"phase": "v4_literary_qe_done", **state["v4_literary_qe"]})
        return translated

    edited = _targeted_edit(
        harness,
        targets,
        flagged,
        translated,
        memory,
        reasons,
        workers=gate_workers,
    )
    candidate_map = dict(translated)
    candidate_map.update(edited)

    # Safety gate: a stylistic edit is accepted only if it remains semantically
    # clean and the literary critic no longer reports the target as defective.
    semantic_batches = _local_batches(
        [by_id[sid] for sid in edited if sid in by_id],
        targets,
        max_segments=8,
        max_chars=7000,
        max_gap=3,
    )
    semantic_bad = {
        finding.id
        for finding in _parallel_qe(harness, semantic_batches, candidate_map, memory, workers=gate_workers)
    }

    literary_recheck = []
    recheck_targets = [by_id[sid] for sid in edited if sid not in semantic_bad and sid in by_id]
    for batch in _local_batches(recheck_targets, targets, max_segments=8, max_chars=7000, max_gap=3):
        literary_recheck.extend(
            resilient_findings(
                list(batch),
                lambda part: literary_gate_batch(harness.gate, part, candidate_map, memory),
                label="v4_literary_recheck",
                attempts=2,
            )
        )
    literary_bad = {finding.id for finding in literary_recheck}
    accepted_ids = set(edited) - semantic_bad - literary_bad
    for sid in accepted_ids:
        translated[sid] = edited[sid]

    stats = {
        "selected": len(selected),
        "flagged": len(flagged_ids),
        "edit_candidates": len(edited),
        "accepted": len(accepted_ids),
        "semantic_rejected": len(semantic_bad),
        "literary_rejected": len(literary_bad),
    }
    state["v4_literary_qe"] = stats
    atomic_persist(_cache_path(v3.SOURCE, v3.CACHE, "optimal"), state, translated, memory)
    hybrid.progress({"phase": "v4_literary_qe_done", **stats})
    return translated


def _configure_v4() -> None:
    slug = v3.CHAPTER_SLUG
    v3.OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V4.fb2")
    v3.CACHE = Path(f".bookai-cache-chapter-eval-v4-{slug}")
    v3.PROGRESS = Path("chapter-v4-progress.json")
    v3.PROBE = Path("chapter-v4-probe.json")
    v3.ROUTING = Path("chapter-v4-routing.json")
    v3.REPORT = Path("chapter-v4.json")
    v3.SOURCE_TXT = Path("chapter-v4-source.txt")
    v3.TRANSLATED_TXT = Path("chapter-v4-translated.txt")
    v3.MAP_JSON = Path("chapter-v4-translation-map.json")

    # Surgical replacement of only the two weak v3 stages. Draft translation,
    # parallel book analysis, retry/recovery, checkpoints and export stay proven v3.
    v3._semantic_short_repair = _semantic_qe_repair
    hybrid._selective_literary_refinement = _literary_qe_refinement


def _annotate_report() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture_revision"] = "quality-v4-qe-targeted-repair"
    report["architecture"] = {
        "primary": "GigaChat-3-Lightning-v3 strict-json-schema",
        "parallel_reasoning_analysis": "deepseek/deepseek-v4.1-flash",
        "semantic_qe": "all-segment explicit-verdict DeepSeek V4.1 Flash",
        "semantic_repair": "reason-aware targeted edit; sentence-decomposed translation only as fallback",
        "literary_qe": "bounded risk-selected audit before edit",
        "literary_repair": "edit only flagged segments; semantic+literary recheck before acceptance",
        "deterministic_invariants": "numbers+coverage+questions+mixed-script+latin-residue+dynamic-entity-ledger+locked-glossary",
        "external_pro_judge": False,
        "full_paid_fallback": False,
    }
    report["v4_literary_qe"] = (report.get("extra") or {}).get("v4_literary_qe") or report.get("v4_literary_qe") or {}
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v4()
    try:
        v3.main()
    finally:
        _annotate_report()


if __name__ == "__main__":
    main()
