from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v4 as v4
import hybrid_reference_translation as hybrid
from bookai.critics import literary_gate_batch
from bookai.literary_context import (
    atomic_persist,
    locked_glossary_violations,
    select_refinement_targets,
)
from bookai.pipeline import _cache_path
from bookai.quality import candidate_issues
from bookai.quality_v3 import enhanced_candidate_issues
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED
from bookai.resilience import resilient_findings


# v4b keeps the fast v3 bulk draft deliberately permissive and moves the new
# strict checks to the post-draft QE stage. This avoids paying DeepSeek to
# retranslate a whole paragraph merely because the draft contains one local
# defect that can be diagnosed and minimally edited later.

_RU_STRAIGHT_QUOTE = re.compile(r"(?:^|[\s(])['\"]\s*[А-Яа-яЁё]")

_V4B_SEMANTIC_STATS: dict = {}
_V4B_SEMANTIC_FINDINGS: list[dict] = []
_V4B_LITERARY_STATS: dict = {}


def _bulk_candidate_bad(segment, candidate: str, memory, source_segments=None) -> bool:
    """Cheap first-pass safety only; rich QE runs after book context is ready."""
    issues = candidate_issues(segment, candidate, memory)
    if any(issue.severity == "hard" for issue in issues):
        return True
    return bool(
        locked_glossary_violations(
            [segment],
            {segment.id: candidate},
            REFERENCE_GLOSSARY_SEED,
        )
    )


def _multi_local_context(targets, batch, translations, *, radius: int = 2, max_rows: int = 28) -> list[dict]:
    """Give every non-contiguous edit target its own nearby bilingual context.

    This lets us batch several sparse QE failures into one API request without
    sacrificing the local context that would be lost by using only the first and
    last target as a single range.
    """
    position = {segment.id: index for index, segment in enumerate(targets)}
    wanted: set[int] = set()
    for segment in batch:
        idx = position.get(segment.id)
        if idx is None:
            continue
        for j in range(max(0, idx - radius), min(len(targets), idx + radius + 1)):
            if targets[j].id != segment.id:
                wanted.add(j)
    rows = [
        {
            "id": targets[index].id,
            "source": targets[index].text,
            "translation": translations.get(targets[index].id, ""),
        }
        for index in sorted(wanted)
    ]
    if len(rows) <= max_rows:
        return rows
    # Keep an even spread so each sparse target retains some surrounding evidence.
    step = (len(rows) - 1) / max(1, max_rows - 1)
    return [rows[round(i * step)] for i in range(max_rows)]


def _sparse_batches(selected, targets, *, max_segments: int = 6, max_chars: int = 7000):
    position = {segment.id: index for index, segment in enumerate(targets)}
    ordered = sorted(selected, key=lambda segment: position.get(segment.id, 10**9))
    batches: list[list] = []
    current: list = []
    chars = 0
    for segment in ordered:
        if current and (len(current) >= max_segments or chars + len(segment.text) > max_chars):
            batches.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += len(segment.text)
    if current:
        batches.append(current)
    return batches


def _targeted_edit_batched(
    harness,
    targets,
    selected,
    translated,
    memory,
    reasons,
    *,
    workers: int,
    memory_builder=None,
):
    """Minimal APE with sparse batching and per-target local context."""
    if not selected:
        return {}
    snapshot = dict(translated)
    accepted: dict[str, str] = {}
    batches = _sparse_batches(selected, targets)

    def run(batch):
        batch_memory = memory_builder(batch) if memory_builder else memory
        context = _multi_local_context(targets, batch, snapshot, radius=2)
        rows = harness.edit(
            batch,
            snapshot,
            batch_memory,
            reasons={segment.id: reasons.get(segment.id, "targeted QE repair") for segment in batch},
            context=context,
        )
        return batch, rows

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bookai-v4b-edit") as pool:
        futures = {pool.submit(run, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch, rows = future.result()
            for segment in batch:
                candidate = rows.get(segment.id)
                if candidate:
                    accepted[segment.id] = candidate
    return accepted


def _deterministic_hard(targets, translations, memory) -> tuple[set[str], dict[str, str]]:
    hard: set[str] = set()
    reasons: dict[str, list[str]] = {}
    for segment in targets:
        candidate = translations.get(segment.id, "")
        for issue in enhanced_candidate_issues(
            segment,
            candidate,
            memory,
            source_segments=targets,
        ):
            if issue.severity != "hard":
                continue
            hard.add(segment.id)
            reasons.setdefault(segment.id, []).append(f"{issue.code}: {issue.reason}")
        if locked_glossary_violations(
            [segment],
            {segment.id: candidate},
            REFERENCE_GLOSSARY_SEED,
        ):
            hard.add(segment.id)
            reasons.setdefault(segment.id, []).append("locked glossary mismatch")
    return hard, {sid: "; ".join(dict.fromkeys(rows)) for sid, rows in reasons.items()}


def _semantic_recheck(harness, targets, subset, translations, memory, *, workers: int):
    batches = v4._local_batches(
        subset,
        targets,
        max_segments=8,
        max_chars=8500,
        max_gap=10**9,
    )
    return v4._parallel_qe(harness, batches, translations, memory, workers=workers)


def _merge_reasons(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, list[str]] = {}
    for mapping in maps:
        for sid, reason in mapping.items():
            if reason:
                merged.setdefault(sid, []).append(reason)
    return {sid: "; ".join(dict.fromkeys(rows)) for sid, rows in merged.items()}


def _findings_reason_map(findings) -> dict[str, str]:
    rows: dict[str, list[str]] = {}
    for finding in findings:
        rows.setdefault(finding.id, []).append(finding.reason)
    return {sid: "; ".join(dict.fromkeys(reasons)) for sid, reasons in rows.items()}


def _semantic_qe_repair(harness, targets, translated, memory) -> dict:
    """All-segment QE -> diagnosis-guided minimal edits -> bounded fallback.

    Compared with the first v4 attempt, this does not jump from one failed edit
    straight to a fresh translation. It performs a second diagnose/edit round
    first, because a precise local correction is both cheaper and less likely to
    introduce new literary drift. Fresh sentence-decomposed translation is the
    last resort only.
    """
    global _V4B_SEMANTIC_STATS, _V4B_SEMANTIC_FINDINGS

    qe_chars = max(3000, int(os.getenv("BOOKAI_SEMANTIC_QE_BATCH_CHARS") or "9000"))
    qe_workers = max(1, min(6, int(os.getenv("BOOKAI_SEMANTIC_QE_WORKERS") or "4")))
    edit_workers = max(1, min(4, int(os.getenv("BOOKAI_TARGETED_EDIT_WORKERS") or "4")))
    repair_rounds = max(1, min(3, int(os.getenv("BOOKAI_SEMANTIC_REPAIR_ROUNDS") or "2")))
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
            "phase": "v4b_semantic_qe",
            "audited": len(targets),
            "batches": len(batches),
            "workers": qe_workers,
            "strategy": "all-segment-QE→diagnose→minimal-edit×2→sentence-decomposed-fallback→final-edit",
        }
    )

    critic_findings = v4._parallel_qe(harness, batches, translated, memory, workers=qe_workers)
    critic_reasons = _findings_reason_map(critic_findings)
    deterministic_ids, deterministic_reasons = _deterministic_hard(targets, translated, memory)
    reasons = _merge_reasons(critic_reasons, deterministic_reasons)
    flagged_ids = set(critic_reasons) | deterministic_ids

    position = {segment.id: index for index, segment in enumerate(targets)}
    ordered_flagged = sorted(flagged_ids, key=lambda sid: position.get(sid, 10**9))
    if cap and len(ordered_flagged) > cap:
        raise RuntimeError(
            f"v4b semantic QE flagged {len(ordered_flagged)} segments, exceeding paid repair cap {cap}"
        )

    _V4B_SEMANTIC_FINDINGS = [
        {
            "id": sid,
            "reason": reasons.get(sid, "semantic/deterministic failure"),
        }
        for sid in ordered_flagged
    ]
    for row in _V4B_SEMANTIC_FINDINGS[:40]:
        print("[v4b-semantic-finding] " + json.dumps(row, ensure_ascii=False), flush=True)

    if not ordered_flagged:
        _V4B_SEMANTIC_STATS = {
            "audited": len(targets),
            "flagged": 0,
            "edited_accepted": 0,
            "fallback_retranslated": 0,
            "remaining": 0,
            "repair_rounds": 0,
        }
        hybrid.progress({"phase": "v4b_semantic_qe_done", **_V4B_SEMANTIC_STATS})
        return dict(_V4B_SEMANTIC_STATS)

    by_id = {segment.id: segment for segment in targets}
    pending = set(ordered_flagged)
    edited_accepted: set[str] = set()
    rounds_used = 0

    for round_index in range(1, repair_rounds + 1):
        if not pending:
            break
        rounds_used = round_index
        selected = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        edits = _targeted_edit_batched(
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
        semantic_findings = _semantic_recheck(
            harness,
            targets,
            edited_targets,
            candidate_map,
            memory,
            workers=qe_workers,
        )
        semantic_bad = {finding.id for finding in semantic_findings}
        semantic_reasons = _findings_reason_map(semantic_findings)
        deterministic_bad, det_reasons = _deterministic_hard(edited_targets, candidate_map, memory)
        bad = semantic_bad | deterministic_bad

        accepted = set(edits) - bad
        for sid in accepted:
            translated[sid] = edits[sid]
        edited_accepted.update(accepted)
        pending -= accepted
        reasons = _merge_reasons(reasons, semantic_reasons, det_reasons)

        hybrid.progress(
            {
                "phase": "v4b_semantic_repair_round",
                "round": round_index,
                "attempted": len(selected),
                "accepted": len(accepted),
                "remaining": len(pending),
            }
        )

    fallback_retranslated = 0
    if pending:
        fallback_targets = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        for batch in _sparse_batches(fallback_targets, targets, max_segments=4, max_chars=5000):
            before, after = v3._context_for(targets, batch, radius=5)
            rows = harness.translate(batch, memory, context_before=before, context_after=after)
            for segment in batch:
                candidate = rows.get(segment.id)
                if candidate:
                    translated[segment.id] = candidate
                    fallback_retranslated += 1

        fallback_findings = _semantic_recheck(
            harness,
            targets,
            fallback_targets,
            translated,
            memory,
            workers=qe_workers,
        )
        fallback_semantic_bad = {finding.id for finding in fallback_findings}
        fallback_semantic_reasons = _findings_reason_map(fallback_findings)
        fallback_det_bad, fallback_det_reasons = _deterministic_hard(fallback_targets, translated, memory)
        pending = fallback_semantic_bad | fallback_det_bad
        reasons = _merge_reasons(reasons, fallback_semantic_reasons, fallback_det_reasons)

    # One final diagnosis-guided edit after fallback is cheaper and safer than an
    # unbounded retranslation loop. This specifically handles cases where a fresh
    # translation fixed most of a paragraph but left one terminology/actor defect.
    if pending:
        final_targets = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        final_edits = _targeted_edit_batched(
            harness,
            targets,
            final_targets,
            translated,
            memory,
            reasons,
            workers=edit_workers,
        )
        candidate_map = dict(translated)
        candidate_map.update(final_edits)
        final_findings = _semantic_recheck(
            harness,
            targets,
            final_targets,
            candidate_map,
            memory,
            workers=qe_workers,
        )
        final_semantic_bad = {finding.id for finding in final_findings}
        final_det_bad, _ = _deterministic_hard(final_targets, candidate_map, memory)
        final_bad = final_semantic_bad | final_det_bad
        accepted = set(final_edits) - final_bad
        for sid in accepted:
            translated[sid] = final_edits[sid]
        edited_accepted.update(accepted)
        pending = (pending - accepted) | final_bad

    if pending:
        raise RuntimeError(
            "v4b semantic QE still rejects repaired segments: " + ", ".join(sorted(pending)[:20])
        )

    _V4B_SEMANTIC_STATS = {
        "audited": len(targets),
        "flagged": len(ordered_flagged),
        "edited_accepted": len(edited_accepted),
        "fallback_retranslated": fallback_retranslated,
        "remaining": 0,
        "repair_rounds": rounds_used,
    }
    hybrid.progress({"phase": "v4b_semantic_qe_done", **_V4B_SEMANTIC_STATS})
    return dict(_V4B_SEMANTIC_STATS)


def _dialogue_typography_risk(segment, translation: str) -> bool:
    if not translation or not _RU_STRAIGHT_QUOTE.search(translation):
        return False
    source = segment.text.strip()
    return source.startswith(("'", '"', "‘", "“")) or bool(re.search(r"[.!?]\s*['\"’”]", source))


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
    """Bounded literary QE, including dialogue typography, before any rewrite."""
    global _V4B_LITERARY_STATS

    risk_limit = max(0, int(os.getenv("BOOKAI_RAPID_REFINE_MAX") or "20"))
    dialogue_limit = max(0, int(os.getenv("BOOKAI_DIALOGUE_QE_MAX") or "24"))
    workers = max(1, min(4, int(os.getenv("BOOKAI_LITERARY_QE_WORKERS") or "4")))

    glossary = locked_glossary_violations(targets, translated, REFERENCE_GLOSSARY_SEED)
    risk_selected = select_refinement_targets(
        targets,
        translated,
        limit=risk_limit,
        locked_violations=glossary,
    )
    dialogue_selected = [
        segment for segment in targets
        if _dialogue_typography_risk(segment, translated.get(segment.id, ""))
    ][:dialogue_limit]

    seen: set[str] = set()
    selected = []
    for segment in [*risk_selected, *dialogue_selected]:
        if segment.id not in seen:
            selected.append(segment)
            seen.add(segment.id)

    if not selected:
        _V4B_LITERARY_STATS = {"selected": 0, "flagged": 0, "accepted": 0, "remaining": 0}
        state["v4b_literary_qe"] = dict(_V4B_LITERARY_STATS)
        return translated

    batches = v4._local_batches(selected, targets, max_segments=8, max_chars=7500, max_gap=4)

    def audit(batch, mapping):
        return resilient_findings(
            list(batch),
            lambda part: literary_gate_batch(harness.gate, part, mapping, memory),
            label="v4b_literary_qe",
            attempts=2,
        )

    findings = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v4b-literary-qe") as pool:
        futures = {pool.submit(audit, batch, translated): batch for batch in batches}
        for future in as_completed(futures):
            findings.extend(future.result())

    reasons = _findings_reason_map(findings)
    flagged_ids = set(reasons)
    by_id = {segment.id: segment for segment in targets}
    position = {segment.id: index for index, segment in enumerate(targets)}
    pending = set(flagged_ids)
    accepted_total: set[str] = set()

    def rich_memory(batch):
        return hybrid._batch_memory(
            memory,
            batch,
            chapter_digests,
            book_synopsis,
            context_index,
        )

    for round_index in range(1, 3):
        if not pending:
            break
        batch_targets = [by_id[sid] for sid in sorted(pending, key=lambda x: position.get(x, 10**9))]
        edits = _targeted_edit_batched(
            harness,
            targets,
            batch_targets,
            translated,
            memory,
            reasons,
            workers=workers,
            memory_builder=rich_memory,
        )
        candidate_map = dict(translated)
        candidate_map.update(edits)

        semantic_findings = _semantic_recheck(
            harness,
            targets,
            [by_id[sid] for sid in edits if sid in by_id],
            candidate_map,
            memory,
            workers=workers,
        )
        semantic_bad = {finding.id for finding in semantic_findings}
        det_bad, _ = _deterministic_hard(
            [by_id[sid] for sid in edits if sid in by_id],
            candidate_map,
            memory,
        )
        safe_ids = set(edits) - semantic_bad - det_bad

        literary_recheck = []
        safe_targets = [by_id[sid] for sid in safe_ids if sid in by_id]
        for batch in v4._local_batches(safe_targets, targets, max_segments=8, max_chars=7500, max_gap=4):
            literary_recheck.extend(audit(batch, candidate_map))
        literary_bad = {finding.id for finding in literary_recheck}
        literary_reasons = _findings_reason_map(literary_recheck)

        accepted = safe_ids - literary_bad
        for sid in accepted:
            translated[sid] = edits[sid]
        accepted_total.update(accepted)
        pending -= accepted
        reasons = _merge_reasons(reasons, literary_reasons)

    _V4B_LITERARY_STATS = {
        "selected": len(selected),
        "risk_selected": len(risk_selected),
        "dialogue_selected": len(dialogue_selected),
        "flagged": len(flagged_ids),
        "accepted": len(accepted_total),
        "remaining": len(pending),
    }
    state["v4b_literary_qe"] = dict(_V4B_LITERARY_STATS)
    atomic_persist(_cache_path(v3.SOURCE, v3.CACHE, "optimal"), state, translated, memory)
    hybrid.progress({"phase": "v4b_literary_qe_done", **_V4B_LITERARY_STATS})
    return translated


def _configure() -> None:
    slug = v3.CHAPTER_SLUG
    v3.OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V4B.fb2")
    v3.CACHE = Path(f".bookai-cache-chapter-eval-v4b-{slug}")
    v3.PROGRESS = Path("chapter-v4b-progress.json")
    v3.PROBE = Path("chapter-v4b-probe.json")
    v3.ROUTING = Path("chapter-v4b-routing.json")
    v3.REPORT = Path("chapter-v4b.json")
    v3.SOURCE_TXT = Path("chapter-v4b-source.txt")
    v3.TRANSLATED_TXT = Path("chapter-v4b-translated.txt")
    v3.MAP_JSON = Path("chapter-v4b-translation-map.json")

    # Crucial architectural split: cheap baseline QA during bulk generation,
    # strict reference-free QE only after rich context is ready.
    v3._candidate_bad_v3 = _bulk_candidate_bad
    v3._semantic_short_repair = _semantic_qe_repair
    hybrid._selective_literary_refinement = _literary_qe_refinement


def _annotate_report() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture_revision"] = "quality-v4b-diagnose-then-repair"
    report["v4b_semantic_qe"] = dict(_V4B_SEMANTIC_STATS)
    report["v4b_semantic_findings"] = list(_V4B_SEMANTIC_FINDINGS)
    report["v4b_literary_qe"] = dict(_V4B_LITERARY_STATS)
    report["architecture"] = {
        "bulk": "GigaChat-3-Lightning fast draft + baseline deterministic safety",
        "parallel_context": "DeepSeek V4.1 Flash source style card + synopsis + chapter digest + IDF distant context",
        "semantic_qe": "all-segment explicit-verdict proposition-level DeepSeek V4.1 Flash",
        "deterministic_invariants": "coverage + questions + mixed-script + residual Latin + dynamic entity ledger + locked glossary",
        "repair": "diagnosis-guided minimal edit, semantic recheck, max 2 rounds",
        "hard_fallback": "sentence-decomposed DeepSeek translation only for unresolved QE failures",
        "literary_qe": "bounded risk + dialogue-typography candidates, rewrite only flagged passages",
        "literary_context": "local bilingual context + style/synopsis/chapter digest/IDF distant source context",
        "external_pro_judge": False,
        "full_paid_fallback": False,
    }
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def _enforce_final_report() -> None:
    if not v3.REPORT.exists():
        return
    report = json.loads(v3.REPORT.read_text("utf-8"))
    issue_counts = report.get("post_export_issue_counts") or {}
    hard = {key: value for key, value in issue_counts.items() if key.startswith("hard:") and int(value or 0) > 0}
    final_quality = report.get("final_quality") or {}
    final_hard = int(final_quality.get("hard_issues") or 0)
    if hard or final_hard:
        raise RuntimeError(f"v4b final export still has hard quality issues: post_export={hard} final_hard={final_hard}")


def main() -> None:
    _configure()
    try:
        v3.main()
    finally:
        _annotate_report()
    _enforce_final_report()


if __name__ == "__main__":
    main()
