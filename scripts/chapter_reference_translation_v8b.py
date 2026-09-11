from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v6 as v6
import chapter_reference_translation_v7 as v7
import chapter_reference_translation_v8 as v8
import hybrid_reference_translation as hybrid


# v8b keeps v8 quality mechanisms but removes the sequential hard-tail waterfall.
# Specialized repair is tried first; fresh retranslation is only used for generic
# unresolved hard cases or when a catastrophic rescue remains fatally incomplete.

_V8B_STATS: dict = {}


def _process_one(harness, targets, translated, memory, sid: str, quantity: dict[str, str], catastrophic: set[str]):
    by_id = {segment.id: segment for segment in targets}
    segment = by_id[sid]
    original = str(translated.get(sid) or "")
    candidates: dict[str, str] = {"original": original}
    rescue_attempted = 0
    quantity_attempted = 0
    fresh_attempted = 0

    if sid in catastrophic:
        rescue_attempted = 1
        rescue = v8._obligation_rescue_v8(harness, targets, segment, memory)
        if rescue:
            candidates["rescue"] = rescue
        # Only pay for a generic fresh translation when the dedicated rescue did
        # not produce a clearly non-catastrophic candidate.
        if not rescue or v7._catastrophic(segment, rescue) or v7._quantity_risk(segment, rescue):
            fresh_attempted = 1
            fresh = v7._direct_candidate(harness, targets, segment, memory)
            if fresh:
                candidates["fresh"] = fresh
    elif sid in quantity:
        quantity_attempted = 1
        repaired = v8._quantity_repair(harness, targets, segment, original, memory, quantity[sid])
        if repaired:
            candidates["quantity_repair"] = repaired
    else:
        fresh_attempted = 1
        fresh = v7._direct_candidate(harness, targets, segment, memory)
        if fresh:
            candidates["fresh"] = fresh

    chosen_name, chosen, vectors = v8._select_candidate(harness, segment, candidates, memory, targets)
    return {
        "sid": sid,
        "original": original,
        "chosen_name": chosen_name,
        "chosen": chosen,
        "vectors": vectors,
        "rescue_attempted": rescue_attempted,
        "quantity_attempted": quantity_attempted,
        "fresh_attempted": fresh_attempted,
    }


def _v8b_semantic_qe_repair(harness, targets, translated, memory) -> dict:
    global _V8B_STATS
    base_stats = v6._semantic_qe_repair(harness, targets, translated, memory)
    by_id = {segment.id: segment for segment in targets}

    quantity: dict[str, str] = {}
    catastrophic: set[str] = set()
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        reason = v7._quantity_risk(segment, current)
        if reason:
            quantity[segment.id] = reason
        if v7._catastrophic(segment, current):
            catastrophic.add(segment.id)

    candidate_ids: list[str] = []
    for sid in [*sorted(catastrophic), *quantity.keys(), *sorted(v6._V6_REMAINING_HARD)]:
        if sid in by_id and sid not in candidate_ids:
            candidate_ids.append(sid)
    cap = max(1, int(os.getenv("BOOKAI_V8B_COUNCIL_MAX") or "8"))
    candidate_ids = candidate_ids[:cap]
    workers = max(1, min(4, int(os.getenv("BOOKAI_V8B_COUNCIL_WORKERS") or "4")))

    hybrid.progress({
        "phase": "v8b_hard_tail_start",
        "candidates": len(candidate_ids),
        "catastrophic": len(catastrophic),
        "quantity": len(quantity),
        "workers": workers,
    })

    results = []
    if candidate_ids:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, harness, targets, translated, memory, sid, quantity, catastrophic): sid
                for sid in candidate_ids
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"[v8b-hard-tail] id={sid} error={type(exc).__name__}", flush=True)
                    continue
                results.append(row)
                print("[v8b-hard-tail] " + json.dumps({
                    "id": sid,
                    "choice": row["chosen_name"],
                    "vectors": row["vectors"],
                }, ensure_ascii=False), flush=True)

    council_changed = rescue_accepted = quantity_accepted = 0
    rescue_attempted = quantity_attempted = fresh_attempted = 0
    for row in results:
        rescue_attempted += row["rescue_attempted"]
        quantity_attempted += row["quantity_attempted"]
        fresh_attempted += row["fresh_attempted"]
        sid = row["sid"]
        chosen = str(row["chosen"] or "")
        if chosen and chosen.strip() != str(row["original"] or "").strip():
            translated[sid] = chosen
            council_changed += 1
            rescue_accepted += int(row["chosen_name"] == "rescue")
            quantity_accepted += int(row["chosen_name"] == "quantity_repair")

    entity_fixes = dialogue_normalized = 0
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        fixed, count = v8._fix_near_entity_typos(segment, current, memory)
        entity_fixes += count
        fixed, changed = v8._normalize_dialogue_v8(segment, fixed)
        dialogue_normalized += changed
        translated[segment.id] = fixed

    hard_after, hard_reasons = v8._v8_source_only_hard(targets, translated, memory)
    v6._V6_REMAINING_HARD = set(hard_after)
    for sid, reason in hard_reasons.items():
        v6._V6_REASONS[sid] = reason

    _V8B_STATS = {
        **dict(base_stats),
        "quantity_flagged": len(quantity),
        "catastrophic_detected": len(catastrophic),
        "catastrophic_ids": sorted(catastrophic),
        "hard_tail_candidates": len(candidate_ids),
        "hard_tail_workers": workers,
        "rescue_attempted": rescue_attempted,
        "rescue_accepted": rescue_accepted,
        "quantity_repair_attempted": quantity_attempted,
        "quantity_repair_accepted": quantity_accepted,
        "fresh_attempted": fresh_attempted,
        "council_changed": council_changed,
        "entity_typo_fixes": entity_fixes,
        "dialogue_segments_normalized": dialogue_normalized,
        "remaining_hard_after_v8b": len(hard_after),
    }
    v8._V8_STATS = dict(_V8B_STATS)
    hybrid.progress({"phase": "v8b_quality_done", **_V8B_STATS})
    return dict(_V8B_STATS)


def _configure_v8b() -> None:
    v8._configure_v8()
    v3._semantic_short_repair = _v8b_semantic_qe_repair


def _annotate_v8b() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        **dict(report.get("architecture") or {}),
        "version": "quality-v8b-source-only-parallel-hard-tail",
        "hard_tail": "specialized-first, parallel, max-8",
        "gold_reference_available_to_pipeline": False,
    }
    report["v8b_stats"] = dict(_V8B_STATS)
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v8b()
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v7._annotate_v7()
        v8._annotate_v8()
        _annotate_v8b()
        v6._source_only_hard = v8._ORIGINAL_V6_HARD


if __name__ == "__main__":
    main()
