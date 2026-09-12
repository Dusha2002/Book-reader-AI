from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9s as v9s

v9r = v9s.v9r
v9 = v9s.v9
v8 = v9s.v8
v6 = v9s.v6
v3 = v9s.v3

# v9t keeps v9s publication semantics but removes its two chapter-wide
# post-v9r re-audits. v9r already audits every segment and re-audits accepted
# general repairs. Only semantic-lock/restored/formatter-changed segments can
# make those findings stale. v9t therefore refreshes that delta, independently
# verifies all remaining blocking semantic findings, repairs confirmed blockers
# in parallel, and re-audits only the repaired delta. Deterministic invariants
# and thread/entity hints are still rebuilt over the whole chapter because they
# are cheap and deterministic.
#
# No Russian reference/gold translation is used by this pipeline.

_BASE_QUALITY = v9s._BASE_QUALITY
_V9T_STATS: dict[str, Any] = {}
_DETERMINISTIC_CODES = {"coverage", "raw_english", "material", "number", "entity", "thread"}


def _blocking(row: dict[str, Any]) -> bool:
    return v9s._blocking(row)


def _semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("code") or "") not in _DETERMINISTIC_CODES]


def _minor_semantic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in _semantic_rows(rows) if not _blocking(row)]


def _blocking_semantic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in _semantic_rows(rows) if _blocking(row)]


def _selected_audit(harness, targets, translated, memory, ids: set[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    selected = [segment for segment in targets if segment.id in ids]
    if not selected:
        return []
    return v9._parallel_audit(harness, targets, translated, memory, selected=selected)


def _verify_parallel(harness, targets, translated, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run v9s' independent verifier concurrently over disjoint id groups."""
    candidates = [
        dict(row)
        for row in rows
        if _blocking(row) and str(row.get("code") or "") not in {"entity", "thread", "priority_invariant"}
    ]
    if not candidates:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(str(row.get("id") or ""), []).append(row)
    groups = list(grouped.values())
    workers = max(1, min(5, int(os.getenv("BOOKAI_V9T_VERIFY_WORKERS") or "4"), len(groups)))
    if workers <= 1 or len(groups) <= 3:
        return v9s._verify_llm_findings(harness, targets, translated, candidates)

    # Greedy balancing by payload size; keep all allegations for one segment
    # together so the verifier has a coherent local decision.
    bins: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    weights = [0] * workers
    for group in sorted(groups, key=lambda g: sum(len(str(x)) for x in g), reverse=True):
        idx = min(range(workers), key=lambda i: weights[i])
        bins[idx].extend(group)
        weights[idx] += sum(len(str(x)) for x in group)

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9t-verify") as pool:
        futures = [
            pool.submit(v9s._verify_llm_findings, harness, targets, translated, batch)
            for batch in bins if batch
        ]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception as exc:
                print(f"[v9t-verifier] error={type(exc).__name__}", flush=True)
    return out


def _replace_semantic_for_ids(base_map, refreshed_rows, ids: set[str]):
    out = {sid: [dict(row) for row in rows] for sid, rows in base_map.items()}
    refresh_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in refreshed_rows:
        refresh_by_id.setdefault(str(row.get("id") or ""), []).append(dict(row))
    for sid in ids:
        # Keep deterministic/manual priority invariants until a real repair
        # changes the candidate; all ordinary semantic findings are replaced.
        priority = [
            dict(row) for row in out.get(sid, [])
            if str(row.get("code") or "") == "priority_invariant"
        ]
        out[sid] = priority + _semantic_rows(refresh_by_id.get(sid, []))
    return out


def _verify_map_blockers(harness, targets, translated, semantic_map):
    all_blocking = []
    priority = []
    for rows in semantic_map.values():
        for row in rows:
            if not _blocking(row):
                continue
            if str(row.get("code") or "") == "priority_invariant":
                priority.append(dict(row))
            else:
                all_blocking.append(dict(row))
    verified = _verify_parallel(harness, targets, translated, all_blocking)
    verified_keys = {(str(row.get("id") or ""), str(row.get("code") or "")) for row in verified}

    out: dict[str, list[dict[str, Any]]] = {}
    for sid, rows in semantic_map.items():
        keep = []
        for row in rows:
            if not _blocking(row):
                keep.append(dict(row))
                continue
            if str(row.get("code") or "") == "priority_invariant":
                keep.append(dict(row))
                continue
            if (str(row.get("id") or sid), str(row.get("code") or "")) in verified_keys:
                # Use the verifier's calibrated row where possible.
                match = next(
                    (x for x in verified if str(x.get("id") or "") == str(row.get("id") or sid)
                     and str(x.get("code") or "") == str(row.get("code") or "")),
                    row,
                )
                keep.append(dict(match))
        if keep:
            out[sid] = keep
    return out, len(all_blocking), len(verified), len(priority)


def _rebuild_full_map(targets, translated, memory, semantic_map):
    det_rows = [
        row for segment in targets
        for row in v9s._deterministic_v9s(segment, translated.get(segment.id, ""), memory)
    ]
    thread_rows, thread_index = v9s._thread_v9s(targets, translated, memory)
    det_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in [*det_rows, *thread_rows]:
        det_by_id.setdefault(str(row.get("id") or ""), []).append(dict(row))

    finding_map: dict[str, list[dict[str, Any]]] = {}
    for segment in targets:
        rows = [*semantic_map.get(segment.id, []), *det_by_id.get(segment.id, [])]
        if rows:
            finding_map[segment.id] = v9._merge_findings(rows).get(segment.id, rows)
    score_map = {segment.id: v9._score(finding_map.get(segment.id, [])) for segment in targets}
    return finding_map, score_map, {
        "deterministic": len(det_rows),
        "thread_hints": len(thread_rows),
        "thread_terms": len(thread_index),
    }


def _tail_repair_parallel(harness, targets, translated, memory, finding_map, score_map):
    by_id = {segment.id: segment for segment in targets}
    repair_ids = [
        segment.id for segment in targets
        if any(
            _blocking(row) and str(row.get("code") or "") not in {"entity", "thread"}
            for row in finding_map.get(segment.id, [])
        )
    ]
    repair_ids.sort(key=lambda sid: score_map.get(sid, 100.0))
    cap = max(0, int(os.getenv("BOOKAI_V9S_TAIL_REPAIR_MAX") or "28"))
    repair_ids = repair_ids[:cap]
    workers = max(1, min(6, int(os.getenv("BOOKAI_V9T_TAIL_WORKERS") or "6"), len(repair_ids) or 1))

    def repair_one(sid: str):
        segment = by_id[sid]
        current = str(translated.get(sid) or "")
        rows = [
            row for row in finding_map.get(sid, [])
            if _blocking(row) and str(row.get("code") or "") not in {"entity", "thread"}
        ]
        candidates = v9._specialist_candidates(harness, targets, segment, current, memory, rows)
        if not candidates:
            return sid, ""
        chosen = v9._judge_candidates(harness, targets, segment, [current, *candidates], memory)
        if not chosen or v9._norm_text(chosen) == v9._norm_text(current):
            return sid, ""
        if v9._fatal_count(segment, chosen, memory) > v9._fatal_count(segment, current, memory):
            return sid, ""
        return sid, chosen

    changed: list[str] = []
    results: dict[str, str] = {}
    if repair_ids:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9t-tail") as pool:
            futures = [pool.submit(repair_one, sid) for sid in repair_ids]
            for future in as_completed(futures):
                sid, value = future.result()
                if value:
                    results[sid] = value
    for sid in repair_ids:
        value = results.get(sid)
        if value:
            translated[sid] = value
            changed.append(sid)
    return len(changed), changed, repair_ids


def _quality_v9t(harness, targets, translated, memory):
    global _V9T_STATS

    v9._MATERIALS["coal"] = ("угл",)
    v9._deterministic_findings = v9s._deterministic_v9s
    v9._thread_findings = v9s._thread_v9s
    v8._fix_near_entity_typos = v9s._entity_fix_v9s
    v9._format_dialogue_v9 = v9s._format_dialogue_v9s

    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})
    base_map = {sid: [dict(row) for row in rows] for sid, rows in v9._V9_FINAL_FINDINGS.items()}

    # Refresh only the text that can be stale after v9r's post-QE semantic
    # locks/post-QE priority pass or deterministic typography cleanup.
    stale_ids = set(stats.get("priority_semantic_locks") or [])
    stale_ids.update(stats.get("priority_restored_after_general_qe") or [])
    stale_ids.update((stats.get("priority_post_qe_repair") or {}).get("ids") or [])
    stale_ids.update(stats.get("priority_post_qe_unresolved") or [])

    sanitized = 0
    for segment in targets:
        before = str(translated.get(segment.id) or "")
        value, changed = v9s._format_dialogue_v9s(segment, before)
        translated[segment.id] = value
        if changed:
            stale_ids.add(segment.id)
            sanitized += 1

    refreshed = _selected_audit(harness, targets, translated, memory, stale_ids)
    semantic_map = {sid: _semantic_rows(rows) for sid, rows in base_map.items()}
    semantic_map = _replace_semantic_for_ids(semantic_map, refreshed, stale_ids)
    semantic_map, alleged_before, verified_before, priority_blockers = _verify_map_blockers(
        harness, targets, translated, semantic_map
    )

    finding_map, score_map, det_meta = _rebuild_full_map(targets, translated, memory, semantic_map)
    tail_count, tail_ids, tail_selected = _tail_repair_parallel(
        harness, targets, translated, memory, finding_map, score_map
    )

    # Tail repair is the only remaining semantic mutation. Re-audit exactly
    # those segments, not the whole chapter, then rebuild cheap invariants.
    for sid in tail_ids:
        segment = next((x for x in targets if x.id == sid), None)
        if segment is not None:
            value, _ = v9s._format_dialogue_v9s(segment, translated.get(sid, ""))
            translated[sid] = value

    if tail_ids:
        tail_fresh = _selected_audit(harness, targets, translated, memory, set(tail_ids))
        semantic_map = _replace_semantic_for_ids(semantic_map, tail_fresh, set(tail_ids))
        # A successful repair supersedes an old manual priority invariant.
        for sid in tail_ids:
            semantic_map[sid] = [
                row for row in semantic_map.get(sid, [])
                if str(row.get("code") or "") != "priority_invariant"
            ]
        semantic_map, alleged_after, verified_after, priority_after = _verify_map_blockers(
            harness, targets, translated, semantic_map
        )
    else:
        alleged_after = verified_after = 0
        priority_after = priority_blockers

    final_map, final_scores, final_det_meta = _rebuild_full_map(targets, translated, memory, semantic_map)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)

    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)
    entity_hints = sum(
        1 for rows in final_map.values() for row in rows
        if row.get("code") in {"entity", "thread"}
    )
    mean_score = round(sum(final_scores.values()) / max(1, len(final_scores)), 2)
    _V9T_STATS = {
        "incremental_final_qe": True,
        "stale_segments_reaudited": len(stale_ids),
        "stale_segment_ids": sorted(stale_ids),
        "typography_segments_sanitized": sanitized,
        "blocking_allegations_verified_before_tail": alleged_before,
        "blocking_confirmed_before_tail": verified_before,
        "priority_blockers_before_tail": priority_blockers,
        "tail_repair_selected": len(tail_selected),
        "tail_repair_changed": tail_count,
        "tail_repair_ids": tail_ids,
        "blocking_allegations_verified_after_tail": alleged_after,
        "blocking_confirmed_after_tail": verified_after,
        "priority_blockers_after_tail": priority_after,
        "deterministic_before_tail": det_meta,
        "final_deterministic": final_det_meta,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "entity_thread_hints_nonblocking": entity_hints,
        "mean_quality_score": mean_score,
        "severity_counts": dict(counts),
        "quality_mode": "v9r-full-qe+delta-refresh+parallel-consensus+parallel-tail+delta-reaudit",
    }
    stats.update(_V9T_STATS)
    print("[bookai-v9t] " + json.dumps(_V9T_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9t() -> None:
    report = v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9t-incremental-final-qe",
        "gold_reference_available_to_pipeline": False,
        "performance": "reuse v9r chapter-wide QE; refresh only semantic-lock/formatter/tail deltas",
        "qe_consensus": "parallel independent verification of blocking findings; no duplicate full-chapter final audits",
        "tail_repair": "parallel bounded specialist repairs followed by delta-only re-audit",
        "publication_gate": "confirmed critical or major semantic defect still blocks completion",
    }
    data["v9t_stats"] = dict(_V9T_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    v9._configure_v9()
    v9._deterministic_findings = v9s._deterministic_v9s
    v9._thread_findings = v9s._thread_v9s
    v8._fix_near_entity_typos = v9s._entity_fix_v9s
    v9._format_dialogue_v9 = v9s._format_dialogue_v9s
    v8._normalize_dialogue_v8 = v9s._format_dialogue_v9s
    v3._semantic_short_repair = _quality_v9t
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v9r.v9j._annotate_v9j()
        v9r.v9k._annotate_v9k()
        v9r._annotate_v9r()
        v9s._annotate_v9s()
        _annotate_v9t()


if __name__ == "__main__":
    main()
