from __future__ import annotations

import json

import chapter_reference_translation_v9q as v9q

v9p = v9q.v9p
v9n = v9q.v9n
v9m = v9p.v9m
v9k = v9q.v9k
v9c = v9q.v9c
v9j = v9q.v9j
v9 = v9q.v9
v6 = v9.v6

# v9r closes the regression discovered in the full 1→2→3 run: a priority
# discourse repair could be correct before chapter-wide QE and later be
# rewritten into a weaker/literal form. Validated priority repairs are now
# semantic locks for the remainder of that chapter. Priority items that fail
# the pre-pass receive one fresh bounded repair AFTER normal v9 QE. No gold or
# reference Russian text is used.


def _filtered_priority_repair(harness, targets, translated, memory, ids: set[str]):
    if not ids:
        return {"flagged": 0, "replaced": 0, "rejected": 0, "ids": [], "rejected_ids": []}
    base_selector = v9m._priority_rows_v9m

    def selector(ts, tr):
        return [row for row in base_selector(ts, tr) if str(row.get("id") or "") in ids]

    v9m._priority_rows_v9m = selector
    try:
        return v9n._priority_semantic_repair_v9n(harness, targets, translated, memory)
    finally:
        v9m._priority_rows_v9m = base_selector


def _mark_unresolved_priority(ids: set[str]) -> None:
    for sid in ids:
        reason = "publication-blocking discourse/idiom invariant remains unresolved after post-QE repair"
        v6._V6_REMAINING_HARD.add(sid)
        v6._V6_REASONS[sid] = reason
        rows = list(v9._V9_FINAL_FINDINGS.get(sid, []))
        if not any(str(row.get("code") or "") == "priority_invariant" for row in rows):
            rows.append({
                "id": sid,
                "severity": "critical",
                "confidence": 0.99,
                "code": "priority_invariant",
                "source_span": "",
                "target_span": "",
                "reason": reason,
                "repairability": "contextual",
            })
        v9._V9_FINAL_FINDINGS[sid] = rows
        v9._V9_SCORES[sid] = min(float(v9._V9_SCORES.get(sid, 100.0)), 20.0)


def _quality_v9r(harness, targets, translated, memory):
    # 1) Run the proven v9p source-only specialist before broad QE.
    pre = v9n._priority_semantic_repair_v9n(harness, targets, translated, memory)
    locked = {
        sid: v9._norm_text(translated.get(sid, ""))
        for sid in (pre.get("ids") or [])
        if v9._norm_text(translated.get(sid, ""))
    }
    pre_rejected = {str(sid) for sid in (pre.get("rejected_ids") or []) if sid}

    # 2) Run normal full v9 bilingual QE + targeted repairs.
    v9k._RESOLVED_PRIORITY_IDS = set(locked)
    v9c._parallel_micro_audit = v9k._parallel_micro_v9k
    v9c._parallel_audit_v9c.__globals__["_parallel_micro_audit"] = v9k._parallel_micro_v9k
    v9._parallel_audit = v9c._parallel_audit_v9c
    stats = dict(v9k._BASE_QUALITY(harness, targets, translated, memory) or {})

    # 3) Semantic lock: chapter-wide QE is not allowed to regress a candidate
    # that already passed the stricter discourse invariants.
    restored = []
    for sid, value in locked.items():
        if v9._norm_text(translated.get(sid, "")) != value:
            translated[sid] = value
            restored.append(sid)

    # 4) Items the pre-pass could not solve get a fresh attempt AFTER broad QE,
    # now with the broad-QE candidate as current_ru and the same strict v9p
    # acceptance invariants. Only those ids are sent, never the whole chapter.
    post = _filtered_priority_repair(harness, targets, translated, memory, pre_rejected)
    post_success = {str(sid) for sid in (post.get("ids") or []) if sid}
    unresolved = pre_rejected - post_success
    _mark_unresolved_priority(unresolved)

    stats["priority_prepass"] = pre
    stats["priority_semantic_locks"] = sorted(locked)
    stats["priority_restored_after_general_qe"] = sorted(restored)
    stats["priority_post_qe_repair"] = post
    stats["priority_post_qe_unresolved"] = sorted(unresolved)
    stats["quality_mode"] = "v9p-prepass+full-v9-qe+semantic-lock+post-qe-priority-repair"
    return stats


def _annotate_v9r():
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9r-priority-semantic-lock",
        "priority_semantic_lock": "validated source-only discourse repairs cannot be regressed by broad QE",
        "post_qe_priority_repair": "fresh bounded repair only for priority ids unresolved by pre-pass",
        "unresolved_priority_policy": "publication-blocking critical + excluded from trusted self-TM",
        "general_qe": "normal v9 chapter-wide bilingual QE + targeted repair",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    v9._configure_v9()
    v9.v3._semantic_short_repair = _quality_v9r
    try:
        v9.v3.main()
    finally:
        v9.v6._annotate_report()
        v9j._annotate_v9j()
        v9k._annotate_v9k()
        _annotate_v9r()


if __name__ == "__main__":
    main()
