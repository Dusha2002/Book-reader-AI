from __future__ import annotations

import json

import chapter_reference_translation_v9p as v9p
import chapter_reference_translation_v9k as v9k

v9n = v9p.v9n
v9 = v9k.v9
v9c = v9k.v9c
v9j = v9k.v9j

# v9q is the production-style version of the Chapter Three regression fixes:
# the bounded v9p discourse/idiom specialist runs first, then the normal v9
# chapter-wide bilingual QE and targeted repair run unchanged.


def _quality_v9q(harness, targets, translated, memory):
    # v9p's selector/validation/prompt monkeypatches are installed at import.
    priority = v9n._priority_semantic_repair_v9n(harness, targets, translated, memory)
    v9k._RESOLVED_PRIORITY_IDS = set(priority.get("ids") or [])

    # Suppress only the synthetic routing hint for already repaired priority
    # items. The real bilingual QE and deterministic checks still audit them.
    v9c._parallel_micro_audit = v9k._parallel_micro_v9k
    v9c._parallel_audit_v9c.__globals__["_parallel_micro_audit"] = v9k._parallel_micro_v9k
    v9._parallel_audit = v9c._parallel_audit_v9c

    stats = dict(v9k._BASE_QUALITY(harness, targets, translated, memory) or {})
    stats["priority_prepass"] = priority
    stats["quality_mode"] = "v9p-priority-prepass-plus-full-v9-qe"
    return stats


def _annotate_v9q():
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9q-v9p-prepass-plus-full-qe",
        "priority_prepass": "quote-local ellipsis/with-it + third-person multi-referent semantic invariants",
        "general_qe": "normal v9 chapter-wide bilingual QE + targeted repair after priority prepass",
        "cross_chapter_memory": "v9 working/trusted self-TM persisted through shared cache/state",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    v9._configure_v9()
    v9.v3._semantic_short_repair = _quality_v9q
    try:
        v9.v3.main()
    finally:
        v9.v6._annotate_report()
        v9j._annotate_v9j()
        v9k._annotate_v9k()
        _annotate_v9q()


if __name__ == "__main__":
    main()
