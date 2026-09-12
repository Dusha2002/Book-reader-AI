from __future__ import annotations

import json
import re

import chapter_reference_translation_v9j as v9j

v9i = v9j.v9i
v9h = v9j.v9h
v9g = v9j.v9g
v9f = v9j.v9f
v9c = v9j.v9c
v9 = v9j.v9

# v9k: guaranteed priority service for the rare discourse/idiom tail.
# v9j detects the right cases, but they can still be crowded out by other
# multi-finding critical segments before BOOKAI_V9_REPAIR_MAX is applied.
# This module therefore runs one small batched semantic pre-pass over only the
# deterministic priority findings, before the normal chapter-wide QE.
# No gold/reference Russian translation is available to this pre-pass.

_BASE_QUALITY = v9._quality_v9
_RESOLVED_PRIORITY_IDS: set[str] = set()
_STRONG_MULTI_REFERENT = re.compile(r"\b(?:either|neither|both)\s+of\s+them\b", re.I)


def _priority_rows(targets, translated):
    rows = v9i._synthetic_confirmed_rows_v9i(targets, translated)
    # One issue per segment is enough for this specialist. The deterministic
    # detector is intentionally narrow: explicit referents or an observed RU
    # literalization pattern, never generic style preference.
    out = {}
    for row in rows:
        sid = str(row.get("id") or "")
        if sid and sid not in out:
            out[sid] = row
    return list(out.values())


def _priority_payload(targets, translated, rows):
    by_id = {segment.id: segment for segment in targets}
    payload = []
    for row in rows:
        sid = str(row.get("id") or "")
        segment = by_id.get(sid)
        if segment is None:
            continue
        payload.append({
            "id": sid,
            "issue": str(row.get("code") or ""),
            "detector_reason": str(row.get("reason") or ""),
            "source": segment.text,
            "current_ru": translated.get(sid, ""),
            "context": v9f._context_payload(targets, translated, segment),
        })
    return payload


def _priority_semantic_repair(harness, targets, translated, memory):
    global _RESOLVED_PRIORITY_IDS
    rows = _priority_rows(targets, translated)
    if not rows:
        return {"flagged": 0, "replaced": 0, "ids": []}

    payload = _priority_payload(targets, translated, rows)
    if not payload:
        return {"flagged": 0, "replaced": 0, "ids": []}

    system = """You are a narrow EN→RU literary PRIORITY SEMANTIC REPAIRER.
Every item was flagged by a deterministic detector for one of two proven high-risk classes:
1) REFERENT: explicit either/neither/both/the-other whose concrete antecedents must be recovered from nearby English discourse;
2) IDIOM/ELLIPSIS: the current Russian visibly matches a literalization pattern for a short conversational fragment.

SOURCE English is authoritative. BEFORE/AFTER Russian strings are only this system's imperfect hypotheses, never a gold/reference translation.
For REFERENT items, resolve the exact people/things first. Pay special attention to implicit answers to coordinated questions such as
"Are you married? Children?" followed by an elliptical answer: later "either of them" can refer to two discourse entities even when
only one is repeated in the current sentence. Never invent the speaker as an antecedent unless English requires it.
For IDIOM/ELLIPSIS items, recover pragmatic force from context before translating. Do not translate preposition+pronoun fragments
word-for-word, and do not turn elliptical modal/auxiliary replies into literal obligation.
Translate the WHOLE source segment, preserving all clauses, uncertainty, negation, attribution and tone. If current_ru is actually
semantically correct, keep it. Do not optimize general style outside the flagged defect.

Return ONLY JSON:
{"items":[{"id":"exact id","verdict":"keep|replace","antecedents":["..."],"meaning":"brief English/Russian-neutral gloss","translation":"complete Russian segment"}]}"""

    try:
        obj = v9.v8._complete_json(harness.gate, system, {"items": payload})
    except Exception as exc:
        print(f"[v9k-priority-prepass] error={type(exc).__name__}", flush=True)
        return {"flagged": len(payload), "replaced": 0, "ids": []}

    result_rows = obj.get("items") if isinstance(obj, dict) else []
    result_by_id = {
        str(item.get("id") or ""): item
        for item in (result_rows if isinstance(result_rows, list) else [])
        if isinstance(item, dict)
    }
    source_by_id = {segment.id: segment for segment in targets}
    finding_by_id = {str(row.get("id") or ""): row for row in rows}
    replaced = []

    for sid, finding in finding_by_id.items():
        segment = source_by_id.get(sid)
        item = result_by_id.get(sid)
        if segment is None or item is None:
            continue
        candidate = v9._norm_text(item.get("translation") or "")
        current = v9._norm_text(translated.get(sid, ""))
        if str(item.get("verdict") or "").lower() != "replace" or not candidate or candidate == current:
            continue
        if v9._fatal_count(segment, candidate, memory) > v9._fatal_count(segment, current, memory):
            continue

        code = str(finding.get("code") or "")
        # The dedicated candidate is not accepted if the exact literalization
        # detector that triggered the repair still fires afterwards.
        if code == "idiom" and v9g._literalization_kind(segment, candidate):
            continue

        if code == "referent" and _STRONG_MULTI_REFERENT.search(segment.text or ""):
            antecedents = [v9._norm_text(x) for x in (item.get("antecedents") or []) if v9._norm_text(x)]
            # "either/neither/both of them" semantically requires a two-member
            # antecedent set. This is a generic discourse invariant, not a
            # book-specific expected translation.
            if len(antecedents) < 2:
                continue

        translated[sid] = candidate
        _RESOLVED_PRIORITY_IDS.add(sid)
        replaced.append(sid)
        print(
            "[v9k-priority-repair] "
            + json.dumps(
                {
                    "id": sid,
                    "code": code,
                    "meaning": v9._norm_text(item.get("meaning") or "")[:180],
                    "antecedents": item.get("antecedents") or [],
                    "before": current[:220],
                    "after": candidate[:220],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return {"flagged": len(payload), "replaced": len(replaced), "ids": replaced}


def _parallel_micro_v9k(harness, targets, translations, selected=None):
    rows = v9j._parallel_micro_v9j(harness, targets, translations, selected=selected)
    if selected is None and _RESOLVED_PRIORITY_IDS:
        rows = [row for row in rows if str(row.get("id") or "") not in _RESOLVED_PRIORITY_IDS]
    return rows


def _quality_v9k(harness, targets, translated, memory):
    global _RESOLVED_PRIORITY_IDS
    _RESOLVED_PRIORITY_IDS = set()
    priority = _priority_semantic_repair(harness, targets, translated, memory)

    # The normal general bilingual QE still audits the repaired text. We only
    # suppress the synthetic routing hint for ids already repaired by the
    # specialist; real QE/deterministic findings remain fully active.
    v9c._parallel_micro_audit = _parallel_micro_v9k
    v9c._parallel_audit_v9c.__globals__["_parallel_micro_audit"] = _parallel_micro_v9k
    v9._parallel_audit = v9c._parallel_audit_v9c
    stats = _BASE_QUALITY(harness, targets, translated, memory)
    stats = dict(stats or {})
    stats["priority_prepass"] = priority
    return stats


def _annotate_v9k() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9k-priority-semantic-prepass",
        "priority_prepass": "batched source-only referent + observed-literalization repair before general QE",
        "priority_acceptance": "fatal guard + literalization disappearance + two-member antecedent invariant for either/neither/both",
        "general_qe": "single bilingual chapter audit after priority repair",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    v9._configure_v9()
    v9.v3._semantic_short_repair = _quality_v9k
    try:
        v9.v3.main()
    finally:
        v9.v6._annotate_report()
        v9j._annotate_v9j()
        _annotate_v9k()


if __name__ == "__main__":
    main()
