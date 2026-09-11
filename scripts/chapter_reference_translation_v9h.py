from __future__ import annotations

import json

import chapter_reference_translation_v9g as v9g

v9f = v9g.v9f
v9e = v9g.v9e
v9c = v9g.v9c
v9 = v9g.v9

# v9h removes the remaining repair waterfall.
# One confirmed defect -> one targeted candidate -> deterministic fatal guard ->
# existing bilingual re-QE decides acceptance.  There is no candidate-ranking LLM.
# This preserves a second quality check while removing dozens of judge/fresh calls.

_FACTUAL_MAJOR_CODES = {
    "omission", "addition", "relation", "referent", "number", "material",
    "entity", "term", "coverage", "raw_english",
}
_STYLE_MAJOR_CODES = {"idiom", "dialogue", "voice", "irony", "grammar", "calque", "other"}


def _needs_repair_v9h(rows) -> bool:
    for row in rows:
        severity = str(row.get("severity") or "")
        confidence = float(row.get("confidence") or 0.0)
        code = str(row.get("code") or "")
        reason = str(row.get("reason") or "")
        if severity == "critical" and confidence >= 0.55:
            return True
        if severity != "major":
            continue
        # Deterministic/synthetic high-value signals should not need a stylistic
        # confidence threshold.
        if "observed Russian draft matches a known literalization pattern" in reason:
            return True
        if code in _FACTUAL_MAJOR_CODES and confidence >= 0.82:
            return True
        if code in _STYLE_MAJOR_CODES and confidence >= 0.91:
            return True
        if confidence >= 0.95:
            return True
    return False


def _specialist_candidates_v9h(harness, targets, segment, current, memory, findings):
    text = str(segment.text or "")
    codes = {str(row.get("code") or "") for row in findings}
    literal_kind = v9g._literalization_kind(segment, current)

    if v9e._explicit_referent_risk(text):
        candidate = v9f._referent_candidate(harness, targets, segment, current, memory)
        return [candidate] if candidate else []

    confirmed_opaque = v9f._opaque_short_risk(text) and bool(codes & {"idiom", "dialogue", "relation", "voice"})
    if literal_kind or confirmed_opaque:
        candidate = v9f._idiom_candidate(harness, targets, segment, current, memory)
        return [candidate] if candidate else []

    # Use the original surgical editor, not v9c's independent-fresh-candidate
    # expansion.  Even if it emits two options, keep only the first: re-QE is the
    # selector/acceptance layer in v9h.
    values = list(v9c._BASE_SPECIALIST_CANDIDATES(harness, targets, segment, current, memory, findings))
    return values[:1]


def _judge_candidates_v9h(harness, targets, segment, candidates, memory):
    values = [v9._norm_text(value) for value in candidates if v9._norm_text(value)]
    if not values:
        return ""
    current = values[0]
    if len(values) == 1:
        return current
    proposed = values[1]
    current_fatal = v9._fatal_count(segment, current, memory)
    proposed_fatal = v9._fatal_count(segment, proposed, memory)
    if proposed_fatal > current_fatal:
        return current
    # Do not spend another LLM call choosing between current and proposed.
    # _quality_v9 re-audits every changed segment and accepts only a material
    # score/fatal improvement, so this remains fail-safe rather than blind.
    return proposed


v9._needs_repair = _needs_repair_v9h
v9._specialist_candidates = _specialist_candidates_v9h
v9._judge_candidates = _judge_candidates_v9h
# Keep v9g's confirmed-only discourse routing.
v9c._parallel_micro_audit = v9g._parallel_micro_v9g


def _annotate_v9h() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9h-single-candidate-re-qe",
        "repair_policy": "critical/factual first; style-major requires >=0.91 confidence",
        "candidate_policy": "one targeted candidate per confirmed defect",
        "candidate_judge": "deterministic fatal guard; no LLM ranker",
        "acceptance": "existing bilingual re-QE must show improvement before commit",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9g.main()
    finally:
        _annotate_v9h()


if __name__ == "__main__":
    main()
