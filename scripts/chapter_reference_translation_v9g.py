from __future__ import annotations

import json
import re

import chapter_reference_translation_v9f as v9f

v9e = v9f.v9e
v9c = v9f.v9c
v9 = v9f.v9

# v9g removes v9f's specialist waterfall.  The cheap micro-QE remains the
# primary filter.  Dedicated discourse repair is forced only for:
#   * explicit either/neither/both/other antecedent ambiguity; or
#   * an observed literalization pattern in the actual Russian draft.
# Everything else reaches the specialist only after real QE evidence.

_BASE_SPARSE_MICRO = v9f._BASE_SPARSE_MICRO
_BASE_SPECIALISTS = v9f._BASE_SPECIALISTS

_SHOULD_DO_RE = re.compile(r"\bi\s+should\s+do\b", re.I)
_WITH_IT_RE = re.compile(r"\bwith\s+it\b", re.I)
_COME_ON_RE = re.compile(r"\bcome\s+on\b", re.I)
_GO_ON_RE = re.compile(r"\bgo\s+on\b", re.I)
_UP_TO_IT_RE = re.compile(r"\bup\s+to\s+it\b", re.I)


def _literalization_kind(segment, translation: str) -> str:
    source = str(segment.text or "")
    target = str(translation or "").casefold()
    if _SHOULD_DO_RE.search(source) and re.search(r"\b(?:должен|должна|должны|должно|следует)\b", target):
        return "elliptical_modal"
    if _WITH_IT_RE.search(source) and re.search(r"\bс\s+(?:ним|ней|этим|тем)\b", target):
        return "preposition_idiom"
    if _COME_ON_RE.search(source) and re.search(r"\b(?:приходи|приходите|приходить)\b", target):
        return "come_on_literal"
    if _GO_ON_RE.search(source) and re.search(r"\b(?:иди|идите|пойди|пойдите)\b", target):
        return "go_on_literal"
    if _UP_TO_IT_RE.search(source) and re.search(r"\bдо\s+(?:него|неё|этого|того)\b", target):
        return "up_to_it_literal"
    return ""


def _synthetic_confirmed_rows(targets, translations, selected=None):
    rows = selected if selected is not None else targets
    out = []
    for segment in rows:
        text = str(segment.text or "")
        if v9e._explicit_referent_risk(text):
            out.append({
                "id": segment.id,
                "severity": "major",
                "confidence": 0.86,
                "code": "referent",
                "source_span": "explicit discourse referent",
                "target_span": "",
                "reason": "explicit either/neither/both/other construction requires antecedent resolution from local discourse",
                "repairability": "contextual",
            })
            continue
        kind = _literalization_kind(segment, translations.get(segment.id, ""))
        if kind:
            out.append({
                "id": segment.id,
                "severity": "major",
                "confidence": 0.88,
                "code": "idiom",
                "source_span": kind,
                "target_span": "",
                "reason": f"observed Russian draft matches a known literalization pattern: {kind}",
                "repairability": "contextual",
            })
    return out


def _parallel_micro_v9g(harness, targets, translations, selected=None):
    # v9f already patched the risk selector and source-span ownership guard, so
    # the base sparse call is cheap and cannot leak a neighbour's finding.
    rows = _BASE_SPARSE_MICRO(harness, targets, translations, selected=selected)
    if selected is None:
        existing = {(str(row.get("id")), str(row.get("code"))) for row in rows}
        for row in _synthetic_confirmed_rows(targets, translations):
            key = (row["id"], row["code"])
            if key not in existing:
                rows.append(row)
                existing.add(key)
    print(
        "[v9g-confirmed-discourse] "
        + json.dumps(
            {
                "selected_segments": sum(v9f._risk_segment_v9f(segment) for segment in (selected or targets)),
                "returned_findings": len(rows),
                "forced_confirmed": 0 if selected is not None else len(_synthetic_confirmed_rows(targets, translations)),
                "initial": selected is None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rows


def _specialist_candidates_v9g(harness, targets, segment, current, memory, findings):
    text = str(segment.text or "")
    codes = {str(row.get("code") or "") for row in findings}
    literal_kind = _literalization_kind(segment, current)

    # Explicit referents are rare and were a proven v9e failure.  Use the
    # dedicated antecedent resolver directly rather than spawning the generic
    # two-candidate + fresh-candidate fan-out first.
    if v9e._explicit_referent_risk(text):
        dedicated = v9f._referent_candidate(harness, targets, segment, current, memory)
        return [dedicated] if dedicated else []

    # For opaque ellipsis/idiom, call the specialist only after QE confirms an
    # idiom/dialogue defect, or when the actual RU draft shows a known literal
    # artifact.  This is the main latency fix over v9f.
    confirmed_opaque = v9f._opaque_short_risk(text) and bool(codes & {"idiom", "dialogue", "relation", "voice"})
    if literal_kind or confirmed_opaque:
        dedicated = v9f._idiom_candidate(harness, targets, segment, current, memory)
        if dedicated:
            return [dedicated]

    return list(_BASE_SPECIALISTS(harness, targets, segment, current, memory, findings))[:2]


# Keep v9f's discourse-sensitive judge and formatter, but replace the routing
# points responsible for the specialist waterfall.
v9c._parallel_micro_audit = _parallel_micro_v9g
v9._specialist_candidates = _specialist_candidates_v9g
v9._judge_candidates = v9f._judge_candidates_v9f
v9._format_dialogue_v9 = v9f._format_dialogue_v9f


def _annotate_v9g() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9g-confirmed-discourse-routing",
        "discourse_audit": "cue-bearing sparse micro-QE with source-span ownership guard",
        "forced_referent_rescue": "explicit either/neither/both/other only",
        "idiom_rescue": "only confirmed QE defects or observed literalization in RU draft",
        "specialist_policy": "no synthetic opaque-risk waterfall",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9f.main()
    finally:
        _annotate_v9g()


if __name__ == "__main__":
    main()
