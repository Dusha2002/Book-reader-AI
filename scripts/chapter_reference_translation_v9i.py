from __future__ import annotations

import json
import re

import chapter_reference_translation_v9h as v9h

v9g = v9h.v9g
v9f = v9g.v9f
v9e = v9g.v9e
v9c = v9g.v9c
v9 = v9g.v9

# v9i fixes priority starvation discovered in v9h.  A known literalization in
# the actual RU draft, or a short/dialogue explicit antecedent construction, is
# semantic corruption rather than optional style.  Mark only those cases as
# synthetic critical so they cannot be crowded out by the bounded repair queue.
# The synthetic critical is not re-added on candidate re-QE, so a successful
# dedicated repair demonstrates a fatal-count reduction and is accepted.

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_STRONG_REFERENT_RE = re.compile(
    r"\b(?:either\s+of\s+them|neither\s+of\s+them|both\s+of\s+them|the\s+other)\b",
    re.I,
)


def _must_referent_rescue(segment) -> bool:
    text = str(segment.text or "")
    if not _STRONG_REFERENT_RE.search(text):
        return False
    words = _WORD_RE.findall(text)
    # Long expository sentences such as "either of them would..." are usually
    # explicit enough for normal QE.  Short dialogue/replies are the proven hard
    # case because Russian must recover concrete antecedents from discourse.
    return len(words) <= 80 or bool(v9f._quoted_bodies(text))


def _synthetic_confirmed_rows_v9i(targets, translations, selected=None):
    rows = selected if selected is not None else targets
    out = []
    for segment in rows:
        if _must_referent_rescue(segment):
            out.append({
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.98,
                "code": "referent",
                "source_span": "explicit short discourse referent",
                "target_span": "",
                "reason": "short either/neither/both/the-other construction requires exact antecedent resolution before publication",
                "repairability": "contextual",
            })
            continue
        kind = v9g._literalization_kind(segment, translations.get(segment.id, ""))
        if kind:
            out.append({
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.97,
                "code": "idiom",
                "source_span": kind,
                "target_span": "",
                "reason": f"observed Russian draft matches a known literalization pattern: {kind}",
                "repairability": "contextual",
            })
    return out


# _parallel_micro_v9g resolves this helper from v9g module globals at runtime.
v9g._synthetic_confirmed_rows = _synthetic_confirmed_rows_v9i
v9c._parallel_micro_audit = v9g._parallel_micro_v9g


def _annotate_v9i() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9i-priority-discourse-rescue",
        "priority_rescue": "short strong referents + observed RU literalization are synthetic critical",
        "long_referents": "normal QE unless actual defect is observed",
        "candidate_policy": "v9h single candidate + deterministic guard + re-QE",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9h.main()
    finally:
        _annotate_v9i()


if __name__ == "__main__":
    main()
