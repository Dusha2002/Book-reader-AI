from __future__ import annotations

import json
import re

import chapter_reference_translation_v9l as v9l

v9k = v9l.v9k
v9i = v9l.v9i
v9g = v9l.v9g
v9f = v9l.v9f
v9 = v9l.v9

# v9m keeps the fast priority-only regression harness but fixes v9l's selector:
# "opaque short" was too broad and could create a huge specialist batch. Route
# only high-precision discourse/idiom classes relevant to demonstrated failures.

_MULTI_REFERENT = re.compile(r"\b(?:either|neither|both)\s+of\s+them\b", re.I)
_WITH_IT = re.compile(r"\bwith\s+it\b", re.I)
_ELLIPTICAL_AUX = re.compile(
    r"^\s*(?:[\"'“‘]\s*)?(?:i|you|he|she|we|they)\s+(?:should|would|could|can|do|does|did|may|might|must|need)\s*(?:[\"'”’.,!?…]|$)",
    re.I,
)


def _short_quoted_source(text: str, max_words: int = 7) -> bool:
    bodies = v9f._quoted_bodies(text)
    if not bodies:
        return False
    for body in bodies:
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", body)
        if len(words) <= max_words:
            return True
    return False


def _priority_rows_v9m(targets, translated):
    out = {}

    # Keep deterministic observed-literalization rows, but only force referent
    # repair for truly plural discourse forms.  This removes false positives like
    # "the other side of the brake".
    for row in v9i._synthetic_confirmed_rows_v9i(targets, translated):
        sid = str(row.get("id") or "")
        segment = next((x for x in targets if x.id == sid), None)
        if not sid or segment is None:
            continue
        if str(row.get("code") or "") == "referent" and not _MULTI_REFERENT.search(str(segment.text or "")):
            continue
        out[sid] = dict(row)

    # Source-side backstop: catch the class even when GigaChat made a different
    # calque that the RU literalization regex does not know (e.g. "при этом").
    for segment in targets:
        source = str(segment.text or "")
        if segment.id in out:
            continue
        source_risk = (
            (_WITH_IT.search(source) and _short_quoted_source(source, 7))
            or (_ELLIPTICAL_AUX.search(source) and _short_quoted_source(source, 7))
        )
        if source_risk:
            out[segment.id] = {
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.98,
                "code": "idiom",
                "source_span": "high-precision short dialogue idiom/ellipsis",
                "target_span": "",
                "reason": "high-precision source construction requires pragmatic resolution",
                "repairability": "contextual",
            }

    # Hard bound prevents regression runs from ever turning back into a broad QA
    # pass. Deterministic/current-literal rows come first, then source backstops.
    rows = list(out.values())
    rows.sort(key=lambda r: (0 if str(r.get("reason") or "").startswith("observed Russian") else 1, str(r.get("id") or "")))
    cap = 8
    selected = rows[:cap]
    print("[v9m-priority-selector] " + json.dumps({
        "selected": len(selected),
        "ids": [row["id"] for row in selected],
        "referents": sum(str(row.get("code")) == "referent" for row in selected),
        "idioms": sum(str(row.get("code")) == "idiom" for row in selected),
    }, ensure_ascii=False), flush=True)
    return selected


# v9l's repair function resolves this selector through its module globals.
v9l._priority_rows_v9l = _priority_rows_v9m


def main():
    v9l.main()
    report = v9.v3.REPORT
    if report.exists():
        try:
            data = json.loads(report.read_text("utf-8"))
            data["architecture"] = {
                **dict(data.get("architecture") or {}),
                "version": "quality-v9m-bounded-priority-regression",
                "priority_selector": "observed literalization + either/neither/both-of-them + short with-it/elliptical-aux only",
                "priority_cap": 8,
                "gold_reference_available_to_pipeline": False,
            }
            report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
