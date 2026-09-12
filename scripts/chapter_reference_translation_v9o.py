from __future__ import annotations

import json
import re

import chapter_reference_translation_v9n as v9n

v9m = v9n.v9m
v9l = v9n.v9l
v9i = v9m.v9i
v9f = v9m.v9f
v9 = v9n.v9

# v9o fixes the remaining Chapter Three regression without phrase-specific
# hardcoding. Risk routing is now evaluated inside each quoted utterance rather
# than over the whole source segment, and short auxiliary/pro-verb ellipses such
# as "I should do" are treated as one generic discourse class.

_MULTI_REFERENT = re.compile(r"\b(?:either|neither|both)\s+of\s+them\b", re.I)
_WITH_IT_BODY = re.compile(r"\bwith\s+it\b", re.I)
_ELLIPTICAL_AUX_BODY = re.compile(
    r"^\s*(?:i|you|he|she|we|they)\s+"
    r"(?:should|would|could|can|do|does|did|may|might|must|need)"
    r"(?:\s+(?:do|have|be)(?:\s+(?:so|it|that))?)?\s*[,.!?…]*\s*$",
    re.I,
)


def _quoted_risk(source: str) -> bool:
    for body in v9f._quoted_bodies(source):
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", body)
        if len(words) > 7:
            continue
        if _WITH_IT_BODY.search(body) or _ELLIPTICAL_AUX_BODY.fullmatch(body.strip()):
            return True
    return False


def _priority_rows_v9o(targets, translated):
    out = {}
    by_id = {segment.id: segment for segment in targets}

    # Keep only genuinely plural referents from the deterministic detector.
    # Observed idiom literalizations stay eligible regardless of their exact RU
    # wording because they are already high-confidence failures.
    for row in v9i._synthetic_confirmed_rows_v9i(targets, translated):
        sid = str(row.get("id") or "")
        segment = by_id.get(sid)
        if not sid or segment is None:
            continue
        if str(row.get("code") or "") == "referent" and not _MULTI_REFERENT.search(str(segment.text or "")):
            continue
        out[sid] = dict(row)

    # Source-side backstop is quote-local. This avoids routing unrelated long
    # paragraphs merely because they contain some other "with it" later on.
    for segment in targets:
        if segment.id in out:
            continue
        source = str(segment.text or "")
        if _quoted_risk(source):
            out[segment.id] = {
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.99,
                "code": "idiom",
                "source_span": "quote-local short auxiliary/with-it discourse construction",
                "target_span": "",
                "reason": "high-precision quote-local construction requires pragmatic resolution",
                "repairability": "contextual",
            }

    rows = list(out.values())
    rows.sort(key=lambda r: (0 if str(r.get("reason") or "").startswith("observed Russian") else 1, str(r.get("id") or "")))
    selected = rows[:8]
    print("[v9o-priority-selector] " + json.dumps({
        "selected": len(selected),
        "ids": [row["id"] for row in selected],
        "referents": sum(str(row.get("code")) == "referent" for row in selected),
        "idioms": sum(str(row.get("code")) == "idiom" for row in selected),
    }, ensure_ascii=False), flush=True)
    return selected


# v9n calls v9m._priority_rows_v9m dynamically, so replacing it here changes
# only the selector while preserving v9n's semantic invariants and bounded retry.
v9m._priority_rows_v9m = _priority_rows_v9o
v9l._priority_rows_v9l = _priority_rows_v9o


def main():
    v9n.main()
    report = v9.v3.REPORT
    if report.exists():
        try:
            data = json.loads(report.read_text("utf-8"))
            data["architecture"] = {
                **dict(data.get("architecture") or {}),
                "version": "quality-v9o-quote-local-ellipsis-routing",
                "priority_selector": "quote-local observed literalization + plural referents + short with-it/auxiliary-proverb ellipsis",
                "elliptical_auxiliary": "generic short subject+auxiliary(+do/have/be/pro-form), not phrase-specific",
                "gold_reference_available_to_pipeline": False,
            }
            report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
