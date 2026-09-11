from __future__ import annotations

import json
import os
import re

import chapter_reference_translation_v9c as v9c

v9 = v9c.v9


# v9d is a narrow structural hardening of v9c. English-style dialogue quotes and
# ordinary quoted phrases are different punctuation modes and must never share a
# destructive cleanup rule. It also detects direct speech that starts after an
# author/narration sentence inside the same paragraph.

_DIALOGUE_IN_SOURCE = re.compile(r"(^|[.!?…]\s+)[\"'“‘]", re.M)


def _source_has_dialogue(text: str) -> bool:
    return bool(_DIALOGUE_IN_SOURCE.search(str(text or "")))


def _risk_segment_v9d(segment) -> bool:
    text = str(segment.text or "")
    if v9._extract_invariants(text).get("referent_risk"):
        return True
    return _source_has_dialogue(text) and len(text) <= max(180, int(os.getenv("BOOKAI_V9C_DIALOGUE_MAX_CHARS") or "520"))


def _format_dialogue_v9d(segment, text: str):
    before = v9._norm_text(text)
    value = before.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r",\s*,+", ",", value)

    if _source_has_dialogue(segment.text):
        # Only in direct-speech mode may English/Russian quote delimiters be
        # interpreted as dialogue boundaries. Never apply these rules to ordinary
        # quoted phrases in narration.
        if re.match(r"^\s*[\"'«]", value):
            value = re.sub(r"^\s*[\"'«]\s*", "— ", value, count=1)
        value = re.sub(r"—\s*[\"'«]\s*(?=[А-ЯЁ])", "— ", value)
        value = re.sub(r"(?<=[.!?…])\s*[\"'«]\s*(?=[А-ЯЁ])", " — ", value)
        value = re.sub(r"([,!?….])\s*[\"'»](?=\s*(?:—|-|$))", r"\1", value)
        value = re.sub(r"[\"'»]\s*$", "", value)
    else:
        # Narration/citation mode: paired straight quotes are semantic quotation,
        # not dialogue punctuation. Convert only balanced pairs.
        value = re.sub(r"['\"]([^'\"\n]{1,240})['\"]", r"«\1»", value)

    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    value = v9c._remove_unmatched_guillemets(value)
    value = re.sub(r",\s*,+", ",", value)
    return value, int(value != before)


# v9c micro-audit resolves its risk function from the v9c module at runtime.
v9c._risk_segment = _risk_segment_v9d
# v9 quality/finalizer resolves the formatter from v9 module globals.
v9._format_dialogue_v9 = _format_dialogue_v9d


def _annotate_v9d() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9d-focused-discourse-structural-dialogue",
        "dialogue_detection": "source structural direct speech at paragraph start or after sentence boundary",
        "quote_formatting": "separate direct-speech and narrative-quotation modes; lexically inert",
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9c.main()
    finally:
        _annotate_v9d()


if __name__ == "__main__":
    main()
