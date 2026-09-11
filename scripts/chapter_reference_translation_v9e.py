from __future__ import annotations

import json
import os
import re

import chapter_reference_translation_v9d as v9d

v9c = v9d.v9c
v9 = v9d.v9


# v9e keeps v9d's semantic/format improvements but makes discourse QA sparse.
# Document-level dependencies are sparse: do not audit every dialogue paragraph.
# Trigger on explicit ambiguity markers OR very short quoted utterances where
# English ellipsis/idiomatic pragmatics is most likely. Then confidence-gate and
# cap the micro findings before they can consume repair/candidate budget.

# Do not mistake apostrophes in man's / Valens' / didn't for dialogue delimiters.
_QUOTED_SPAN_RE = re.compile(
    r"(?<![A-Za-z])(?:'([^'\n]{1,180})'|\"([^\"\n]{1,180})\"|“([^”\n]{1,180})”|‘([^’\n]{1,180})’)(?![A-Za-z])"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_BASE_PARALLEL_MICRO = v9c._parallel_micro_audit

# Broad pronoun spotting was too expensive and produced many false positives.
# These are compact lexical constructions whose interpretation genuinely depends
# on discourse context and that have already caused observed translation errors.
_EXPLICIT_REFERENT_RE = re.compile(
    r"\b(?:either|neither|both)\b"
    r"|\b(?:one|each|any|none)\s+of\s+(?:them|us|you)\b"
    r"|\b(?:the|that|this)\s+other\b"
    r"|\b(?:the\s+)?other\s+one\b",
    re.I,
)


def _quoted_body(match: re.Match) -> str:
    return next((part for part in match.groups() if part is not None), "")


def _short_utterance_risk(text: str) -> bool:
    max_words = max(3, int(os.getenv("BOOKAI_V9E_SHORT_UTTERANCE_WORDS") or "6"))
    for match in _QUOTED_SPAN_RE.finditer(str(text or "")):
        words = _WORD_RE.findall(_quoted_body(match))
        if words and len(words) <= max_words:
            return True
    return False


def _explicit_referent_risk(text: str) -> bool:
    return bool(_EXPLICIT_REFERENT_RE.search(str(text or "")))


def _risk_segment_v9e(segment) -> bool:
    text = str(segment.text or "")
    return _explicit_referent_risk(text) or _short_utterance_risk(text)


def _parallel_micro_v9e(harness, targets, translations, selected=None):
    rows = _BASE_PARALLEL_MICRO(harness, targets, translations, selected=selected)
    threshold = max(0.0, min(1.0, float(os.getenv("BOOKAI_V9E_MICRO_CONFIDENCE") or "0.90")))
    cap = max(1, int(os.getenv("BOOKAI_V9E_MICRO_FINDINGS_MAX") or "12"))
    accepted = [row for row in rows if float(row.get("confidence") or 0.0) >= threshold]
    priority = {"critical": 0, "major": 1}
    accepted.sort(
        key=lambda row: (
            priority.get(str(row.get("severity") or "major"), 2),
            -float(row.get("confidence") or 0.0),
        )
    )
    kept = accepted[:cap]
    print(
        "[v9e-sparse-discourse] "
        + json.dumps(
            {
                "raw_findings": len(rows),
                "high_confidence": len(accepted),
                "kept": len(kept),
                "threshold": threshold,
                "cap": cap,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return kept


# v9c's parallel audit resolves both helpers from its own module globals.
v9c._risk_segment = _risk_segment_v9e
v9c._parallel_micro_audit = _parallel_micro_v9e


def _annotate_v9e() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9e-sparse-discourse",
        "discourse_audit": "explicit ambiguity markers + very-short utterances only",
        "micro_findings": "confidence-gated and capped before repair routing",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9d.main()
    finally:
        _annotate_v9e()


if __name__ == "__main__":
    main()
