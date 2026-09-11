from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

import chapter_reference_translation_v9e as v9e

v9d = v9e.v9d
v9c = v9e.v9c
v9 = v9e.v9


# v9f: make discourse repair both cheaper and more precise.
#
# v9e proved that document context helps, but auditing every tiny quotation still
# consumed too much latency and a batched micro-auditor could attach an issue to
# the neighbouring segment. v9f therefore:
#   1) audits only explicit referent ambiguity and genuinely pragmatic/elliptical
#      short utterances, not every short quote;
#   2) rejects micro-QE findings whose claimed source span is not in that segment;
#   3) injects a cheap risk signal for a few high-risk constructions so they can
#      reach the candidate path even when the auditor misses them;
#   4) gives referent and idiom/ellipsis cases dedicated candidates and a
#      context-aware candidate judge;
#   5) removes mechanically stranded English dialogue apostrophes without
#      touching lexical content.
# No reference/gold Russian text is used anywhere in this module.

_BASE_MICRO_BATCH = v9c._micro_audit_batch
_BASE_SPARSE_MICRO = v9e._parallel_micro_v9e
_BASE_SPECIALISTS = v9._specialist_candidates
_BASE_JUDGE = v9._judge_candidates

_CUE_RE = re.compile(
    r"\b(?:do|does|did|should|would|could|can|may|might|must|need|"
    r"it|that|this|one|ones|them|him|her|mean|suppose|guess|reckon|seem|"
    r"rather|better|enough|too|so|not|no|yes)\b"
    r"|\b(?:with|at|of|for|to)\s+it\b"
    r"|\b(?:up\s+to|come\s+on|go\s+on|what\s+about|how\s+about)\b",
    re.I,
)
_OPAQUE_IDIOM_RE = re.compile(
    r"\b(?:with|at|of|for|to)\s+it\b|\b(?:come\s+on|go\s+on|up\s+to)\b",
    re.I,
)
_ELLIPTICAL_AUX_RE = re.compile(
    r"^\s*(?:i|you|he|she|we|they)\s+(?:do|does|did|should|would|could|can|may|might|must|need)\b",
    re.I,
)


def _quoted_bodies(text: str) -> list[str]:
    out: list[str] = []
    for match in v9e._QUOTED_SPAN_RE.finditer(str(text or "")):
        body = v9e._quoted_body(match).strip()
        if body:
            out.append(body)
    return out


def _pragmatic_short_risk(text: str) -> bool:
    max_words = max(3, int(os.getenv("BOOKAI_V9F_CUE_WORDS") or "5"))
    for body in _quoted_bodies(text):
        words = v9e._WORD_RE.findall(body)
        if words and len(words) <= max_words and _CUE_RE.search(body):
            return True
    return False


def _opaque_short_risk(text: str) -> bool:
    for body in _quoted_bodies(text):
        words = v9e._WORD_RE.findall(body)
        if not words or len(words) > 5:
            continue
        if _OPAQUE_IDIOM_RE.search(body) or _ELLIPTICAL_AUX_RE.search(body):
            return True
    return False


def _risk_segment_v9f(segment) -> bool:
    text = str(segment.text or "")
    return v9e._explicit_referent_risk(text) or _pragmatic_short_risk(text)


def _span_norm(value: str) -> str:
    value = str(value or "").casefold().replace("’", "'").replace("‘", "'")
    value = value.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'.,!?;:—–-")


def _micro_audit_batch_v9f(provider, batch, translations, targets):
    rows = _BASE_MICRO_BATCH(provider, batch, translations, targets)
    by_id = {segment.id: segment for segment in batch}
    kept = []
    rejected = 0
    for row in rows:
        segment = by_id.get(str(row.get("id") or ""))
        span = _span_norm(row.get("source_span") or "")
        if segment is None or not span:
            rejected += 1
            continue
        source = _span_norm(segment.text)
        if span not in source:
            # The micro auditor sometimes described the neighbouring pair while
            # returning this id. Never let a cross-segment finding trigger repair.
            rejected += 1
            continue
        kept.append(row)
    if rejected:
        print(f"[v9f-micro-span-guard] rejected={rejected} kept={len(kept)}", flush=True)
    return kept


def _synthetic_risk_rows(targets, selected=None):
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
                "reason": "explicit either/neither/both/other construction requires antecedent resolution from local discourse before candidate selection",
                "repairability": "contextual",
            })
        elif _opaque_short_risk(text):
            out.append({
                "id": segment.id,
                "severity": "major",
                "confidence": 0.82,
                "code": "idiom",
                "source_span": "short elliptical/idiomatic utterance",
                "target_span": "",
                "reason": "very short pragmatic English fragment is high-risk for literal expansion; compare a context-resolved candidate against the current translation",
                "repairability": "contextual",
            })
    return out


def _parallel_micro_v9f(harness, targets, translations, selected=None):
    rows = _BASE_SPARSE_MICRO(harness, targets, translations, selected=selected)
    # Synthetic rows are routing hints, not persistent defects. Add them only on
    # the initial audit; changed candidates must be able to pass the re-audit.
    if selected is None:
        existing = {(str(row.get("id")), str(row.get("code"))) for row in rows}
        for row in _synthetic_risk_rows(targets):
            key = (row["id"], row["code"])
            if key not in existing:
                rows.append(row)
                existing.add(key)
    print(
        "[v9f-targeted-discourse] "
        + json.dumps(
            {
                "selected_segments": sum(_risk_segment_v9f(segment) for segment in (selected or targets)),
                "returned_findings": len(rows),
                "initial": selected is None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rows


def _context_payload(targets, translations, segment):
    pos = {row.id: i for i, row in enumerate(targets)}
    i = pos.get(segment.id, 0)
    before = targets[max(0, i - 4):i]
    after = targets[i + 1:i + 3]
    return {
        "before_en": [row.text for row in before],
        "before_ru_hypotheses": [translations.get(row.id, "") for row in before],
        "after_en": [row.text for row in after],
        "after_ru_hypotheses": [translations.get(row.id, "") for row in after],
    }


def _referent_candidate(harness, targets, segment, current, memory) -> str:
    provider = v9.v8._provider(harness)
    if provider is None:
        return ""
    payload = {
        "source": segment.text,
        "current_ru": current,
        "context": _context_payload(targets, {segment.id: current}, segment),
        "book_blueprint": {"characters": memory.characters, "glossary": memory.glossary},
    }
    system = """You are an EN→RU literary discourse resolver. SOURCE is authoritative; neighbouring Russian is absent or an
imperfect machine hypothesis. This segment contains an explicit referential construction such as either/neither/both/the other.
First resolve its exact antecedents from neighbouring English: list the concrete people/things referred to and exclude the
speaker unless the English context truly includes them. Then produce one complete natural Russian translation preserving all
clauses, uncertainty, negation and voice. Never translate 'either of them' as an arbitrary pronoun pair.
Return ONLY JSON {"antecedents":["..."],"translation":"..."}."""
    try:
        obj = v9.v8._complete_json(provider, system, payload)
        return v9._norm_text(obj.get("translation") or "")
    except Exception as exc:
        print(f"[v9f-referent-candidate] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _idiom_candidate(harness, targets, segment, current, memory) -> str:
    provider = v9.v8._provider(harness)
    if provider is None:
        return ""
    payload = {
        "source": segment.text,
        "current_ru": current,
        "context": v9._active_context(targets, segment),
        "book_blueprint": {"style": asdict(memory.style), "characters": memory.characters},
    }
    system = """You are an EN→RU fiction dialogue specialist. The segment contains a short colloquial or elliptical English
utterance. Before translating, silently expand its implied syntax and pragmatic force from context (attitude, agreement,
challenge, irony, omitted verb/complement). Do NOT translate a preposition+pronoun fragment word-for-word when it is idiomatic,
and do NOT turn an elliptical modal reply into literal obligation. Preserve speaker attitude in concise natural Russian and
translate the whole source segment, not only the fragment. Return ONLY JSON {"meaning":"brief pragmatic gloss",
"translation":"..."}."""
    try:
        obj = v9.v8._complete_json(provider, system, payload)
        return v9._norm_text(obj.get("translation") or "")
    except Exception as exc:
        print(f"[v9f-idiom-candidate] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _specialist_candidates_v9f(harness, targets, segment, current, memory, findings):
    values = list(_BASE_SPECIALISTS(harness, targets, segment, current, memory, findings))
    text = str(segment.text or "")
    dedicated = ""
    if v9e._explicit_referent_risk(text):
        dedicated = _referent_candidate(harness, targets, segment, current, memory)
    elif _opaque_short_risk(text):
        dedicated = _idiom_candidate(harness, targets, segment, current, memory)
    if dedicated:
        seen = {v9._norm_text(value).casefold() for value in [current, *values] if v9._norm_text(value)}
        if dedicated.casefold() not in seen:
            values.insert(0, dedicated)
    return values[:3]


def _judge_candidates_v9f(harness, targets, segment, candidates, memory):
    text = str(segment.text or "")
    special = v9e._explicit_referent_risk(text) or _opaque_short_risk(text)
    if not special:
        return _BASE_JUDGE(harness, targets, segment, candidates, memory)

    values = [v9._norm_text(value) for value in candidates if v9._norm_text(value)]
    if not values:
        return ""
    fatal = [v9._fatal_count(segment, value, memory) for value in values]
    best = min(fatal)
    pool = [value for value, count in zip(values, fatal) if count == best]
    if len(pool) == 1:
        return pool[0]

    payload = {
        "source": segment.text,
        "context": v9._active_context(targets, segment),
        "candidates": {chr(65 + i): value for i, value in enumerate(pool)},
        "book_blueprint": {"characters": memory.characters, "glossary": memory.glossary},
    }
    system = """Choose the best EN→RU literary candidate; do not rewrite. This is a discourse-sensitive short segment.
First resolve exact antecedents for either/neither/both/other and expand any elliptical/idiomatic conversational fragment into
its intended pragmatic meaning. Then choose the candidate that preserves that meaning, every referent, negation and speaker
attitude in natural Russian. Reject literal nonsense and invented self/people. Fidelity dominates elegance.
Return ONLY JSON {"choice":"A|B|C|D"}."""
    try:
        obj = v9.v8._complete_json(harness.gate, system, payload)
        choice = str(obj.get("choice") or "").strip().upper()
        idx = ord(choice) - 65 if len(choice) == 1 else -1
        if 0 <= idx < len(pool):
            return pool[idx]
    except Exception as exc:
        print(f"[v9f-candidate-judge] id={segment.id} error={type(exc).__name__}", flush=True)
    return _BASE_JUDGE(harness, targets, segment, pool, memory)


def _format_dialogue_v9f(segment, text: str):
    value, _ = v9d._format_dialogue_v9d(segment, text)
    before = value
    if v9d._source_has_dialogue(segment.text):
        # English quote marks stranded around Russian direct speech are always
        # punctuation artifacts here. Transform only boundary patterns; apostrophes
        # inside Latin tokens are untouched.
        value = re.sub(r"([,!?….])'\s*,?\s*—\s*", r"\1 — ", value)
        value = re.sub(r"([,!?….])'\s+(?=[А-Яа-яЁё])", r"\1 — ", value)
        value = re.sub(r"([,:;])\s*'\s*(?=[А-Яа-яЁё])", r"\1 — ", value)
        value = re.sub(r"(?<=[А-Яа-яЁё0-9])'\s+(?=[А-Яа-яЁё])", ", — ", value)
        value = re.sub(r"(?<=[А-Яа-яЁё0-9.!?…])'\s*$", "", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
    return value, int(value != before)


# Patch runtime symbols used by v9/v9c/v9e.
v9c._risk_segment = _risk_segment_v9f
v9c._micro_audit_batch = _micro_audit_batch_v9f
v9c._parallel_micro_audit = _parallel_micro_v9f
v9._specialist_candidates = _specialist_candidates_v9f
v9._judge_candidates = _judge_candidates_v9f
v9._format_dialogue_v9 = _format_dialogue_v9f


def _annotate_v9f() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9f-targeted-discourse-specialists",
        "discourse_audit": "explicit referents + cue-bearing short ellipsis/idioms only",
        "micro_qe_guard": "source-span ownership validation prevents cross-segment issue leakage",
        "referent_repair": "dedicated antecedent resolver + discourse-aware candidate judge",
        "idiom_repair": "dedicated pragmatic expansion candidate for opaque short fragments",
        "formatter": "structural Russian dialogue + stranded-apostrophe boundary cleanup",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9e.main()
    finally:
        _annotate_v9f()


if __name__ == "__main__":
    main()
