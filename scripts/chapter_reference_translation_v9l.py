from __future__ import annotations

import json
import re
import time
from collections import Counter

import chapter_reference_translation_v9k as v9k

v9j = v9k.v9j
v9i = v9k.v9i
v9g = v9k.v9g
v9f = v9k.v9f
v9 = v9k.v9

# v9l is intentionally a Chapter-3 regression harness, not the final production
# quality architecture.  It runs the real source-only GigaChat draft, then only
# the rare discourse/idiom priority repair, deterministic safety checks and
# typography.  Once the known classes pass, the same specialist is returned to
# normal v9 general-QE and only then to the full sequential 1→2→3 benchmark.

_ELLIPTICAL_MODAL = re.compile(
    r"(?:^|[\"'“‘]\s*)(?:i|you|he|she|we|they)\s+(?:do|does|did|should|would|could|can|may|might|must|need)\b",
    re.I,
)
_WITH_IT = re.compile(r"\bwith\s+it\b", re.I)
_NEG_EITHER = re.compile(r"\b(?:won't|wouldn't|can't|couldn't|didn't|don't|doesn't|not)\b[^.!?]{0,120}\beither\s+of\s+them\b", re.I)
_RU_OBLIGATION = re.compile(r"\b(?:должен|должна|должны|должно|следует|надо|нужно)\b", re.I)
_RU_WEAK_HEDGE = re.compile(r"\b(?:пожалуй|возможно|может\s+быть|наверное|вероятно)\b", re.I)
_RU_WITH_IT_LITERAL = re.compile(r"\b(?:с\s+(?:ним|ней|этим|тем)|при\s+этом)\b", re.I)
_RU_BAD_REFERENT = re.compile(r"\b(?:ни\s+их|их\s+обоих\s+не\s+увижу|ни\s+их,\s+ни\s+себя)\b", re.I)


def _priority_rows_v9l(targets, translated):
    # Keep v9k/v9i's deterministic signals, but also route source-side opaque
    # short idioms even when GigaChat produced a *different* literal calque than
    # the narrow RU regex knows about (e.g. "with it" -> "при этом").
    out = {str(row.get("id") or ""): dict(row) for row in v9k._priority_rows(targets, translated) if row.get("id")}
    for segment in targets:
        if segment.id in out:
            continue
        if v9f._opaque_short_risk(str(segment.text or "")):
            out[segment.id] = {
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.97,
                "code": "idiom",
                "source_span": "opaque short dialogue idiom/ellipsis",
                "target_span": "",
                "reason": "source-side opaque short dialogue construction requires pragmatic resolution before publication",
                "repairability": "contextual",
            }
    return list(out.values())


def _candidate_still_bad(segment, candidate: str, code: str, item: dict) -> tuple[bool, str]:
    source = str(segment.text or "")
    value = v9._norm_text(candidate)
    low = value.casefold()

    if code == "idiom":
        if v9g._literalization_kind(segment, value):
            return True, "known literalization detector still fires"
        if _WITH_IT.search(source) and _RU_WITH_IT_LITERAL.search(value):
            return True, "with-it remains a literal/comitative connective instead of pragmatic additive meaning"
        if _ELLIPTICAL_MODAL.search(source):
            # A bare elliptical auxiliary/modal reply normally inherits its
            # proposition from context.  Unless the English itself contains a
            # hedge, do not turn it into obligation or invent hesitation.
            if _RU_OBLIGATION.search(value):
                return True, "elliptical auxiliary remains literal obligation"
            source_low = source.casefold()
            source_has_hedge = any(x in source_low for x in ("suppose", "perhaps", "maybe", "probably", "possibly"))
            if not source_has_hedge and _RU_WEAK_HEDGE.search(value):
                return True, "elliptical confident reply was weakened by an invented hedge"

    if code == "referent" and re.search(r"\b(?:either|neither|both)\s+of\s+them\b", source, re.I):
        antecedents = [v9._norm_text(x) for x in (item.get("antecedents") or []) if v9._norm_text(x)]
        if len(antecedents) < 2:
            return True, "multi-member referent has fewer than two resolved antecedents"
        if _RU_BAD_REFERENT.search(value):
            return True, "Russian still contains malformed/incorrect plural referent wording"
        if _NEG_EITHER.search(source):
            # Preserve negative scope over the pair.  Russian may use either
            # "ни X, ни Y" or "и X, и Y ... не", but it must retain negation.
            if not re.search(r"\b(?:не|ни)\b", low):
                return True, "negative either-of-them lost negative scope"

    return False, ""


def _priority_semantic_repair_v9l(harness, targets, translated, memory):
    rows = _priority_rows_v9l(targets, translated)
    payload = v9k._priority_payload(targets, translated, rows)
    if not payload:
        return {"flagged": 0, "replaced": 0, "rejected": 0, "ids": []}

    system = """You are a narrow EN→RU FICTION DISCOURSE AND IDIOM REPAIRER.
The listed items were selected by source-side or deterministic risk rules. SOURCE English is authoritative. Neighbouring Russian strings are imperfect machine hypotheses, never a reference/gold translation.

For each item, first state the contextual/pragmatic meaning, then translate the WHOLE source segment.

REFERENTS:
- Resolve either/neither/both/the other to concrete discourse entities from nearby English, including entities implied by coordinated questions and elliptical answers.
- In a negative construction such as "won't ... either of them", preserve the negative scope naturally over both antecedents in Russian. Do not invent the speaker/self.

ELLIPSIS / AUXILIARIES:
- A short reply such as "I should do" may mean "of course I know / I'd better know / I certainly should know" because the omitted proposition comes from the prior question. It is not automatically obligation.
- Preserve pragmatic certainty. Do not add hedges like perhaps/maybe when the English/context is confident.

IDIOMS / PRAGMATIC MODIFIERS:
- Do not translate preposition+pronoun fragments mechanically.
- In a short attitude phrase of the form adjective/attitude + "with it", test whether "with it" means additive "as well / on top of that / besides". If so, Russian should express the additive attitude naturally (e.g. an "ещё и / к тому же" relation), not a literal "с этим / при этом" connective.
- Preserve speaker attitude (mockery, challenge, cheek, irony) concisely rather than explaining it.

Return ONLY JSON:
{"items":[{"id":"exact id","verdict":"keep|replace","antecedents":["..."],"certainty":"strong|neutral|hedged","meaning":"brief pragmatic gloss","translation":"complete Russian segment"}]}"""

    try:
        obj = v9.v8._complete_json(harness.gate, system, {"items": payload})
    except Exception as exc:
        print(f"[v9l-priority] error={type(exc).__name__}", flush=True)
        return {"flagged": len(payload), "replaced": 0, "rejected": len(payload), "ids": []}

    returned = obj.get("items") if isinstance(obj, dict) else []
    by_result = {str(x.get("id") or ""): x for x in returned if isinstance(x, dict)} if isinstance(returned, list) else {}
    by_segment = {segment.id: segment for segment in targets}
    by_finding = {str(row.get("id") or ""): row for row in rows}
    replaced, rejected = [], []

    for sid, finding in by_finding.items():
        segment = by_segment.get(sid)
        item = by_result.get(sid)
        if segment is None or item is None:
            rejected.append((sid, "missing model result"))
            continue
        candidate = v9._norm_text(item.get("translation") or "")
        current = v9._norm_text(translated.get(sid, ""))
        if str(item.get("verdict") or "").casefold() != "replace" or not candidate or candidate == current:
            # If source-side opaque risk was selected, a keep verdict is allowed
            # only if deterministic post-checks agree that the current text is not
            # a known literalization. This avoids forced stylistic rewrites.
            bad, reason = _candidate_still_bad(segment, current, str(finding.get("code") or ""), item)
            if bad:
                rejected.append((sid, "keep rejected: " + reason))
            continue
        if v9._fatal_count(segment, candidate, memory) > v9._fatal_count(segment, current, memory):
            rejected.append((sid, "fatal guard"))
            continue
        bad, reason = _candidate_still_bad(segment, candidate, str(finding.get("code") or ""), item)
        if bad:
            rejected.append((sid, reason))
            continue
        translated[sid] = candidate
        replaced.append(sid)
        print("[v9l-priority-repair] " + json.dumps({
            "id": sid,
            "code": finding.get("code"),
            "meaning": v9._norm_text(item.get("meaning") or "")[:180],
            "certainty": item.get("certainty"),
            "antecedents": item.get("antecedents") or [],
            "before": current[:260],
            "after": candidate[:260],
        }, ensure_ascii=False), flush=True)

    for sid, reason in rejected:
        print("[v9l-priority-reject] " + json.dumps({"id": sid, "reason": reason}, ensure_ascii=False), flush=True)
    return {"flagged": len(payload), "replaced": len(replaced), "rejected": len(rejected), "ids": replaced, "rejected_ids": [sid for sid, _ in rejected]}


def _priority_only_quality(harness, targets, translated, memory):
    started = time.perf_counter()
    priority = _priority_semantic_repair_v9l(harness, targets, translated, memory)

    # Deterministic safety/typography only. No chapter-wide LLM QE in this
    # regression harness: we are iterating the discourse specialist itself.
    finding_map = {}
    score_map = {}
    formatted = 0
    entity_fixes = 0
    unresolved = set()
    reasons = {}
    for segment in targets:
        value = str(translated.get(segment.id) or "")
        value, count = v9.v8._fix_near_entity_typos(segment, value, memory)
        entity_fixes += count
        value, changed = v9f._format_dialogue_v9f(segment, value)
        formatted += changed
        translated[segment.id] = value
        rows = v9._deterministic_findings(segment, value, memory)
        finding_map[segment.id] = rows
        score_map[segment.id] = v9._score(rows)
        if any(row.get("severity") == "critical" and float(row.get("confidence") or 0) >= 0.55 for row in rows):
            unresolved.add(segment.id)
            reasons[segment.id] = "; ".join(str(row.get("reason") or "") for row in rows)

    # Priority items that were rejected remain explicitly unresolved in the
    # regression report, even if the generic deterministic checks cannot see the
    # pragmatic defect.
    for sid in priority.get("rejected_ids") or []:
        unresolved.add(sid)
        reasons[sid] = "priority discourse/idiom candidate did not pass semantic acceptance"

    v9._V9_SCORES = score_map
    v9._V9_FINAL_FINDINGS = finding_map
    v9._V9_THREAD_MISMATCHES = set()
    v9.v6._V6_REMAINING_HARD = set(unresolved)
    v9.v6._V6_REMAINING_MEDIUM = set()
    v9.v6._V6_REASONS = reasons
    counts = Counter(row.get("severity") for rows in finding_map.values() for row in rows)
    v9._V9_STATS = {
        "mode": "priority-only-regression",
        "audited": len(targets),
        "priority_prepass": priority,
        "entity_typo_fixes": entity_fixes,
        "dialogue_segments_formatted": formatted,
        "remaining_deterministic_critical": len(unresolved),
        "severity_counts": dict(counts),
        "qe_elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    v9.hybrid.progress({"phase": "v9l_priority_only_done", **v9._V9_STATS})
    return dict(v9._V9_STATS)


def _annotate_v9l():
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9l-priority-only-regression",
        "purpose": "Chapter Three discourse/idiom regression before restoring general QE",
        "source_side_opaque_routing": True,
        "general_qe_in_this_harness": False,
        "gold_reference_available_to_pipeline": False,
    }
    data["v9l_priority_only"] = dict(v9._V9_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    v9._configure_v9()
    v9.v3._semantic_short_repair = _priority_only_quality
    try:
        v9.v3.main()
    finally:
        v9.v6._annotate_report()
        _annotate_v9l()


if __name__ == "__main__":
    main()
