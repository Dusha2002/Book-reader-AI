from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chapter_reference_translation_v9r as v9r

v9 = v9r.v9
v8 = v9.v8
v6 = v9r.v6
v3 = v9.v3

# v9s addresses defects exposed by the blind full Chapters 1->3 evaluation:
#   * semantic locks changed the final text after broad QE, but v9r kept stale
#     findings/scores from the pre-lock candidate;
#   * uncertain entity/thread hints could consume the bounded semantic repair cap;
#   * the deterministic coal stem missed Russian inflections such as "углём";
#   * model edits could leak orphan English quote punctuation and raw English;
#   * one QE opinion could over-flag a faithful contextual expansion.
#
# The reference Russian translation is still evaluation-only and is NEVER read
# by this pipeline. v9s adds a final source-only consensus audit + bounded tail
# repair, and rebuilds publication status from the actual final text.

_BASE_QUALITY = v9r._quality_v9r
_BASE_FORMATTER = v9._format_dialogue_v9
_BASE_DETERMINISTIC = v9._deterministic_findings
_BASE_THREAD = v9._thread_findings
_BASE_ENTITY_FIX = v8._fix_near_entity_typos

_V9S_STATS: dict[str, Any] = {}


def _blocking(row: dict[str, Any]) -> bool:
    severity = str(row.get("severity") or "minor")
    confidence = float(row.get("confidence") or 0.0)
    return (severity == "critical" and confidence >= 0.55) or (severity == "major" and confidence >= 0.78)


def _format_dialogue_v9s(segment, text: str) -> tuple[str, int]:
    original = v9._norm_text(text)
    value, _ = _BASE_FORMATTER(segment, original)

    # Convert paired straight apostrophe quotes around Russian text first.
    # This catches nested direct speech that remains after a model edit.
    value = re.sub(
        r"'([^'\n]{0,220}[А-Яа-яЁё][^'\n]{0,220})'",
        r"«\1»",
        value,
    )
    # Remove only quote punctuation that cannot be a meaningful Russian
    # apostrophe. Do not touch apostrophes embedded inside Latin words: raw
    # English is a semantic hard issue and must go through the repair tail.
    value = re.sub(r"([.!?…])\s*['’]\s*,?", r"\1", value)
    value = re.sub(r"([,;:])\s*['’]\s*(?=(?:—|[А-ЯЁ]))", r"\1 ", value)
    value = re.sub(r"—\s*['’]\s*(?=[А-ЯЁа-яё])", "— ", value)
    value = re.sub(r"(?<=[А-Яа-яЁё])['’](?=[,.;:!?…])", "", value)
    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value, int(value != original)


def _deterministic_v9s(segment, candidate: str, memory):
    rows = []
    for raw in _BASE_DETERMINISTIC(segment, candidate, memory):
        row = dict(raw)
        # A source-only Book Blueprint can choose a defensible but different
        # transliteration. Entity consistency matters, but it must not crowd
        # out meaning/coverage repairs or alone block publication.
        if row.get("code") == "entity":
            row["severity"] = "minor"
            row["confidence"] = min(float(row.get("confidence") or 0.0), 0.72)
        rows.append(row)
    return rows


def _thread_v9s(targets, translations, memory):
    rows, index = _BASE_THREAD(targets, translations, memory)
    softened = []
    for raw in rows:
        row = dict(raw)
        row["severity"] = "minor"
        row["confidence"] = min(float(row.get("confidence") or 0.0), 0.72)
        softened.append(row)
    return softened, index


def _entity_fix_v9s(segment, candidate: str, memory) -> tuple[str, int]:
    value, fixes = _BASE_ENTITY_FIX(segment, candidate, memory)
    # A second, still conservative typo pass for long recurring names. The
    # source segment must explicitly mention the entity, the candidate token
    # must share its Cyrillic prefix, and edit distance is tightly bounded.
    for source, target in v8._entity_map(memory).items():
        if not v8._source_mentions(segment.text, source) or v8._entity_present(value, target):
            continue
        words = target.split()
        if not words:
            continue
        canonical = words[-1]
        allowed = v8._inflected_last_forms(canonical)
        prefix = canonical.casefold()[: max(2, min(4, len(canonical) // 2))]
        threshold = 2 if len(canonical) < 8 else 3
        best = None
        for match in v8._CYR_WORD_RE.finditer(value):
            token = match.group(0)
            if not token.casefold().startswith(prefix):
                continue
            options = sorted((v8._edit_distance(token, form), form) for form in allowed)
            if not options or options[0][0] > threshold:
                continue
            distance, form = options[0]
            if best is None or distance < best[0]:
                best = (distance, match.start(), match.end(), form)
        if best is None:
            continue
        _, start, end, replacement = best
        original = value[start:end]
        if original[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        value = value[:start] + replacement + value[end:]
        fixes += 1
    return value, fixes


def _verify_llm_findings(harness, targets, translated, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Second independent source-only opinion for alleged publication defects.

    It is intentionally conservative: explicit contextual expansions are legal
    when the neighboring English source resolves the antecedent. This removes
    false positives such as either-of-them -> wife+daughter while preserving
    real relation/negation/idiom failures.
    """
    candidates = [dict(row) for row in rows if _blocking(row) and row.get("code") not in {"entity", "thread"}]
    if not candidates:
        return []
    by_id = {segment.id: segment for segment in targets}
    pos = {segment.id: i for i, segment in enumerate(targets)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        sid = str(row.get("id") or "")
        if sid in by_id:
            grouped.setdefault(sid, []).append(row)

    items = []
    for sid, alleged in grouped.items():
        i = pos[sid]
        items.append({
            "id": sid,
            "source": by_id[sid].text,
            "current_ru": translated.get(sid, ""),
            "before_en": [x.text for x in targets[max(0, i - 2):i]],
            "after_en": [x.text for x in targets[i + 1:i + 3]],
            "alleged": [
                {
                    "code": row.get("code"),
                    "severity": row.get("severity"),
                    "confidence": row.get("confidence"),
                    "source_span": row.get("source_span"),
                    "target_span": row.get("target_span"),
                    "reason": row.get("reason"),
                }
                for row in alleged
            ],
        })

    system = """You are the FINAL conservative verifier for an EN→RU literary translation QE system. Do not rewrite.
For every alleged issue decide whether it is a REAL publication defect in current_ru when SOURCE and neighboring English
context are considered. A faithful Russian paraphrase is not an error. Explicitly naming an antecedent (for example
'either of them' -> 'wife and daughter') is valid when context unambiguously identifies it. Do not flag harmless
transliteration preference or stylistic taste. Confirm only real meaning loss/addition, wrong actor/referent/relation,
negation/modality/chronology error, corrupted number/material/technical fact, raw English, broken grammar, or a genuinely
failed idiom/joke. Return ONLY JSON {"confirmed":[{"id":"...","code":"...","severity":"critical|major",
"confidence":0.0,"reason":"concise"}]}. Use [] if the allegation is not real. Never invent a new issue."""

    out: list[dict[str, Any]] = []
    max_chars = max(5000, int(os.getenv("BOOKAI_V9S_VERIFY_BATCH_CHARS") or "11500"))
    batches = []
    current = []
    size = 0
    for item in items:
        weight = len(item["source"]) + len(item["current_ru"]) + sum(len(str(x)) for x in item["alleged"])
        if current and size + weight > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += weight
    if current:
        batches.append(current)

    lookup = {(str(row.get("id")), str(row.get("code"))): row for row in candidates}
    for batch in batches:
        try:
            obj = v8._complete_json(harness.gate, system, {"items": batch})
            confirmed = obj.get("confirmed") or []
        except Exception as exc:
            print(f"[v9s-verifier] size={len(batch)} error={type(exc).__name__}", flush=True)
            confirmed = []
            # Fail closed for critical allegations only; a verifier outage must
            # not silently bless a known deterministic/semantic critical issue.
            batch_ids = {str(x["id"]) for x in batch}
            for row in candidates:
                if str(row.get("id")) in batch_ids and str(row.get("severity")) == "critical":
                    confirmed.append({"id": row.get("id"), "code": row.get("code"), "severity": "critical", "confidence": row.get("confidence"), "reason": row.get("reason")})
        for raw in confirmed if isinstance(confirmed, list) else []:
            if not isinstance(raw, dict):
                continue
            key = (str(raw.get("id") or ""), str(raw.get("code") or ""))
            base = lookup.get(key)
            if not base:
                continue
            severity = str(raw.get("severity") or base.get("severity") or "major").lower()
            if severity not in {"critical", "major"}:
                severity = str(base.get("severity") or "major")
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence") or base.get("confidence") or 0.8)))
            except Exception:
                confidence = float(base.get("confidence") or 0.8)
            row = dict(base)
            row["severity"] = severity
            row["confidence"] = confidence
            if raw.get("reason"):
                row["reason"] = v9._norm_text(raw.get("reason"))[:500]
            out.append(row)
    return out


def _fresh_final_findings(harness, targets, translated, memory):
    llm_rows = v9._parallel_audit(harness, targets, translated, memory)
    verified = _verify_llm_findings(harness, targets, translated, llm_rows)
    minor_llm = [dict(row) for row in llm_rows if not _blocking(row)]
    det_rows = [row for segment in targets for row in _deterministic_v9s(segment, translated.get(segment.id, ""), memory)]
    thread_rows, thread_index = _thread_v9s(targets, translated, memory)
    finding_map = v9._merge_findings(verified, minor_llm, det_rows, thread_rows)
    score_map = {segment.id: v9._score(finding_map.get(segment.id, [])) for segment in targets}
    return finding_map, score_map, {
        "raw_llm": len(llm_rows),
        "verified_blocking": len(verified),
        "deterministic": len(det_rows),
        "thread_hints": len(thread_rows),
        "thread_terms": len(thread_index),
    }


def _tail_repair(harness, targets, translated, memory, finding_map, score_map) -> tuple[int, list[str]]:
    by_id = {segment.id: segment for segment in targets}
    repair_ids = [
        segment.id for segment in targets
        if any(_blocking(row) and row.get("code") not in {"entity", "thread"} for row in finding_map.get(segment.id, []))
    ]
    repair_ids.sort(key=lambda sid: score_map.get(sid, 100.0))
    cap = max(0, int(os.getenv("BOOKAI_V9S_TAIL_REPAIR_MAX") or "28"))
    repair_ids = repair_ids[:cap]
    changed = []
    for sid in repair_ids:
        segment = by_id[sid]
        current = str(translated.get(sid) or "")
        rows = [row for row in finding_map.get(sid, []) if _blocking(row) and row.get("code") not in {"entity", "thread"}]
        candidates = v9._specialist_candidates(harness, targets, segment, current, memory, rows)
        if not candidates:
            continue
        chosen = v9._judge_candidates(harness, targets, segment, [current, *candidates], memory)
        if not chosen or v9._norm_text(chosen) == v9._norm_text(current):
            continue
        if v9._fatal_count(segment, chosen, memory) > v9._fatal_count(segment, current, memory):
            continue
        translated[sid] = chosen
        changed.append(sid)
    return len(changed), changed


def _publish_final_state(targets, finding_map, score_map):
    critical: set[str] = set()
    major: set[str] = set()
    reasons: dict[str, str] = {}
    for segment in targets:
        rows = finding_map.get(segment.id, [])
        blockers = [row for row in rows if _blocking(row) and row.get("code") not in {"entity", "thread"}]
        if any(str(row.get("severity")) == "critical" for row in blockers):
            critical.add(segment.id)
        elif blockers:
            major.add(segment.id)
        if blockers:
            reasons[segment.id] = "; ".join(dict.fromkeys(str(row.get("reason") or "") for row in blockers if row.get("reason")))

    v9._V9_FINAL_FINDINGS = finding_map
    v9._V9_SCORES = score_map
    # Any independently verified remaining major is publication-blocking. This
    # prevents a chapter with known meaning errors from being labelled complete.
    v6._V6_REMAINING_HARD = set(critical | major)
    v6._V6_REMAINING_MEDIUM = set()
    v6._V6_REASONS = reasons
    return critical, major


def _quality_v9s(harness, targets, translated, memory):
    global _V9S_STATS

    # Fix morphology-sensitive material detection before any v9 audit runs.
    v9._MATERIALS["coal"] = ("угл",)

    # Keep uncertain Book-Blueprint entity/thread hints out of the semantic
    # repair budget. They remain diagnostics and still exclude bad TM rows.
    v9._deterministic_findings = _deterministic_v9s
    v9._thread_findings = _thread_v9s
    v8._fix_near_entity_typos = _entity_fix_v9s
    v9._format_dialogue_v9 = _format_dialogue_v9s

    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    # v9r may restore semantic locks after broad QE. Normalize typography on
    # the ACTUAL final candidate before rebuilding findings from scratch.
    sanitized = 0
    for segment in targets:
        value, changed = _format_dialogue_v9s(segment, translated.get(segment.id, ""))
        translated[segment.id] = value
        sanitized += changed

    first_map, first_scores, first_meta = _fresh_final_findings(harness, targets, translated, memory)
    tail_count, tail_ids = _tail_repair(harness, targets, translated, memory, first_map, first_scores)

    # Clean any punctuation introduced by the tail and then independently
    # re-audit the post-repair text. These are the only findings used for final
    # status/TM trust; stale pre-lock findings are discarded.
    for segment in targets:
        value, changed = _format_dialogue_v9s(segment, translated.get(segment.id, ""))
        translated[segment.id] = value
        sanitized += changed

    final_map, final_scores, final_meta = _fresh_final_findings(harness, targets, translated, memory)
    critical, major = _publish_final_state(targets, final_map, final_scores)

    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)
    entity_hints = sum(1 for rows in final_map.values() for row in rows if row.get("code") in {"entity", "thread"})
    mean_score = round(sum(final_scores.values()) / max(1, len(final_scores)), 2)
    _V9S_STATS = {
        "stale_findings_discarded": True,
        "typography_segments_sanitized": sanitized,
        "pre_tail_audit": first_meta,
        "tail_repair_selected": min(
            int(os.getenv("BOOKAI_V9S_TAIL_REPAIR_MAX") or "28"),
            sum(any(_blocking(r) and r.get("code") not in {"entity", "thread"} for r in rows) for rows in first_map.values()),
        ),
        "tail_repair_changed": tail_count,
        "tail_repair_ids": tail_ids,
        "final_audit": final_meta,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "entity_thread_hints_nonblocking": entity_hints,
        "mean_quality_score": mean_score,
        "severity_counts": dict(counts),
        "quality_mode": "v9r-semantic-lock+nonblocking-entity-hints+consensus-final-qe+bounded-tail+final-rebuild",
    }
    stats.update(_V9S_STATS)
    print("[bookai-v9s] " + json.dumps(_V9S_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9s() -> None:
    report = v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9s-final-consensus-qe",
        "gold_reference_available_to_pipeline": False,
        "stale_finding_policy": "discard all pre-lock QE findings; rebuild from actual post-lock final text",
        "qe_consensus": "bilingual audit -> independent conservative verifier -> bounded specialist tail -> fresh final audit",
        "entity_policy": "source-only entity/thread hints remain consistency diagnostics and TM trust guards, not semantic repair blockers",
        "formatter": "deterministic orphan-quote cleanup + raw-English publication blocker",
        "publication_gate": "verified critical OR major semantic defect blocks completion",
    }
    data["v9s_stats"] = dict(_V9S_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    v9._configure_v9()
    # _configure_v9 writes the v9 defaults back into v3/v8. Override them only
    # after configuration so every call in this run sees v9s behavior.
    v9._deterministic_findings = _deterministic_v9s
    v9._thread_findings = _thread_v9s
    v8._fix_near_entity_typos = _entity_fix_v9s
    v9._format_dialogue_v9 = _format_dialogue_v9s
    v8._normalize_dialogue_v8 = _format_dialogue_v9s
    v3._semantic_short_repair = _quality_v9s
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v9r.v9j._annotate_v9j()
        v9r.v9k._annotate_v9k()
        v9r._annotate_v9r()
        _annotate_v9s()


if __name__ == "__main__":
    main()
