from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9v as v9v

v9u, v9t = v9v.v9u, v9v.v9t
v9s, v9r, v9, v8, v6, v3 = v9v.v9s, v9v.v9r, v9v.v9, v9v.v8, v9v.v6, v9v.v3
_BASE_QUALITY = v9t._quality_v9t
_V9W_STATS: dict[str, Any] = {}

# v9w removes the second LLM scan/verifier layer entirely. It routes only
# deterministic high-risk source patterns and glossary mismatches into the
# already proven fresh-candidate + judge repair path. No gold/reference RU.

_PATTERNS: list[tuple[re.Pattern, str, int, str]] = [
    (re.compile(r"\boutnumber(?:ed|ing)?\b", re.I), "comparison", 12, "preserve who is outnumbered and the exact ratio direction"),
    (re.compile(r"\blast\s+[^.!?]{0,45}\s+but\s+one\b", re.I), "ordinal", 12, "British 'last but one' means penultimate, not last"),
    (re.compile(r"\bcan(?:not|'t)\s+be\s+too\b", re.I), "idiom", 12, "preserve the idiomatic scope: one cannot be excessively X / caution is always warranted"),
    (re.compile(r"\bindulge\s+me\b", re.I), "idiom", 12, "render the pragmatic request in context, not a literal comfort/consolation sense"),
    (re.compile(r"\b(?:i(?:'ve| have)?\s+got\s+a\s+living\s+to\s+make|make\s+a\s+living)\b", re.I), "idiom", 12, "means needing to earn a living / make a livelihood"),
    (re.compile(r"\btakes?\s+(?:a\s+)?(?:little\s+)?(?:while|time)\s+to\s+learn\b", re.I), "learning", 12, "keep the act of learning; do not weaken it to merely feeling unfamiliar"),
    (re.compile(r"^\s*[\"'“‘]?\s*(?:you\s+did|i\s+should\s+do)\b", re.I), "ellipsis", 12, "reconstruct the omitted predicate from the immediately preceding dialogue"),
    (re.compile(r"\b(?:either|neither|both)\b", re.I), "referent", 9, "resolve the exact antecedents and negative scope from context"),
    (re.compile(r"\b(?:advocate|counsel|prosecutor|attorney)\b", re.I), "role", 11, "infer the courtroom function from context; do not default to a dictionary label if the speaker is accusing rather than defending"),
    (re.compile(r"\bwith\s+it\b", re.I), "pragmatics", 8, "verify whether 'with it' is additive/pragmatic rather than literal comitative wording"),
]
_AUX_RE = re.compile(r"\b(?:do|does|did|have|has|had|can|could|shall|should|will|would|may|might|must)\b", re.I)
_FIRST_RE = re.compile(r"\b(?:I|I've|I'd|I'll|I'm)\b", re.I)


def _route(targets, translated, memory):
    by_id: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        reasons = []
        best = 0
        code = "challenge"

        for pattern, pcode, priority, reason in _PATTERNS:
            if pattern.search(source):
                reasons.append(reason)
                if priority > best:
                    best, code = priority, pcode

        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", source)
        if v9._is_dialogue(source) and len(words) <= 8 and (_AUX_RE.search(source) or _FIRST_RE.search(source)):
            reasons.append("short dialogue may contain ellipsis/modality that must be reconstructed from neighbors")
            if best < 9:
                best, code = 9, "ellipsis"

        if _FIRST_RE.search(source) and re.search(r"\b(?:сказала|была|сделала|думала|знала|хотела|могла|должна|решила)\b", current, re.I):
            reasons.append("possible first-person speaker-gender mismatch")
            if best < 11:
                best, code = 11, "gender"

        # Source-only glossary intelligence is cheap and especially useful for
        # rare technical/book-specific terms. Only route a mismatch when the
        # glossary target is non-empty and absent from the current translation.
        source_low = source.casefold()
        current_low = current.casefold()
        glossary_hits = []
        for src, ru in dict(getattr(memory, "glossary", {}) or {}).items():
            src = v9._norm_text(src)
            ru = v9._norm_text(ru)
            if len(src) < 4 or len(ru) < 3:
                continue
            if src.casefold() in source_low and ru.casefold() not in current_low:
                glossary_hits.append({"source": src, "expected_ru": ru})
            if len(glossary_hits) >= 4:
                break
        if glossary_hits:
            reasons.append("source glossary terms are missing or translated inconsistently: " + json.dumps(glossary_hits, ensure_ascii=False))
            if best < 10:
                best, code = 10, "glossary_term"

        if best:
            by_id[segment.id] = {
                "id": segment.id,
                "index": index,
                "priority": best,
                "code": code,
                "reason": "; ".join(reasons),
                "glossary_hits": glossary_hits,
            }

    rows = list(by_id.values())
    rows.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    cap = max(1, int(os.getenv("BOOKAI_V9W_RETRANSLATE_MAX") or "8"))
    return rows[:cap], rows


def _issue_row(route):
    return {
        "id": route["id"],
        "severity": "major",
        "confidence": 0.95 if int(route["priority"]) >= 11 else 0.88,
        "code": route["code"],
        "source_span": "",
        "target_span": "",
        "reason": route["reason"],
        "repairability": "local",
    }


def _repair_direct(harness, targets, translated, memory, selected):
    pos = {s.id: i for i, s in enumerate(targets)}
    recurring = v9v._recurring_name_tokens(targets, translated)
    workers = max(1, min(4, int(os.getenv("BOOKAI_V9W_RETRANSLATE_WORKERS") or "4"), len(selected) or 1))

    def one(route):
        sid = route["id"]
        i = pos[sid]
        segment = targets[i]
        current = str(translated.get(sid) or "")
        issue = _issue_row(route)
        payload = v9u._payload(targets, translated, memory, i, [issue])
        payload["router_reason"] = route["reason"]
        payload["glossary_hits"] = route.get("glossary_hits") or []
        names, glossary = v9v._locks_for(segment, current, memory, recurring)
        payload["hard_locks"] = {"proper_names": names, "glossary_terms": glossary}

        fresh = v9v._two_candidates(harness, payload)
        choices = [current]
        for value in fresh:
            if v9._fatal_count(segment, value, memory) > v9._fatal_count(segment, current, memory):
                continue
            if not v9v._preserves_locks(value, names, glossary):
                print(f"[v9w-lock-reject] id={sid}", flush=True)
                continue
            choices.append(value)

        dedup, seen = [], set()
        for value in choices:
            key = v9._norm_text(value)
            if key and key not in seen:
                seen.add(key)
                dedup.append(value)
        if len(dedup) < 2:
            return sid, ""

        chosen = dedup[v9u._judge(harness, payload, dedup)]
        if v9._norm_text(chosen) == v9._norm_text(current):
            return sid, ""
        return sid, chosen

    results = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9w-repair") as pool:
        for future in as_completed([pool.submit(one, route) for route in selected]):
            sid, value = future.result()
            if value:
                results[sid] = value

    changed = []
    for route in selected:
        sid = route["id"]
        if sid in results:
            translated[sid] = results[sid]
            changed.append(sid)
    return changed


def _demote_unconfirmed_priority(semantic):
    out = {}
    demoted = 0
    for sid, rows in semantic.items():
        keep = []
        for row in rows:
            row = dict(row)
            if str(row.get("code") or "") == "priority_invariant":
                row["severity"] = "minor"
                row["confidence"] = min(0.70, float(row.get("confidence") or 0.70))
                row["reason"] = "unconfirmed priority heuristic: " + str(row.get("reason") or "")
                demoted += 1
            keep.append(row)
        if keep:
            out[sid] = keep
    return out, demoted


def _quality_v9w(harness, targets, translated, memory):
    global _V9W_STATS
    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    base = {sid: v9t._semantic_rows(rows) for sid, rows in v9._V9_FINAL_FINDINGS.items()}
    semantic, demoted = _demote_unconfirmed_priority(base)

    selected, all_routed = _route(targets, translated, memory)
    changed = _repair_direct(harness, targets, translated, memory, selected)
    changed_set = set(changed)

    for sid in changed_set:
        segment = next((s for s in targets if s.id == sid), None)
        if segment is not None:
            translated[sid] = v9s._format_dialogue_v9s(segment, translated.get(sid, ""))[0]

    # One delta audit only. No second all-segment scanner and no independent
    # verifier round: the source-aware judge already compared current + fresh
    # candidates, and v9's audit now checks only text that actually changed.
    fresh_rows = []
    if changed_set:
        fresh_rows = v9t._selected_audit(harness, targets, translated, memory, changed_set)
        by_id = {}
        for row in fresh_rows:
            by_id.setdefault(str(row.get("id") or ""), []).append(dict(row))
        for sid in changed_set:
            semantic[sid] = v9t._semantic_rows(by_id.get(sid, []))

    for s in targets:
        translated[s.id] = v9s._format_dialogue_v9s(s, translated.get(s.id, ""))[0]

    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(r.get("severity") for rows in final_map.values() for r in rows)

    _V9W_STATS = {
        "deterministic_router_candidates": len(all_routed),
        "deterministic_router_selected": len(selected),
        "selected_ids": [r["id"] for r in selected],
        "two_candidate_changed": len(changed),
        "two_candidate_changed_ids": changed,
        "priority_invariants_demoted_without_independent_support": demoted,
        "hard_entity_glossary_locks": True,
        "delta_audit_findings": len(fresh_rows),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
        "quality_mode": "v9t-fast+deterministic-challenge-router+2way-retranslation+hard-locks+single-delta-audit",
    }
    stats.update(_V9W_STATS)
    print("[bookai-v9w] " + json.dumps(_V9W_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9w():
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9w-deterministic-router",
        "gold_reference_available_to_pipeline": False,
        "risk_policy": "no LLM challenge scan; deterministic semantic cues + source-only glossary mismatches",
        "priority_policy": "unconfirmed priority invariants are nonblocking hints",
        "repair_policy": "max-8 direct 2-way fresh retranslations + current translation + source-aware judge",
        "entity_policy": "hard-lock recurring established RU names and active glossary terms",
        "performance": "v9t fast path plus bounded direct repairs and one delta audit only",
    }
    data["v9w_stats"] = dict(_V9W_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    v9._configure_v9()
    v9._MATERIALS["coal"] = ("угл",)
    v9s._deterministic_v9s = v9u._deterministic
    v9._deterministic_findings = v9u._deterministic
    v9._thread_findings = v9s._thread_v9s
    v8._fix_near_entity_typos = v9s._entity_fix_v9s
    v9._format_dialogue_v9 = v9s._format_dialogue_v9s
    v8._normalize_dialogue_v8 = v9s._format_dialogue_v9s
    v3._semantic_short_repair = _quality_v9w
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v9r.v9j._annotate_v9j()
        v9r.v9k._annotate_v9k()
        v9r._annotate_v9r()
        v9s._annotate_v9s()
        v9t._annotate_v9t()
        _annotate_v9w()


if __name__ == "__main__":
    main()
