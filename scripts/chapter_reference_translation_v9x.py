from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9w as v9w

v9v, v9u, v9t = v9w.v9v, v9w.v9v.v9u, v9w.v9t
v9s, v9r, v9, v8, v6, v3 = v9w.v9s, v9w.v9r, v9w.v9, v9w.v8, v9w.v6, v9w.v3
_V9X_STATS: dict[str, Any] = {}

# v9x keeps v9w's fast deterministic challenge router, but removes the older
# priority pre/post retry loop from the base QE. One source-side problem should
# have one repair owner. It also makes OpenRouter failures survivable through
# provider retries (workflow), lower burst concurrency and a candidate-first
# fallback if the final judge alone is unavailable. No Russian gold/reference.

_PATTERNS: list[tuple[re.Pattern, str, int, str]] = [
    (re.compile(r"\boutnumber(?:ed|ing)?\b", re.I), "comparison", 15,
     "preserve who is outnumbered and the exact numerical direction"),
    (re.compile(r"\blast\s+[^.!?]{0,45}\s+but\s+one\b", re.I), "ordinal", 14,
     "British 'last but one' means penultimate, not last"),
    (re.compile(r"\bcan(?:not|'t)\s+be\s+too\b", re.I), "idiom", 14,
     "preserve idiomatic scope: caution cannot be excessive / one cannot be too careful"),
    (re.compile(r"\bindulge\s+me\b", re.I), "idiom", 14,
     "render the pragmatic request from context, not comfort/consolation"),
    (re.compile(r"\b(?:i(?:'ve| have)?\s+got\s+a\s+living\s+to\s+make|make\s+a\s+living)\b", re.I), "idiom", 14,
     "means needing to earn a living / make a livelihood"),
    (re.compile(r"\btakes?\s+(?:a\s+)?(?:little\s+)?(?:while|time)\s+to\s+learn\b", re.I), "learning", 14,
     "keep the act of learning; do not weaken it to merely feeling unfamiliar"),
    (re.compile(r"^\s*[\"'“‘]?\s*(?:you\s+did|i\s+should\s+do)\b", re.I), "ellipsis", 14,
     "reconstruct the omitted predicate from the immediately preceding dialogue"),
    (re.compile(r"\b(?:advocate|counsel|prosecutor|attorney)\b", re.I), "role", 13,
     "infer the legal/courtroom function from actions and context, not the nearest dictionary label"),
    (re.compile(r"\bi(?:'ve| have)\s+(?:already|just|never|ever)\b", re.I), "present_perfect_dialogue", 12,
     "verify short first-person perfect tense, discourse force and speaker morphology from context"),
    (re.compile(r"\b(?:either|neither|both)\b", re.I), "referent", 11,
     "resolve exact antecedents and negative scope from neighboring discourse"),
    (re.compile(r"\bwith\s+it\b", re.I), "pragmatics", 10,
     "verify whether 'with it' is additive/pragmatic rather than literal comitative wording"),
]
_AUX_RE = re.compile(r"\b(?:do|does|did|have|has|had|can|could|shall|should|will|would|may|might|must)\b", re.I)
_FIRST_RE = re.compile(r"\b(?:I|I've|I'd|I'll|I'm|my|me)\b", re.I)
_RU_NAME_RE = re.compile(r"(?<![А-Яа-яЁё])([А-ЯЁ][а-яё]{2,})(?![А-Яа-яЁё])")
_RU_STOP = {"Он", "Она", "Они", "Это", "Если", "Когда", "Потом", "Теперь", "Тогда", "Да", "Нет", "Но", "Так", "Что", "Как", "Вот", "Глава"}


def _parse_source_memory_v9x(provider, sample: str):
    system = """Build a SOURCE-ONLY BOOK INTELLIGENCE bible for EN→RU literary translation.
You have only the English book. No published/reference translation exists. Never claim otherwise.
Infer authorial voice, rhythm, irony, dialogue behavior, characters, relationships/register and continuity.
Build a HIGH-PRECISION Russian glossary for:
1) recurring proper names, places, institutions and world terms;
2) recurring technical terms;
3) RARE BUT HIGH-IMPACT specialized lexical items visible in the sample even if they occur once, when a wrong dictionary sense would materially corrupt the scene: historical arms/armour, tools, crafts, mechanisms, plants/species, materials/processes, units and legal/official roles.
For a legal/official role infer FUNCTION from source context (for example accusing vs defending) before choosing Russian wording.
For invented/proper names preserve visible source syllables/vowels conservatively; do not shorten a name merely because an English pronunciation could permit it.
Do NOT lock ordinary vocabulary or context-dependent idioms. Prefer omission over guessing when evidence is genuinely insufficient.
Return ONLY compact JSON; do not translate passages.
Schema:
{"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"...","taboos":["..."]},"glossary":{"English name/term":"preferred Russian rendering"},"characters":{"English name":"ru=<name>;gender=male|female|unknown;voice=<brief>;role=<brief>;register=<brief>"},"rolling_summary":"compact factual continuity summary in Russian"}
Keep glossary <=110 entries and characters <=45."""
    try:
        obj = v6.extract_json(provider.complete(system, "SOURCE_BOOK_EVIDENCE:\n" + sample, temperature=0.0))
        if not isinstance(obj, dict):
            raise ValueError("book intelligence returned non-object JSON")
        style_obj = obj.get("style") or {}
        fallback = v6.BookMemory()
        style = fallback.style
        style.narrative_voice = str(style_obj.get("narrative_voice") or style.narrative_voice)
        style.rhythm = str(style_obj.get("rhythm") or style.rhythm)
        style.dialogue = str(style_obj.get("dialogue") or style.dialogue)
        style.humor = str(style_obj.get("humor") or style.humor)
        style.taboos = [str(x) for x in (style_obj.get("taboos") or style.taboos)][:12]
        glossary = {
            str(k).strip(): str(v).strip()
            for k, v in dict(obj.get("glossary") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        characters = {
            str(k).strip(): str(v).strip()
            for k, v in dict(obj.get("characters") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        return v6.BookMemory(
            style=style,
            glossary=dict(list(glossary.items())[:110]),
            characters=dict(list(characters.items())[:45]),
            rolling_summary=" ".join(str(obj.get("rolling_summary") or "").split())[:7000],
        )
    except Exception as exc:
        print(f"[v9x-book-intelligence] structured_pass_failed error={type(exc).__name__}; fallback=safe", flush=True)
        return v6.safe_analyze_memory(provider, sample, attempts=2)


def _base_quality_no_priority(harness, targets, translated, memory):
    """Run the normal broad v9 QE exactly once; defer priority idioms to v9x."""
    v9k, v9c = v9r.v9k, v9r.v9c
    v9k._RESOLVED_PRIORITY_IDS = set()
    v9c._parallel_micro_audit = v9k._parallel_micro_v9k
    v9c._parallel_audit_v9c.__globals__["_parallel_micro_audit"] = v9k._parallel_micro_v9k
    v9._parallel_audit = v9c._parallel_audit_v9c
    stats = dict(v9k._BASE_QUALITY(harness, targets, translated, memory) or {})
    stats["priority_prepass"] = {"flagged": 0, "replaced": 0, "rejected": 0, "ids": [], "deferred_to_v9x": True}
    stats["priority_semantic_locks"] = []
    stats["priority_restored_after_general_qe"] = []
    stats["priority_post_qe_repair"] = {"flagged": 0, "replaced": 0, "rejected": 0, "ids": [], "deferred_to_v9x": True}
    stats["priority_post_qe_unresolved"] = []
    stats["quality_mode"] = "one-full-v9-qe+priority-deferred-to-v9x-router"
    return stats


def _route_v9x(targets, translated, memory):
    all_rows: list[dict[str, Any]] = []
    for index, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        matched: list[tuple[int, str, str]] = []
        for pattern, code, priority, reason in _PATTERNS:
            if pattern.search(source):
                matched.append((priority, code, reason))

        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", source)
        if v9._is_dialogue(source) and len(words) <= 8 and (_AUX_RE.search(source) or _FIRST_RE.search(source)):
            matched.append((7, "short_dialogue", "short dialogue may contain omitted predicate/modality resolved by neighbors"))

        source_low, current_low = source.casefold(), current.casefold()
        glossary_hits = []
        for src, ru in dict(getattr(memory, "glossary", {}) or {}).items():
            src, ru = v9._norm_text(src), v9._norm_text(ru)
            if len(src) < 4 or len(ru) < 3:
                continue
            if src.casefold() in source_low and not v8._entity_present(current, ru):
                glossary_hits.append({"source": src, "expected_ru": ru})
            if len(glossary_hits) >= 5:
                break
        if glossary_hits:
            matched.append((12, "glossary_term", "source-only canonical glossary item missing/inconsistent: " + json.dumps(glossary_hits, ensure_ascii=False)))

        if not matched:
            continue
        matched.sort(key=lambda x: -x[0])
        priority, code, _ = matched[0]
        reasons = [reason for _, _, reason in matched]
        all_rows.append({
            "id": segment.id,
            "index": index,
            "priority": priority,
            "code": code,
            "reason": "; ".join(dict.fromkeys(reasons)),
            "glossary_hits": glossary_hits,
        })

    ranked = sorted(all_rows, key=lambda r: (-int(r["priority"]), int(r["index"])))
    cap = max(1, int(os.getenv("BOOKAI_V9X_RETRANSLATE_MAX") or "8"))

    # Diversity first: one highest-risk example per semantic class before a
    # second example from the same class. This prevents early generic dialogue
    # from starving a later comparison, role, glossary or morphology problem.
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    codes = []
    for row in ranked:
        if row["code"] not in codes:
            codes.append(row["code"])
    for code in codes:
        row = next((x for x in ranked if x["code"] == code and x["id"] not in seen_ids), None)
        if row is not None:
            selected.append(row); seen_ids.add(row["id"])
        if len(selected) >= cap:
            break
    if len(selected) < cap:
        for row in ranked:
            if row["id"] in seen_ids:
                continue
            selected.append(row); seen_ids.add(row["id"])
            if len(selected) >= cap:
                break
    selected.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return selected, ranked


def _canonical_locks(segment, current: str, memory, recurring: set[str]):
    entity_map = dict(v8._entity_map(memory))
    canon = []
    for src, ru in entity_map.items():
        if len(str(src)) >= 2 and v8._source_mentions(segment.text, str(src)):
            ru = v9._norm_text(ru)
            if ru:
                canon.append(ru)

    # If source intelligence knows the canonical entity, do not lock a competing
    # recurring draft spelling. Otherwise preserve established recurring RU names.
    names = list(dict.fromkeys(canon))
    if not canon:
        for token in _RU_NAME_RE.findall(current):
            if token not in _RU_STOP and token.casefold() in recurring:
                names.append(token)

    glossary = []
    source_low, current_low = segment.text.casefold(), current.casefold()
    for src, ru in dict(getattr(memory, "glossary", {}) or {}).items():
        src, ru = v9._norm_text(src), v9._norm_text(ru)
        if len(src) >= 3 and len(ru) >= 3 and src.casefold() in source_low and v8._entity_present(current, ru):
            glossary.append(ru)
        if len(glossary) >= 12:
            break
    return sorted(set(names)), sorted(set(glossary))


def _judge_v9x(harness, payload, choices):
    system = """Choose the best COMPLETE EN→RU literary candidate. SOURCE is authoritative. Fidelity outranks fluency: exact actor/action/object, comparison direction, polarity, numbers, role/function, idiom, ellipsis/referent and chronology. Then prefer natural professional Russian preserving established names/terms and voice. A fluent mistranslation always loses. ONLY JSON {\"best_index\":0}."""
    try:
        idx = int(v8._complete_json(harness.gate, system, {**payload, "candidates": choices}).get("best_index"))
        if 0 <= idx < len(choices):
            return idx, False
    except Exception as exc:
        print(f"[v9x-judge-fallback] id={payload.get('id')} error={type(exc).__name__}", flush=True)
    # Candidate generation itself is source-aware and targeted. If only the
    # judge is unavailable, prefer the first safe fresh candidate over knowingly
    # retaining the risky draft.
    return (1 if len(choices) > 1 else 0), True


def _repair_direct_v9x(harness, targets, translated, memory, selected):
    pos = {s.id: i for i, s in enumerate(targets)}
    recurring = v9v._recurring_name_tokens(targets, translated)
    workers = max(1, min(3, int(os.getenv("BOOKAI_V9X_RETRANSLATE_WORKERS") or "2"), len(selected) or 1))

    def one(route):
        sid = route["id"]
        i = pos[sid]
        segment = targets[i]
        current = str(translated.get(sid) or "")
        issue = v9w._issue_row(route)
        payload = v9u._payload(targets, translated, memory, i, [issue])
        payload["router_reason"] = route["reason"]
        payload["glossary_hits"] = route.get("glossary_hits") or []
        names, glossary = _canonical_locks(segment, current, memory, recurring)
        payload["hard_locks"] = {"proper_names": names, "glossary_terms": glossary}

        fresh = v9v._two_candidates(harness, payload)
        choices = [current]
        for value in fresh:
            if v9._fatal_count(segment, value, memory) > v9._fatal_count(segment, current, memory):
                continue
            if not v9v._preserves_locks(value, names, glossary):
                print(f"[v9x-lock-reject] id={sid}", flush=True)
                continue
            choices.append(value)
        dedup, seen = [], set()
        for value in choices:
            key = v9._norm_text(value)
            if key and key not in seen:
                seen.add(key); dedup.append(value)
        if len(dedup) < 2:
            return sid, "", False
        idx, judge_fallback = _judge_v9x(harness, payload, dedup)
        chosen = dedup[idx]
        if v9._norm_text(chosen) == v9._norm_text(current):
            return sid, "", judge_fallback
        return sid, chosen, judge_fallback

    results, fallback_count = {}, 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9x-repair") as pool:
        for future in as_completed([pool.submit(one, route) for route in selected]):
            sid, value, fallback = future.result()
            fallback_count += int(fallback)
            if value:
                results[sid] = value
    changed = []
    for route in selected:
        sid = route["id"]
        if sid in results:
            translated[sid] = results[sid]
            changed.append(sid)
    return changed, fallback_count


def _quality_v9x(harness, targets, translated, memory):
    global _V9X_STATS
    stats = dict(v9t._quality_v9t(harness, targets, translated, memory) or {})
    base = {sid: v9t._semantic_rows(rows) for sid, rows in v9._V9_FINAL_FINDINGS.items()}
    semantic, demoted = v9w._demote_unconfirmed_priority(base)

    selected, all_routed = _route_v9x(targets, translated, memory)
    changed, judge_fallbacks = _repair_direct_v9x(harness, targets, translated, memory, selected)
    changed_set = set(changed)
    for sid in changed_set:
        segment = next((s for s in targets if s.id == sid), None)
        if segment is not None:
            translated[sid] = v9s._format_dialogue_v9s(segment, translated.get(sid, ""))[0]

    fresh_rows = []
    if changed_set:
        fresh_rows = v9t._selected_audit(harness, targets, translated, memory, changed_set)
        by_id = {}
        for row in fresh_rows:
            by_id.setdefault(str(row.get("id") or ""), []).append(dict(row))
        for sid in changed_set:
            semantic[sid] = v9t._semantic_rows(by_id.get(sid, []))

    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]

    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(r.get("severity") for rows in final_map.values() for r in rows)
    _V9X_STATS = {
        "old_priority_repair_passes_disabled": True,
        "deterministic_router_candidates": len(all_routed),
        "deterministic_router_selected": len(selected),
        "selected_ids": [r["id"] for r in selected],
        "selected_codes": [r["code"] for r in selected],
        "two_candidate_changed": len(changed),
        "two_candidate_changed_ids": changed,
        "judge_fallbacks": judge_fallbacks,
        "priority_invariants_demoted_without_independent_support": demoted,
        "source_blueprint_rare_high_impact_terms": True,
        "source_canonical_entity_locks": True,
        "delta_audit_findings": len(fresh_rows),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
        "quality_mode": "single-broad-qe+diverse-deterministic-router+2way-retranslation+source-canonical-locks+single-delta-audit",
    }
    stats.update(_V9X_STATS)
    print("[bookai-v9x] " + json.dumps(_V9X_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9x():
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9x-stable-deterministic-router",
        "gold_reference_available_to_pipeline": False,
        "priority_policy": "legacy pre/post priority retries disabled; one deterministic post-QE owner",
        "risk_policy": "diverse deterministic semantic classes; no LLM challenge scan",
        "repair_policy": "max-8 2-way fresh retranslations + source-aware judge; safe fresh fallback only if judge unavailable",
        "entity_policy": "source-only blueprint canonical beats repeated draft typo",
        "blueprint_policy": "recurring terms plus rare high-impact specialized lexical items; no extra call",
        "reliability": "provider retries + reduced burst concurrency configured by workflow",
    }
    data["v9x_stats"] = dict(_V9X_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    # Version-specific source intelligence and base-QE ownership.
    v6._parse_source_memory = _parse_source_memory_v9x
    v9t._BASE_QUALITY = _base_quality_no_priority

    v9._configure_v9()
    v9._MATERIALS["coal"] = ("угл",)
    v9s._deterministic_v9s = v9u._deterministic
    v9._deterministic_findings = v9u._deterministic
    v9._thread_findings = v9s._thread_v9s
    v8._fix_near_entity_typos = v9s._entity_fix_v9s
    v9._format_dialogue_v9 = v9s._format_dialogue_v9s
    v8._normalize_dialogue_v8 = v9s._format_dialogue_v9s
    v3._semantic_short_repair = _quality_v9x
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v9r.v9j._annotate_v9j()
        v9r.v9k._annotate_v9k()
        v9r._annotate_v9r()
        v9s._annotate_v9s()
        v9t._annotate_v9t()
        _annotate_v9x()


if __name__ == "__main__":
    main()
