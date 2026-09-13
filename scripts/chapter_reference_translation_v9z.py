from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9y as v9y
import rapid_reference_translation as rapid

v9x = v9y.v9x
v9v, v9u, v9t = v9x.v9v, v9x.v9u, v9x.v9t
v9w, v9s, v9, v8, v6, v3 = v9x.v9w, v9x.v9s, v9x.v9, v9x.v8, v9x.v6, v9x.v3

_BASE_PARSE = v9x._parse_source_memory_v9x
_BASE_ROUTE = v9x._route_v9x
_BASE_QUALITY = v9x._quality_v9x
_V9Z_STATS: dict[str, Any] = {}
_LEGAL_CACHE: dict[str, dict[str, Any]] = {}

_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:[-'][A-Z]?[a-z]+)?\b")
_NAME_STOP = {
    "The", "This", "That", "Then", "There", "When", "While", "With", "Without", "What", "Why", "How",
    "He", "She", "His", "Her", "They", "Their", "It", "Its", "But", "And", "Or", "If", "As", "At", "By",
    "For", "From", "In", "Into", "No", "Not", "Of", "On", "So", "To", "Was", "Were", "Chapter", "Monday",
    "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "City", "Guild", "Specification", "Duke",
    "Count", "Sir", "Lord", "Lady", "Republic", "Army", "Court", "Committee", "Master", "Doctor", "Captain",
}
_LIVING_RE = re.compile(r"\b(?:i(?:'ve| have)?\s+got\s+a\s+living\s+to\s+make|make\s+a\s+living)\b", re.I)
_INDULGE_RE = re.compile(r"\bindulge\s+me\b", re.I)
_TOO_RE = re.compile(r"\bcan(?:not|'t)\s+be\s+too\b", re.I)
_ROLE_RE = re.compile(r"\b(?:advocate|counsel|prosecutor|attorney)\b", re.I)


def _parse_source_memory_v9z(provider, sample: str):
    """v9x book intelligence plus one cheap source-only lexical sentinel.

    The second pass is intentionally lexical, not stylistic: it catches rare
    armour/tool/plant/mechanism/legal terms that a book-level summary can omit.
    No target-language reference or gold translation is available.
    """
    memory = _BASE_PARSE(provider, sample)
    system = """Extract a HIGH-PRECISION source-only EN→RU lexical safety glossary from this English book sample.
No Russian reference translation exists. Do not translate passages. Return only concrete lexical items whose wrong
sense would materially corrupt a scene, especially rare historical armour/weapons, tools, crafts, mechanisms,
plants/fruits/species, materials/processes, measurements and legal/official role nouns. Include one-off terms when
specialized or polysemous. Preserve semantic class exactly: armour must stay the correct armour type; a fruit/plant
must stay a fruit/plant; a tool part such as the eye of a needle must be translated as that physical part rather than
an anatomical eye. Prefer precise standard Russian terminology over a generic near-synonym. Omit uncertain items.
Use the exact English source span as the key. ONLY JSON {"glossary":{"English term":"precise Russian term"}}.
Keep at most 80 entries."""
    added = 0
    try:
        obj = v6.extract_json(provider.complete(system, "SOURCE_ONLY_SAMPLE:\n" + sample, temperature=0.0))
        raw = dict(obj.get("glossary") or {}) if isinstance(obj, dict) else {}
        for src, ru in raw.items():
            src = v9._norm_text(src)
            ru = v9._norm_text(ru)
            if not src or not ru or len(src) < 3 or len(ru) < 2:
                continue
            if src not in memory.glossary:
                memory.glossary[src] = ru
                added += 1
            if len(memory.glossary) >= 170:
                break
    except Exception as exc:
        print(f"[v9z-lexical-sentinel] error={type(exc).__name__}", flush=True)
    print(f"[v9z-lexical-sentinel] added={added} glossary={len(memory.glossary)} source_only=true", flush=True)
    return memory


def _candidate_bad_source_only(segment, candidate: str, memory, source_segments: list):
    issues = v3.enhanced_candidate_issues(segment, candidate, memory, source_segments=source_segments)
    if v3.hard_ids(issues):
        return True
    glossary = dict(getattr(memory, "glossary", {}) or {})
    return bool(v3.locked_glossary_violations([segment], {segment.id: candidate}, glossary))


def _source_name_items(targets) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    contexts: dict[str, list[str]] = {}
    for segment in targets:
        text = str(segment.text or "")
        for token in _NAME_RE.findall(text):
            if token in _NAME_STOP:
                continue
            counts[token] += 1
            bucket = contexts.setdefault(token, [])
            if len(bucket) < 3:
                bucket.append(text[:900])
    limit = max(8, int(os.getenv("BOOKAI_V9Z_NAME_MAX") or "36"))
    rows = []
    for source, count in counts.most_common(limit * 2):
        if count < 2:
            continue
        rows.append({"source": source, "occurrences": count, "source_contexts": contexts.get(source, [])})
        if len(rows) >= limit:
            break
    return rows


def _learn_name_canon(harness, targets, translated, memory) -> tuple[dict[str, str], int]:
    items = _source_name_items(targets)
    if not items:
        return {}, 0
    system = """Create a conservative SOURCE-ONLY canonical Russian spelling registry for recurring proper names in an
English literary text. You have no published/reference translation and must not pretend otherwise. Use only source
spelling and source contexts. Return dictionary-form Russian spellings, not case-inflected forms. Preserve visible
source syllables and vowel sequences conservatively for invented names; do not silently drop letters/syllables merely
because an English pronunciation might allow it. Ordinary English words/titles accidentally included in the list must
be omitted. Prefer a stable transparent transliteration over a clever guess. ONLY JSON
{"canonicals":[{"source":"...","ru":"...","confidence":0.0}]}."""
    learned: dict[str, str] = {}
    try:
        obj = v8._complete_json(harness.gate, system, {"names": items})
        raw = obj.get("canonicals") or []
    except Exception as exc:
        print(f"[v9z-name-canon] error={type(exc).__name__}", flush=True)
        raw = []
    allowed = {row["source"] for row in items}
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "").strip()
        ru = v9._norm_text(row.get("ru") or "")
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        if src not in allowed or confidence < 0.85 or not ru or len(ru.split()) > 3:
            continue
        if not re.search(r"[А-Яа-яЁё]", ru):
            continue
        learned[src] = ru
        memory.glossary[src] = ru
        desc = str(memory.characters.get(src) or "")
        if desc:
            if re.search(r"\bru=[^;]+", desc):
                memory.characters[src] = re.sub(r"\bru=[^;]+", f"ru={ru}", desc, count=1)
            else:
                memory.characters[src] = f"ru={ru};" + desc

    fixes = 0
    if learned:
        for segment in targets:
            current = str(translated.get(segment.id) or "")
            value, count = v9s._entity_fix_v9s(segment, current, memory)
            if count:
                translated[segment.id] = value
                fixes += count
    print(f"[v9z-name-canon] learned={len(learned)} entity_fixes={fixes} source_only=true", flush=True)
    return learned, fixes


def _route_v9z(targets, translated, memory):
    _, ranked = _BASE_ROUTE(targets, translated, memory)
    by_id = {segment.id: segment for segment in targets}
    recoded = []
    for raw in ranked:
        row = dict(raw)
        source = str(by_id[row["id"]].text or "")
        if _ROLE_RE.search(source):
            row["code"] = "legal_role"
            row["priority"] = max(int(row["priority"]), 16)
        elif _LIVING_RE.search(source):
            row["code"] = "livelihood"
            row["priority"] = max(int(row["priority"]), 15)
        elif _INDULGE_RE.search(source):
            row["code"] = "request_pragmatics"
            row["priority"] = max(int(row["priority"]), 15)
        elif _TOO_RE.search(source):
            row["code"] = "polarity_idiom"
            row["priority"] = max(int(row["priority"]), 15)
        hits = list(row.get("glossary_hits") or [])
        if len(hits) >= 2:
            row["code"] = "specialized_lexicon"
            row["priority"] = max(int(row["priority"]), 15)
        recoded.append(row)

    ranked = sorted(recoded, key=lambda r: (-int(r["priority"]), int(r["index"])))
    cap = max(1, int(os.getenv("BOOKAI_V9X_RETRANSLATE_MAX") or "12"))
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_codes: set[str] = set()
    for row in ranked:
        if row["code"] in seen_codes:
            continue
        selected.append(row)
        seen_ids.add(row["id"])
        seen_codes.add(row["code"])
        if len(selected) >= cap:
            break
    if len(selected) < cap:
        for row in ranked:
            if row["id"] in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(row["id"])
            if len(selected) >= cap:
                break
    selected.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return selected, ranked


def _legal_role_analysis(harness, targets, translated, index: int) -> dict[str, Any]:
    segment = targets[index]
    if segment.id in _LEGAL_CACHE:
        return dict(_LEGAL_CACHE[segment.id])
    item = {
        "id": segment.id,
        "source": segment.text,
        "before_en": [x.text for x in targets[max(0, index - 3):index]],
        "after_en": [x.text for x in targets[index + 1:index + 4]],
    }
    system = """Infer the FUNCTION of the speaker's legal/courtroom role from English source context only. Do not trust the
surface dictionary sense of advocate/counsel/attorney. Decide whether this speaker is functionally accusing/prosecuting,
defending the accused, or neutral/unknown from what the speaker actually argues and whose case they advance. Then give
the most natural Russian role noun in nominative singular for that function. No reference translation is available.
ONLY JSON {"function":"prosecution|defense|neutral|unknown","preferred_ru_role":"...","confidence":0.0,"reason":"..."}."""
    try:
        obj = v8._complete_json(harness.gate, system, item)
    except Exception as exc:
        print(f"[v9z-legal-role] id={segment.id} error={type(exc).__name__}", flush=True)
        obj = {}
    function = str(obj.get("function") or "unknown").lower()
    role = v9._norm_text(obj.get("preferred_ru_role") or "")
    try:
        confidence = float(obj.get("confidence") or 0)
    except Exception:
        confidence = 0.0
    if function not in {"prosecution", "defense", "neutral", "unknown"}:
        function = "unknown"
    out = {"function": function, "preferred_ru_role": role, "confidence": confidence, "reason": v9._norm_text(obj.get("reason") or "")[:400]}
    _LEGAL_CACHE[segment.id] = dict(out)
    return out


def _fresh_candidates_v9z(harness, payload):
    legal = payload.get("legal_role_analysis") or {}
    if not legal or float(legal.get("confidence") or 0) < 0.80 or not legal.get("preferred_ru_role"):
        return v9v._two_candidates(harness, payload)
    system = """Retranslate ONE EN segment into publication-quality Russian. SOURCE and neighboring English context are
authoritative. A separate source-only role analysis has already inferred the speaker's courtroom FUNCTION. Respect that
function and use preferred_ru_role for the role noun; do not fall back to a dictionary gloss that reverses prosecution
and defense. Return exactly two COMPLETE fresh alternatives, preserving all facts, polarity, numbers, relations,
ellipsis, names and valid glossary terms. Natural literary Russian, no added facts. ONLY JSON
{"faithful_literary":"...","contextual_literary":"..."}."""
    try:
        obj = v8._complete_json(harness.gate, system, payload)
    except Exception as exc:
        print(f"[v9z-legal-candidates] {payload.get('id')} {type(exc).__name__}", flush=True)
        return []
    out = []
    for key in ("faithful_literary", "contextual_literary"):
        value = v9._norm_text(obj.get(key) or "")
        if value and value not in out:
            out.append(value)
    return out


def _repair_direct_v9z(harness, targets, translated, memory, selected):
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
        names, glossary = v9x._canonical_locks(segment, current, memory, recurring)

        if route.get("code") == "legal_role":
            legal = _legal_role_analysis(harness, targets, translated, i)
            payload["legal_role_analysis"] = legal
            if float(legal.get("confidence") or 0) >= 0.80 and legal.get("preferred_ru_role"):
                glossary = sorted(set([*glossary, str(legal["preferred_ru_role"])]))
        payload["hard_locks"] = {"proper_names": names, "glossary_terms": glossary}

        fresh = _fresh_candidates_v9z(harness, payload)
        choices = [current]
        for value in fresh:
            if v9._fatal_count(segment, value, memory) > v9._fatal_count(segment, current, memory):
                continue
            if not v9v._preserves_locks(value, names, glossary):
                print(f"[v9z-lock-reject] id={sid}", flush=True)
                continue
            choices.append(value)
        dedup, seen = [], set()
        for value in choices:
            key = v9._norm_text(value)
            if key and key not in seen:
                seen.add(key)
                dedup.append(value)
        if len(dedup) < 2:
            return sid, "", False
        idx, fallback = v9x._judge_v9x(harness, payload, dedup)
        chosen = dedup[idx]
        if v9._norm_text(chosen) == v9._norm_text(current):
            return sid, "", fallback
        return sid, chosen, fallback

    results, fallback_count = {}, 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9z-repair") as pool:
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


def _blind_risk_verify(harness, targets, translated, memory):
    _, ranked = _route_v9z(targets, translated, memory)
    cap = max(4, int(os.getenv("BOOKAI_V9Z_VERIFY_MAX") or "24"))
    alleged = []
    route_by_id: dict[str, dict[str, Any]] = {}
    for route in ranked[:cap]:
        row = dict(v9w._issue_row(route))
        row["severity"] = "major"
        row["confidence"] = max(0.90, float(row.get("confidence") or 0))
        row["reason"] = "blind final risk verification: " + str(route.get("reason") or "")
        alleged.append(row)
        route_by_id.setdefault(str(route["id"]), route)
    confirmed = v9u._verify(harness, targets, translated, alleged)
    ids = []
    for row in confirmed:
        sid = str(row.get("id") or "")
        if sid in route_by_id and sid not in ids:
            ids.append(sid)
    routes = [route_by_id[sid] for sid in ids]
    routes.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return confirmed, routes, len(alleged)


def _refresh_after_second_wave(harness, targets, translated, memory, changed_ids: set[str]):
    if not changed_ids:
        return None
    for sid in changed_ids:
        segment = next((s for s in targets if s.id == sid), None)
        if segment is not None:
            translated[sid] = v9s._format_dialogue_v9s(segment, translated.get(sid, ""))[0]
    semantic = {sid: v9t._semantic_rows(rows) for sid, rows in v9._V9_FINAL_FINDINGS.items()}
    fresh = v9t._selected_audit(harness, targets, translated, memory, changed_ids)
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in fresh:
        by_id.setdefault(str(row.get("id") or ""), []).append(dict(row))
    for sid in changed_ids:
        semantic[sid] = v9t._semantic_rows(by_id.get(sid, []))
    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]
    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)
    return {
        "delta_audit_findings": len(fresh),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
    }


def _quality_v9z(harness, targets, translated, memory):
    global _V9Z_STATS
    learned, entity_fixes = _learn_name_canon(harness, targets, translated, memory)
    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    confirmed, routes, verified_count = _blind_risk_verify(harness, targets, translated, memory)
    second_cap = max(0, int(os.getenv("BOOKAI_V9Z_SECOND_REPAIR_MAX") or "6"))
    routes = routes[:second_cap]
    second_changed, second_fallbacks = _repair_direct_v9z(harness, targets, translated, memory, routes) if routes else ([], 0)
    refreshed = _refresh_after_second_wave(harness, targets, translated, memory, set(second_changed)) or {}

    _V9Z_STATS = {
        "source_only_runtime": True,
        "reference_seed_disabled": True,
        "lexical_sentinel": True,
        "name_canon_learned": learned,
        "name_entity_fixes": entity_fixes,
        "router_cap": int(os.getenv("BOOKAI_V9X_RETRANSLATE_MAX") or "12"),
        "blind_final_verified_candidates": verified_count,
        "blind_final_confirmed": len(confirmed),
        "blind_final_confirmed_ids": [str(row.get("id") or "") for row in confirmed],
        "second_wave_selected": len(routes),
        "second_wave_selected_ids": [str(row.get("id") or "") for row in routes],
        "second_wave_changed": len(second_changed),
        "second_wave_changed_ids": second_changed,
        "second_wave_judge_fallbacks": second_fallbacks,
        **refreshed,
        "quality_mode_v9z": "source-only-canon+rare-lexicon+class-quota-router+legal-function+blind-risk-verifier",
    }
    stats.update(_V9Z_STATS)
    print("[bookai-v9z] " + json.dumps(_V9Z_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9z():
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9z-source-only-semantic-hardening",
        "gold_reference_available_to_pipeline": False,
        "reference_seed_runtime": False,
        "router_policy": "class quotas + 12 targeted routes",
        "lexical_policy": "one source-only rare-specialized lexical sentinel",
        "entity_policy": "source-only recurring-name canonical registry + conservative inflection-aware normalization",
        "legal_policy": "source-context role-function classification before retranslation",
        "final_verifier": "blind source-vs-final risk verification, bounded second repair wave",
    }
    data["v9z_stats"] = dict(_V9Z_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def _disable_reference_runtime() -> None:
    # The benchmark must be genuinely blind. Older reference-oriented modules
    # still carry historical seed/profile hooks, so neutralize them at runtime.
    hybrid = v3.hybrid
    hybrid.REFERENCE_GLOSSARY_SEED = {}
    hybrid._base_memory = lambda: v6.BookMemory()
    rapid.REFERENCE_GLOSSARY_SEED = {}
    v3.REFERENCE_GLOSSARY_SEED = {}
    v3._candidate_bad_v3 = _candidate_bad_source_only

    refmod = v9y.reference_harness
    refmod.ReferenceTranslationHarness.analyze = lambda self, sample: refmod.TranslationHarness.analyze(self, sample)
    refmod.ReferenceTranslationHarness.update_memory = (
        lambda self, originals, translated, memory: refmod.TranslationHarness.update_memory(self, originals, translated, memory)
    )


def main() -> None:
    _disable_reference_runtime()
    v9x._parse_source_memory_v9x = _parse_source_memory_v9z
    v9x._route_v9x = _route_v9z
    v9x._repair_direct_v9x = _repair_direct_v9z
    v9x._quality_v9x = _quality_v9z
    try:
        v9y.main()
    finally:
        _annotate_v9z()


if __name__ == "__main__":
    main()
