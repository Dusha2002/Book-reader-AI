from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chapter_reference_translation_v9aa as v9aa
from bookai.gigachat_v3 import GigaChatLightningV3Backend

v9z = v9aa.v9z
v9x = v9z.v9x
v9v, v9u, v9t = v9z.v9v, v9z.v9u, v9z.v9t
v9w, v9s, v9, v8, v6, v3 = v9z.v9w, v9z.v9s, v9z.v9, v9z.v8, v9z.v6, v9z.v3

_V9AB_STATS: dict[str, Any] = {}
_GIGA_ANALYST: GigaChatLightningV3Backend | None = None
_GIGA_ANALYST_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

# v9ab turns DeepSeek into the sparse independent specialist rather than the bulk
# reviewer. GigaChat does book-intelligence/canon and broad semantic screening;
# deterministic code owns known risk routing; DeepSeek only performs batched hard
# semantic repair + a short independent final verification. No Russian reference.

_EITHER_THEM_RE = re.compile(r"\b(?:either|neither|both)\s+of\s+(?:them|us|you)\b", re.I)
_ROUND_WOOD_RE = re.compile(r"\b(?:go|going|went|ride|rode|riding|walk|walking|head|heading|come|coming)\b[^.!?]{0,40}\b(?:the\s+)?[a-z]+\s+wood\b", re.I)
_HIGH_LEX_RE = re.compile(r"\b(?:brigandine|damson|darning[- ]needle|hedging tool)\b", re.I)


def _cache_root() -> Path:
    root = Path(os.getenv("BOOKAI_V8_SHARED_CACHE") or ".bookai-cache-v9ab")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _giga() -> GigaChatLightningV3Backend:
    global _GIGA_ANALYST
    if _GIGA_ANALYST is None:
        _GIGA_ANALYST = GigaChatLightningV3Backend()
    return _GIGA_ANALYST


def _parse_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b <= a:
            raise
        obj = json.loads(raw[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("GigaChat analyst returned non-object JSON")
    return obj


def _giga_json(system: str, payload: Any, *, max_tokens: int = 5000) -> dict[str, Any]:
    backend = _giga()
    client = backend._ensure_client()
    request = {
        "model": backend.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": max(1200, min(7000, int(max_tokens))),
    }
    response = client.chat(request)
    usage = backend._usage(response)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _GIGA_ANALYST_USAGE[key] += int(usage.get(key) or 0)
    _GIGA_ANALYST_USAGE["api_calls"] += 1
    return _parse_json(str(response.choices[0].message.content or ""))


def _parse_source_memory_v9ab(provider, sample: str):
    """Do not spend DeepSeek on book analysis.

    First-chapter draft starts with conservative defaults. Immediately after the
    draft, one cached GigaChat call builds the book-wide source-only intelligence;
    subsequent chapters reuse it before/during translation through persisted memory.
    """
    print("[v9ab-book-intelligence] deepseek_skipped=true deferred_to_gigachat=true", flush=True)
    return v6.BookMemory()


def _book_intelligence_path() -> Path:
    return _cache_root() / "giga-source-intelligence-v9ab.json"


def _source_sample(limit: int = 32000) -> str:
    rows = []
    size = 0
    for segment in list(getattr(v6, "_SOURCE_SEGMENTS", []) or []):
        text = str(segment.text or "").strip()
        if not text:
            continue
        if size + len(text) > limit:
            remain = max(0, limit - size)
            if remain:
                rows.append(text[:remain])
            break
        rows.append(text)
        size += len(text)
    return "\n".join(rows)


def _ensure_giga_book_intelligence(memory) -> dict[str, Any]:
    path = _book_intelligence_path()
    data: dict[str, Any] = {}
    cache_hit = False
    try:
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            cache_hit = isinstance(data, dict) and bool(data)
    except Exception:
        data = {}
    if not data:
        names = v9aa._whole_book_name_items()
        system = """Build compact SOURCE-ONLY EN→RU translation intelligence for an English novel. No Russian reference exists.
This is a cheap first-pass assistant, not a semantic judge. Return ONLY JSON with:
{"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},
 "glossary":{"exact English term":"precise Russian term"},
 "canonicals":{"English proper name":"stable Russian spelling"},
 "characters":{"English name":"gender=male|female|unknown;role=...;voice=..."},
 "summary":"..."}.
Prioritize recurring names/places/institutions plus rare high-impact armour, tools, mechanisms, plants/fruits,
materials and legal terms visible in the sample. Do not lock ordinary polysemous words. For invented names preserve
visible source vowel/consonant sequences conservatively. Keep glossary <=90, canonicals <=70, characters <=35.
Do not translate passages."""
        try:
            data = _giga_json(system, {"source_sample": _source_sample(), "recurring_name_candidates": names}, max_tokens=6500)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception as exc:
            print(f"[v9ab-giga-intelligence] error={type(exc).__name__}", flush=True)
            data = {}

    style = dict(data.get("style") or {}) if isinstance(data, dict) else {}
    memory.style.narrative_voice = str(style.get("narrative_voice") or memory.style.narrative_voice)
    memory.style.rhythm = str(style.get("rhythm") or memory.style.rhythm)
    memory.style.dialogue = str(style.get("dialogue") or memory.style.dialogue)
    memory.style.humor = str(style.get("humor") or memory.style.humor)

    glossary = dict(data.get("glossary") or {}) if isinstance(data, dict) else {}
    canon = dict(data.get("canonicals") or {}) if isinstance(data, dict) else {}
    chars = dict(data.get("characters") or {}) if isinstance(data, dict) else {}
    for src, ru in {**glossary, **canon}.items():
        src_n, ru_n = v9._norm_text(src), v9._norm_text(ru)
        if src_n and ru_n:
            memory.glossary[src_n] = ru_n
    for src, desc in chars.items():
        src_n, desc_n = str(src or "").strip(), v9._norm_text(desc)
        if src_n and desc_n:
            ru = v9._norm_text(canon.get(src_n) or "")
            memory.characters[src_n] = (f"ru={ru};" if ru else "") + desc_n
    summary = v9._norm_text(data.get("summary") or "") if isinstance(data, dict) else ""
    if summary:
        memory.rolling_summary = summary[:7000]

    print(
        f"[v9ab-giga-intelligence] cache_hit={str(cache_hit).lower()} glossary={len(glossary)} canon={len(canon)} characters={len(chars)}",
        flush=True,
    )
    return {"cache_hit": cache_hit, "glossary": len(glossary), "canon": len(canon), "characters": len(chars)}


def _apply_name_canon(targets, translated, memory) -> int:
    fixes = 0
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        value, count = v9s._entity_fix_v9s(segment, current, memory)
        if count:
            translated[segment.id] = value
            fixes += count
    return fixes


def _route_v9ab(targets, translated, memory):
    _, ranked0 = v9aa._route_v9aa(targets, translated, memory)
    rows = []
    by_id = {s.id: s for s in targets}
    for raw in ranked0:
        row = dict(raw)
        source = str(by_id[row["id"]].text or "")
        code = str(row.get("code") or "")
        priority = int(row.get("priority") or 0)
        if v9aa._ROLE_RE.search(source):
            code, priority = "legal_role", max(priority, 23)
        elif v9aa._COMPARISON_RE.search(source):
            code, priority = "comparison", max(priority, 23)
        elif _EITHER_THEM_RE.search(source):
            code, priority = "paired_referent", max(priority, 23)
        elif v9aa._LEARNING_RE.search(source):
            code, priority = "learning", max(priority, 22)
        elif v9aa._LIVING_RE.search(source):
            code, priority = "livelihood", max(priority, 22)
        elif _ROUND_WOOD_RE.search(source) or v9aa._PLACE_WOOD_RE.search(source) or v9aa._TOOL_EYE_RE.search(source):
            code, priority = "word_sense", max(priority, 22)
        elif _HIGH_LEX_RE.search(source):
            code, priority = "specialized_lexicon", max(priority, 22)
        elif v9aa._INDULGE_RE.search(source):
            code, priority = "request_pragmatics", max(priority, 20)
        elif v9aa._TOO_RE.search(source):
            code, priority = "polarity_idiom", max(priority, 20)
        elif v9aa._ORDINAL_RE.search(source):
            code, priority = "ordinal", max(priority, 20)
        row["code"], row["priority"] = code, priority
        rows.append(row)
    ranked = sorted(rows, key=lambda r: (-int(r["priority"]), int(r["index"])))

    cap = max(8, int(os.getenv("BOOKAI_V9AB_DEEP_REPAIR_MAX") or "15"))
    quotas = [
        ("legal_role", 3), ("comparison", 1), ("paired_referent", 2), ("learning", 1),
        ("livelihood", 1), ("word_sense", 2), ("request_pragmatics", 1),
        ("polarity_idiom", 1), ("ordinal", 1), ("specialized_lexicon", 3),
    ]
    selected, seen = [], set()
    for code, wanted in quotas:
        n = 0
        for row in ranked:
            if row["id"] in seen or row.get("code") != code:
                continue
            selected.append(row); seen.add(row["id"]); n += 1
            if len(selected) >= cap or n >= wanted:
                break
        if len(selected) >= cap:
            break
    if len(selected) < cap:
        for row in ranked:
            if row["id"] in seen:
                continue
            selected.append(row); seen.add(row["id"])
            if len(selected) >= cap:
                break
    selected.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return selected, ranked


def _screen_batches(targets, translated, memory) -> list[dict[str, Any]]:
    """Cheap broad semantic screen over the entire chapter using GigaChat.

    Only publication-level meaning defects are returned. DeepSeek never sees the
    hundreds of items that GigaChat and deterministic checks consider safe.
    """
    max_items = max(12, int(os.getenv("BOOKAI_V9AB_GIGA_SCREEN_ITEMS") or "26"))
    max_chars = max(7000, int(os.getenv("BOOKAI_V9AB_GIGA_SCREEN_CHARS") or "18000"))
    batches, current, chars = [], [], 0
    for i, segment in enumerate(targets):
        source = str(segment.text or "")
        ru = str(translated.get(segment.id) or "")
        item = {"id": segment.id, "source": source, "ru": ru}
        size = len(source) + len(ru)
        if current and (len(current) >= max_items or chars + size > max_chars):
            batches.append(current); current, chars = [], 0
        current.append(item); chars += size
    if current:
        batches.append(current)

    system = """Screen EN→RU literary translation for MAJOR SEMANTIC defects only. You are the same model family as the draft,
so be conservative: do not rewrite style and do not report harmless paraphrase. Report only likely meaning errors such
as reversed actor/action/object, wrong number/comparison/polarity, lost antecedent or relative, wrong legal function,
wrong specialist word sense, omitted proposition, invented proposition, or a proper name inconsistency. Return ONLY
JSON {"suspects":[{"id":"s000000","confidence":0.0,"code":"...","reason":"..."}]}.
Return an empty suspects list when no major defect is visible. Never output corrected translations."""
    suspects = []
    for n, batch in enumerate(batches, 1):
        try:
            obj = _giga_json(system, {"items": batch}, max_tokens=3200)
            raw = obj.get("suspects") or []
        except Exception as exc:
            print(f"[v9ab-giga-screen] batch={n}/{len(batches)} error={type(exc).__name__}", flush=True)
            raw = []
        allowed = {x["id"] for x in batch}
        for row in raw if isinstance(raw, list) else []:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "")
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if sid not in allowed or confidence < 0.72:
                continue
            suspects.append({
                "id": sid,
                "confidence": confidence,
                "code": str(row.get("code") or "giga_semantic_screen"),
                "reason": v9._norm_text(row.get("reason") or "")[:500],
            })
    print(f"[v9ab-giga-screen] batches={len(batches)} suspects={len(suspects)}", flush=True)
    return suspects


def _merge_repair_routes(targets, translated, memory, giga_suspects):
    selected, ranked = _route_v9ab(targets, translated, memory)
    by_id = {row["id"]: dict(row) for row in ranked}
    index = {s.id: i for i, s in enumerate(targets)}
    seen = {row["id"] for row in selected}
    extra_cap = max(0, int(os.getenv("BOOKAI_V9AB_GIGA_EXTRA_MAX") or "5"))
    extras = sorted(giga_suspects, key=lambda x: -float(x.get("confidence") or 0))
    for suspect in extras:
        if extra_cap <= 0:
            break
        sid = suspect["id"]
        if sid in seen or sid not in index:
            continue
        row = by_id.get(sid) or {
            "id": sid, "index": index[sid], "priority": 16,
            "code": suspect.get("code") or "giga_semantic_screen",
            "reason": suspect.get("reason") or "GigaChat broad screen suspects semantic defect",
            "glossary_hits": [],
        }
        row = dict(row)
        row["priority"] = max(int(row.get("priority") or 0), 16)
        row["reason"] = (str(row.get("reason") or "") + "; GigaChat screen: " + str(suspect.get("reason") or "")).strip("; ")
        selected.append(row); seen.add(sid); extra_cap -= 1
    selected.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return selected, ranked


def _payload_item(targets, translated, memory, route):
    i = int(route["index"])
    segment = targets[i]
    return {
        "id": segment.id,
        "risk_code": route.get("code"),
        "risk_reason": route.get("reason"),
        "source": segment.text,
        "current_ru": str(translated.get(segment.id) or ""),
        "before_en": [x.text for x in targets[max(0, i - 2):i]],
        "after_en": [x.text for x in targets[i + 1:i + 3]],
        "glossary_hints": route.get("glossary_hits") or [],
    }


def _deepseek_batched_repair(harness, targets, translated, memory, routes):
    if not routes:
        return [], 0
    batch_size = max(3, int(os.getenv("BOOKAI_V9AB_DEEP_BATCH") or "7"))
    system = """You are the sparse independent semantic specialist in an EN→RU book pipeline. For EVERY supplied item, compare
English SOURCE + neighboring English context against current_ru and its risk_code. Fidelity outranks fluency.
Pay special attention to: courtroom FUNCTION rather than dictionary title; exact comparison direction and numbers;
'either/neither/both' antecedents; explicit learning/acquisition; need to EARN a living; pragmatic requests; polarity;
proper-place vs common-noun senses; exact specialist armour/tool/plant/mechanism/object-part denotation. Glossary hints
are advisory and may be wrong. If current_ru is semantically publication-ready, keep it unchanged. If not, return one
complete corrected Russian translation of exactly that source segment. Preserve established proper names unless the
source clearly requires inflection. ONLY JSON
{"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}.
Return exactly one row for every input id."""
    changed = []
    calls = 0
    by_id = {s.id: s for s in targets}
    recurring = v9v._recurring_name_tokens(targets, translated)
    for start in range(0, len(routes), batch_size):
        batch_routes = routes[start:start + batch_size]
        items = [_payload_item(targets, translated, memory, row) for row in batch_routes]
        expected = {x["id"] for x in items}
        try:
            obj = v8._complete_json(harness.gate, system, {"items": items})
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ab-deep-repair] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(row.get("id") or ""): row for row in raw if isinstance(row, dict)}
        for route in batch_routes:
            sid = route["id"]
            row = parsed.get(sid)
            if not row or sid not in expected or type(row.get("change")) is not bool or not row.get("change"):
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            threshold = 0.66 if route.get("code") in {"legal_role", "comparison", "paired_referent", "learning", "livelihood", "word_sense", "specialized_lexicon"} else 0.74
            candidate = v9._norm_text(row.get("corrected_ru") or "")
            if confidence < threshold or not candidate:
                continue
            segment = by_id[sid]
            current = str(translated.get(sid) or "")
            names, _ = v9x._canonical_locks(segment, current, memory, recurring)
            if v9._fatal_count(segment, candidate, memory) > v9._fatal_count(segment, current, memory):
                continue
            if names and not v9v._preserves_locks(candidate, names, []):
                print(f"[v9ab-name-lock-reject] id={sid}", flush=True)
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _deepseek_final_verify(harness, targets, translated, memory, routes):
    """Small independent final check. It can repair directly, avoiding candidate+judge loops."""
    if not routes:
        return [], 0, 0
    cap = max(8, int(os.getenv("BOOKAI_V9AB_FINAL_VERIFY_MAX") or "20"))
    routes = routes[:cap]
    batch_size = max(5, int(os.getenv("BOOKAI_V9AB_VERIFY_BATCH") or "10"))
    system = """FINAL independent EN→RU semantic check. Verify each supplied item against English source/context. Ignore stylistic
preferences; report only real meaning defects. Recheck the named risk_code exactly. If defective, provide a complete
corrected_ru for that one source segment. If correct, set ok=true. Glossary hints are not authoritative. Return exactly
one row per input id. ONLY JSON
{"checks":[{"id":"...","ok":true,"corrected_ru":"","confidence":0.0,"reason":"..."}]}.
"""
    changed, confirmed, calls = [], 0, 0
    by_id = {s.id: s for s in targets}
    recurring = v9v._recurring_name_tokens(targets, translated)
    for start in range(0, len(routes), batch_size):
        batch_routes = routes[start:start + batch_size]
        items = [_payload_item(targets, translated, memory, row) for row in batch_routes]
        expected = {x["id"] for x in items}
        try:
            obj = v8._complete_json(harness.gate, system, {"items": items})
            raw = obj.get("checks") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ab-final-verify] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(row.get("id") or ""): row for row in raw if isinstance(row, dict)}
        for route in batch_routes:
            sid = route["id"]
            row = parsed.get(sid)
            if not row or sid not in expected or type(row.get("ok")) is not bool or row.get("ok"):
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if confidence < 0.72:
                continue
            confirmed += 1
            candidate = v9._norm_text(row.get("corrected_ru") or "")
            if not candidate:
                continue
            segment = by_id[sid]
            current = str(translated.get(sid) or "")
            names, _ = v9x._canonical_locks(segment, current, memory, recurring)
            if v9._fatal_count(segment, candidate, memory) > v9._fatal_count(segment, current, memory):
                continue
            if names and not v9v._preserves_locks(candidate, names, []):
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, confirmed, calls


def _quality_v9ab(harness, targets, translated, memory):
    global _V9AB_STATS
    intel = _ensure_giga_book_intelligence(memory)
    name_fixes = _apply_name_canon(targets, translated, memory)

    giga_suspects = _screen_batches(targets, translated, memory)
    repair_routes, ranked = _merge_repair_routes(targets, translated, memory, giga_suspects)
    first_changed, deep_repair_calls = _deepseek_batched_repair(harness, targets, translated, memory, repair_routes)

    # Final verifier sees every mandatory repair route plus changed items. We do not
    # re-send the 80-100 glossary candidates that GigaChat already screened out.
    route_by_id = {row["id"]: dict(row) for row in repair_routes}
    for sid in first_changed:
        if sid not in route_by_id:
            i = next((i for i, s in enumerate(targets) if s.id == sid), 0)
            route_by_id[sid] = {"id": sid, "index": i, "priority": 20, "code": "changed", "reason": "post-repair verification", "glossary_hits": []}
    verify_routes = sorted(route_by_id.values(), key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)))
    second_changed, final_confirmed, deep_verify_calls = _deepseek_final_verify(harness, targets, translated, memory, verify_routes)

    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]

    # Deterministic invariants still run over every segment for free. No broad
    # DeepSeek QE is executed here.
    semantic: dict[str, list[dict[str, Any]]] = {}
    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)

    selected, all_ranked = _route_v9ab(targets, translated, memory)
    _V9AB_STATS = {
        "deepseek_bulk_qe_disabled": True,
        "book_intelligence_provider": "GigaChat-3-Lightning",
        "broad_semantic_screen_provider": "GigaChat-3-Lightning",
        "deepseek_role": "batched-hard-semantic-repair+independent-final-verifier-only",
        "book_intelligence": intel,
        "name_entity_fixes": name_fixes,
        "router_candidates": len(all_ranked),
        "router_mandatory_selected": len(selected),
        "deep_repair_routes": len(repair_routes),
        "giga_screen_suspects": len(giga_suspects),
        "first_wave_changed": len(first_changed),
        "first_wave_changed_ids": first_changed,
        "second_wave_changed": len(second_changed),
        "second_wave_changed_ids": second_changed,
        "final_verifier_confirmed": final_confirmed,
        "deep_repair_batches": deep_repair_calls,
        "deep_verify_batches": deep_verify_calls,
        "giga_analyst_usage": dict(_GIGA_ANALYST_USAGE),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
        "quality_mode": "giga-broad-screen+deterministic-router+batched-deepseek-minimum",
    }
    print("[bookai-v9ab] " + json.dumps(_V9AB_STATS, ensure_ascii=False), flush=True)
    return dict(_V9AB_STATS)


def _annotate_v9ab() -> None:
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9ab-minimal-deepseek",
        "gold_reference_available_to_pipeline": False,
        "bulk_qe": "GigaChat broad semantic screening + deterministic invariants",
        "book_intelligence": "GigaChat cached source-only",
        "deepseek_policy": "only batched hard semantic repair and bounded independent final verification",
        "cost_policy": "never send every segment/glossary candidate to DeepSeek",
    }
    data["v9ab_stats"] = dict(_V9AB_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    # Preserve v9aa's source-only/reference-leak safeguards but replace expensive
    # policy layers with the sparse-provider architecture.
    v9z._parse_source_memory_v9z = _parse_source_memory_v9ab
    v9aa._route_v9aa = _route_v9ab
    v9aa._quality_v9aa = _quality_v9ab
    try:
        v9aa.main()
    finally:
        _annotate_v9ab()


if __name__ == "__main__":
    main()
