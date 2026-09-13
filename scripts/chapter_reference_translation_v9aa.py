from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chapter_reference_translation_v9z as v9z

v9x = v9z.v9x
v9v, v9u, v9t = v9z.v9v, v9z.v9u, v9z.v9t
v9w, v9s, v9, v8, v6, v3 = v9z.v9w, v9z.v9s, v9z.v9, v9z.v8, v9z.v6, v9z.v3

_BASE_ROUTE = v9z._BASE_ROUTE
_BASE_QUALITY_Z = v9z._quality_v9z
_V9AA_STATS: dict[str, Any] = {}
_LEXICAL_CACHE: dict[str, dict[str, Any]] = {}

# v9aa addresses the concrete failures from the true source-only v9z run:
#   * legal/polysemous glossary suggestions must not become hard lexical locks;
#   * the proper-name canon is learned once from the whole English book and reused
#     across chapter processes;
#   * the bounded router reserves capacity for distinct semantic obligations instead
#     of letting one broad lexical class starve learning/referent/livelihood cases;
#   * context-sensitive place/word senses get an explicit source-only analysis;
#   * final verification MUST inspect every first-wave route plus every glossary
#     mismatch, rather than a generic top-N sample.
# No Russian gold/reference translation is available to this runtime.

_ROLE_RE = re.compile(r"\b(?:advocate|counsel|prosecutor|attorney)\b", re.I)
_LIVING_RE = re.compile(r"\b(?:i(?:'ve| have)?\s+got\s+a\s+living\s+to\s+make|make\s+a\s+living)\b", re.I)
_LEARNING_RE = re.compile(r"\btakes?\s+(?:a\s+)?(?:little\s+)?(?:while|time)\s+to\s+learn\b", re.I)
_PAIR_RE = re.compile(r"\b(?:either|neither|both)\b", re.I)
_INDULGE_RE = re.compile(r"\bindulge\s+me\b", re.I)
_TOO_RE = re.compile(r"\bcan(?:not|'t)\s+be\s+too\b", re.I)
_ORDINAL_RE = re.compile(r"\blast\s+[^.!?]{0,45}\s+but\s+one\b", re.I)
_COMPARISON_RE = re.compile(r"\boutnumber(?:ed|ing)?\b", re.I)
# Capitalized "X Wood" is a high-value general toponym cue in English prose. The
# model must infer woodland/place vs material/tree from context rather than translate
# the token Wood in isolation.
_PLACE_WOOD_RE = re.compile(r"\b[A-Z][A-Za-z'’-]{2,}\s+Wood\b")
_TOOL_EYE_RE = re.compile(r"\beye\s+of\s+(?:an?\s+|the\s+)?[^.!?]{0,45}\bneedle\b", re.I)

# These source terms are inherently function/sense dependent. Their glossary values
# may be useful hints, but must never be enforced as exact target strings.
_FLEXIBLE_TERM_RE = re.compile(
    r"^(?:advocate|counsel|attorney|prosecutor|wood|eye|case|charge|issue|office|master|"
    r"right|left|spring|bank|yard|court|appeal|practice|present)$",
    re.I,
)


def _source_mentions(source: str, key: str) -> bool:
    return bool(key and key.casefold() in source.casefold())


def _looks_like_name(source_key: str, memory) -> bool:
    key = str(source_key or "").strip()
    if key in dict(getattr(memory, "characters", {}) or {}):
        return True
    if " " not in key and key[:1].isupper() and key[1:].islower():
        return True
    return False


def _is_flexible_glossary_source(source_key: str, segment_text: str) -> bool:
    key = v9._norm_text(source_key)
    if not key:
        return False
    if _FLEXIBLE_TERM_RE.fullmatch(key):
        return True
    if _ROLE_RE.search(key):
        return True
    if key.casefold() == "wood" and "Wood" in segment_text:
        return True
    if "eye of" in key.casefold() and "needle" in key.casefold():
        return True
    return False


def _candidate_bad_source_only_v9aa(segment, candidate: str, memory, source_segments: list):
    """Keep deterministic safety, but do not hard-fail on context-dependent glossary hints."""
    issues = v3.enhanced_candidate_issues(segment, candidate, memory, source_segments=source_segments)
    if v3.hard_ids(issues):
        return True
    safe_glossary = {
        str(src): str(ru)
        for src, ru in dict(getattr(memory, "glossary", {}) or {}).items()
        if not _is_flexible_glossary_source(str(src), str(segment.text or ""))
    }
    return bool(v3.locked_glossary_violations([segment], {segment.id: candidate}, safe_glossary))


def _whole_book_name_items() -> list[dict[str, Any]]:
    segments = list(getattr(v6, "_SOURCE_SEGMENTS", []) or [])
    if not segments:
        return []
    counts: Counter[str] = Counter()
    contexts: dict[str, list[str]] = {}
    name_re = v9z._NAME_RE
    stop = v9z._NAME_STOP
    for segment in segments:
        text = str(segment.text or "")
        for token in name_re.findall(text):
            if token in stop:
                continue
            counts[token] += 1
            bucket = contexts.setdefault(token, [])
            if len(bucket) < 4:
                bucket.append(text[:1100])
    limit = max(24, int(os.getenv("BOOKAI_V9AA_NAME_MAX") or "64"))
    rows = []
    for source, count in counts.most_common(limit * 3):
        if count < 2:
            continue
        rows.append({"source": source, "occurrences": count, "source_contexts": contexts.get(source, [])})
        if len(rows) >= limit:
            break
    return rows


def _name_cache_path() -> Path:
    root = Path(os.getenv("BOOKAI_V8_SHARED_CACHE") or ".bookai-cache-v9aa")
    root.mkdir(parents=True, exist_ok=True)
    return root / "source-only-name-canon-v9aa.json"


def _load_or_learn_book_name_canon(harness) -> tuple[dict[str, str], bool]:
    path = _name_cache_path()
    try:
        obj = json.loads(path.read_text("utf-8")) if path.exists() else {}
        canon = {str(k): v9._norm_text(v) for k, v in dict(obj.get("canonicals") or {}).items() if str(k) and v9._norm_text(v)}
        if canon:
            print(f"[v9aa-name-canon] cache_hit entries={len(canon)}", flush=True)
            return canon, True
    except Exception:
        pass

    items = _whole_book_name_items()
    if not items:
        return {}, False
    system = """Build ONE book-wide SOURCE-ONLY Russian spelling canon for recurring proper names in an English novel.
No published/reference translation exists. Use the whole-book English occurrence census and contexts only.
Return dictionary-form Russian spellings, never case-inflected forms. Be conservative and information-preserving:
keep visible source vowel sequences and consonant clusters in invented names when Russian allows it; do not change a
final -ea into -eya merely to sound familiar; do not collapse a visible -tz- cluster to -ts- if doing so loses source
spelling information; do not silently drop syllables. Standard well-established transliteration rules may override
letter-by-letter rendering only when clear. Omit ordinary English words/titles accidentally present. Consistency across
the entire book is more important than local improvisation. ONLY JSON
{"canonicals":[{"source":"...","ru":"...","confidence":0.0}]}.
"""
    try:
        obj = v8._complete_json(harness.gate, system, {"names": items})
        raw = obj.get("canonicals") or []
    except Exception as exc:
        print(f"[v9aa-name-canon] error={type(exc).__name__}", flush=True)
        raw = []
    allowed = {row["source"] for row in items}
    canon: dict[str, str] = {}
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "").strip()
        ru = v9._norm_text(row.get("ru") or "")
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        if src not in allowed or confidence < 0.82 or not ru or len(ru.split()) > 3:
            continue
        if not re.search(r"[А-Яа-яЁё]", ru):
            continue
        canon[src] = ru
    if canon:
        path.write_text(json.dumps({"canonicals": canon}, ensure_ascii=False, indent=2), "utf-8")
    print(f"[v9aa-name-canon] learned_bookwide={len(canon)} source_only=true", flush=True)
    return canon, False


def _learn_name_canon_v9aa(harness, targets, translated, memory) -> tuple[dict[str, str], int]:
    canon, _ = _load_or_learn_book_name_canon(harness)
    for src, ru in canon.items():
        memory.glossary[src] = ru
        desc = str(memory.characters.get(src) or "")
        if desc:
            if re.search(r"\bru=[^;]+", desc):
                memory.characters[src] = re.sub(r"\bru=[^;]+", f"ru={ru}", desc, count=1)
            else:
                memory.characters[src] = f"ru={ru};" + desc
    fixes = 0
    if canon:
        for segment in targets:
            current = str(translated.get(segment.id) or "")
            value, count = v9s._entity_fix_v9s(segment, current, memory)
            if count:
                translated[segment.id] = value
                fixes += count
    print(f"[v9aa-name-canon] applied={len(canon)} entity_fixes={fixes}", flush=True)
    return canon, fixes


def _lexical_hits(segment, current: str, memory) -> list[dict[str, str]]:
    source = str(segment.text or "")
    hits: list[dict[str, str]] = []
    for src, ru in dict(getattr(memory, "glossary", {}) or {}).items():
        src_n, ru_n = v9._norm_text(src), v9._norm_text(ru)
        if not src_n or not ru_n or _looks_like_name(src_n, memory):
            continue
        if not _source_mentions(source, src_n):
            continue
        hits.append({
            "source": src_n,
            "suggested_ru": ru_n,
            "present": str(v8._entity_present(current, ru_n)).lower(),
        })
        if len(hits) >= 8:
            break
    return hits


def _route_v9aa(targets, translated, memory):
    _, base_ranked = _BASE_ROUTE(targets, translated, memory)
    by_id = {segment.id: segment for segment in targets}
    rows_by_id: dict[str, dict[str, Any]] = {str(row["id"]): dict(row) for row in base_ranked}

    for index, segment in enumerate(targets):
        sid = segment.id
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")
        row = rows_by_id.get(sid)
        hits = _lexical_hits(segment, current, memory)

        code = str(row.get("code") if row else "")
        priority = int(row.get("priority") if row else 0)
        reasons = [str(row.get("reason") or "")] if row else []

        if _ROLE_RE.search(source):
            code, priority = "legal_role", max(priority, 19)
            reasons.append("infer courtroom function from surrounding source; glossary label is advisory only")
        elif _LIVING_RE.search(source):
            code, priority = "livelihood", max(priority, 18)
            reasons.append("preserve need to earn a living, not possession of existing means")
        elif _LEARNING_RE.search(source):
            code, priority = "learning", max(priority, 18)
            reasons.append("preserve explicit learning/acquisition predicate")
        elif _PAIR_RE.search(source):
            code, priority = "paired_referent", max(priority, 18)
            reasons.append("resolve both/either/neither against the exact neighboring antecedent set")
        elif _PLACE_WOOD_RE.search(source) or _TOOL_EYE_RE.search(source):
            code, priority = "word_sense", max(priority, 18)
            reasons.append("infer contextual lexical sense / entity type before choosing Russian wording")
        elif _COMPARISON_RE.search(source):
            code, priority = "comparison", max(priority, 17)
        elif _INDULGE_RE.search(source):
            code, priority = "request_pragmatics", max(priority, 17)
        elif _TOO_RE.search(source):
            code, priority = "polarity_idiom", max(priority, 17)
        elif _ORDINAL_RE.search(source):
            code, priority = "ordinal", max(priority, 17)
        elif hits:
            # Validate specialist lexical senses even when the draft happens to match
            # the source-only glossary. The glossary itself may have chosen the wrong
            # dictionary sense, so matching it is not proof of correctness.
            code = "specialized_lexicon"
            priority = max(priority, 15 + min(3, len(hits)))
            reasons.append("independently validate specialist lexical denotation; do not trust glossary suggestion blindly")

        if not code:
            continue
        rows_by_id[sid] = {
            "id": sid,
            "index": index,
            "priority": priority,
            "code": code,
            "reason": "; ".join(x for x in dict.fromkeys(reasons) if x),
            "glossary_hits": hits or list(row.get("glossary_hits") or []) if row else hits,
        }

    ranked = sorted(rows_by_id.values(), key=lambda r: (-int(r["priority"]), int(r["index"])))
    cap = max(1, int(os.getenv("BOOKAI_V9X_RETRANSLATE_MAX") or "15"))

    # Reserved capacities are semantic obligations, not merely code diversity.
    # Unused slots automatically fall through to the global ranking.
    quotas = [
        ("legal_role", 2),
        ("learning", 1),
        ("paired_referent", 1),
        ("livelihood", 1),
        ("word_sense", 1),
        ("comparison", 1),
        ("request_pragmatics", 1),
        ("polarity_idiom", 1),
        ("ordinal", 1),
        ("specialized_lexicon", 4),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for code, wanted in quotas:
        taken = 0
        for row in ranked:
            if row["id"] in seen or row["code"] != code:
                continue
            selected.append(row)
            seen.add(row["id"])
            taken += 1
            if len(selected) >= cap or taken >= wanted:
                break
        if len(selected) >= cap:
            break
    if len(selected) < cap:
        for row in ranked:
            if row["id"] in seen:
                continue
            selected.append(row)
            seen.add(row["id"])
            if len(selected) >= cap:
                break
    selected.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    return selected, ranked


def _lexical_sense_analysis(harness, targets, translated, memory, index: int, route) -> dict[str, Any]:
    segment = targets[index]
    if segment.id in _LEXICAL_CACHE:
        return dict(_LEXICAL_CACHE[segment.id])
    glossary_hints = _lexical_hits(segment, str(translated.get(segment.id) or ""), memory)
    item = {
        "id": segment.id,
        "source": segment.text,
        "current_ru": str(translated.get(segment.id) or ""),
        "before_en": [x.text for x in targets[max(0, index - 3):index]],
        "after_en": [x.text for x in targets[index + 1:index + 4]],
        "source_only_glossary_hints": glossary_hints,
        "risk_class": route.get("code"),
    }
    system = """Resolve risky lexical senses in ONE English literary segment before EN→RU repair. Source and neighboring
English context are authoritative. The supplied source-only glossary hints are merely hypotheses and MAY BE WRONG.
Identify only lexical items whose exact denotation matters here: specialist armour/tools/plants/mechanisms/materials,
parts of objects, or context-sensitive proper-place/common-noun senses. For a capitalized place expression such as
'X Wood', decide from context whether Wood denotes woodland/a named place rather than timber or a single tree. For
'eye of ... needle', preserve the physical needle-part sense, not an anatomical eye. Give natural precise Russian
renderings in forms suitable as semantic guidance; inflection in the final sentence may differ. ONLY JSON
{"items":[{"source_span":"...","sense":"...","preferred_ru":"...","confidence":0.0}]}.
Return [] if no lexical correction/guidance is needed."""
    try:
        obj = v8._complete_json(harness.gate, system, item)
        raw = obj.get("items") or []
    except Exception as exc:
        print(f"[v9aa-lexical-analysis] id={segment.id} error={type(exc).__name__}", flush=True)
        raw = []
    out_items = []
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        span = v9._norm_text(row.get("source_span") or "")
        preferred = v9._norm_text(row.get("preferred_ru") or "")
        if confidence < 0.62 or not span or not preferred:
            continue
        out_items.append({
            "source_span": span[:180],
            "sense": v9._norm_text(row.get("sense") or "")[:300],
            "preferred_ru": preferred[:180],
            "confidence": confidence,
        })
    result = {"items": out_items}
    _LEXICAL_CACHE[segment.id] = dict(result)
    return result


def _fresh_candidates_v9aa(harness, payload):
    legal = payload.get("legal_role_analysis") or {}
    lexical = payload.get("lexical_sense_analysis") or {}
    if legal and float(legal.get("confidence") or 0) >= 0.70 and legal.get("preferred_ru_role"):
        system = """Retranslate ONE English segment into publication-quality Russian. SOURCE and neighboring source context
are authoritative. A source-only role analysis inferred the courtroom FUNCTION; follow that function semantically,
but inflect the Russian role noun naturally for the sentence. Do NOT preserve an old glossary/draft role label when it
conflicts with the inferred function. Preserve every other fact, polarity, number, relation, ellipsis and proper name.
Return exactly two COMPLETE alternatives. ONLY JSON
{"faithful_literary":"...","contextual_literary":"..."}."""
    elif lexical.get("items"):
        system = """Retranslate ONE English segment into publication-quality Russian. SOURCE and neighboring English context
are authoritative. A separate source-only lexical-sense analysis is supplied; use its denotational conclusions as
semantic guidance and inflect naturally. Existing glossary hints and current_ru are NOT hard constraints and may carry
the wrong dictionary sense. Preserve all facts, relations, numbers, polarity, names, technical precision and pragmatic
force. Return exactly two COMPLETE alternatives, one maximally faithful and one equally faithful but more literary.
ONLY JSON {"faithful_literary":"...","contextual_literary":"..."}."""
    else:
        return v9v._two_candidates(harness, payload)
    try:
        obj = v8._complete_json(harness.gate, system, payload)
    except Exception as exc:
        print(f"[v9aa-candidates] id={payload.get('id')} error={type(exc).__name__}", flush=True)
        return []
    out = []
    for key in ("faithful_literary", "contextual_literary"):
        value = v9._norm_text(obj.get(key) or "")
        if value and value not in out:
            out.append(value)
    return out


def _repair_direct_v9aa(harness, targets, translated, memory, selected):
    pos = {s.id: i for i, s in enumerate(targets)}
    recurring = v9v._recurring_name_tokens(targets, translated)
    workers = max(1, min(3, int(os.getenv("BOOKAI_V9X_RETRANSLATE_WORKERS") or "2"), len(selected) or 1))
    semantic_codes = {
        "legal_role", "word_sense", "specialized_lexicon", "learning", "paired_referent",
        "livelihood", "request_pragmatics", "polarity_idiom", "comparison", "ordinal",
    }

    def one(route):
        sid = route["id"]
        i = pos[sid]
        segment = targets[i]
        current = str(translated.get(sid) or "")
        issue = v9w._issue_row(route)
        payload = v9u._payload(targets, translated, memory, i, [issue])
        payload["router_reason"] = route.get("reason") or ""
        payload["glossary_hits"] = route.get("glossary_hits") or []
        names, glossary = v9x._canonical_locks(segment, current, memory, recurring)

        # Names remain hard continuity constraints. Context-dependent lexical values
        # become advisory for semantic repair so the repair is allowed to correct
        # a bad source-only glossary guess.
        if route.get("code") in semantic_codes:
            glossary = []

        if route.get("code") == "legal_role":
            payload["legal_role_analysis"] = v9z._legal_role_analysis(harness, targets, translated, i)
        if route.get("code") in {"word_sense", "specialized_lexicon"}:
            payload["lexical_sense_analysis"] = _lexical_sense_analysis(harness, targets, translated, memory, i, route)
        payload["hard_locks"] = {"proper_names": names, "glossary_terms": glossary}

        fresh = _fresh_candidates_v9aa(harness, payload)
        choices = [current]
        for value in fresh:
            if v9._fatal_count(segment, value, memory) > v9._fatal_count(segment, current, memory):
                continue
            if not v9v._preserves_locks(value, names, glossary):
                print(f"[v9aa-lock-reject] id={sid}", flush=True)
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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9aa-repair") as pool:
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


def _mandatory_verify_v9aa(harness, targets, translated, memory):
    selected, ranked = _route_v9aa(targets, translated, memory)
    mandatory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        mandatory.append(row)
        seen.add(str(row["id"]))
    # Every actual glossary mismatch must be inspected even when it lost a router
    # quota. This is the failure mode that the old top-24 verifier could miss.
    for row in ranked:
        if str(row["id"]) in seen:
            continue
        if row.get("glossary_hits"):
            mandatory.append(row)
            seen.add(str(row["id"]))
    extra = max(0, int(os.getenv("BOOKAI_V9AA_VERIFY_EXTRA") or "8"))
    for row in ranked:
        if extra <= 0:
            break
        if str(row["id"]) in seen:
            continue
        mandatory.append(row)
        seen.add(str(row["id"]))
        extra -= 1

    by_id = {s.id: s for s in targets}
    pos = {s.id: i for i, s in enumerate(targets)}
    route_by_id = {str(row["id"]): row for row in mandatory}
    items = []
    for row in mandatory:
        sid = str(row["id"])
        i = pos[sid]
        items.append({
            "id": sid,
            "source": by_id[sid].text,
            "current_ru": str(translated.get(sid) or ""),
            "before_en": [x.text for x in targets[max(0, i - 2):i]],
            "after_en": [x.text for x in targets[i + 1:i + 3]],
            "risk_code": row.get("code"),
            "risk_reason": row.get("reason"),
            "source_only_glossary_hints": row.get("glossary_hits") or [],
        })

    system = """You are a mandatory FINAL semantic verifier for EN→RU literary translation. You MUST explicitly check every
item supplied. Source and neighboring English context are authoritative; current_ru is the candidate under review.
Source-only glossary hints are hypotheses, NOT truth, and may contain the same wrong dictionary sense as current_ru.
For each item verify proposition-level meaning and especially its stated risk_code: legal function; exact paired
referents; explicit learning/acquisition; earn-a-living semantics; comparison direction; polarity/idiom; specialist
lexical denotation; object-part sense; proper-place/common-noun sense; numbers and actor/action/object relations.
Natural paraphrase is fine. Return ok=false for any real publication-level meaning defect, even if the Russian is fluent
or matches a glossary hint. Return EXACTLY one row for every supplied id and no extra ids. ONLY JSON
{"checks":[{"id":"...","ok":true,"code":"...","confidence":0.0,"reason":"..."}]}.
"""

    confirmed: list[dict[str, Any]] = []
    batch_size = max(4, int(os.getenv("BOOKAI_V9AA_VERIFY_BATCH") or "10"))
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        expected = {str(x["id"]) for x in batch}
        checks = None
        for attempt in range(2):
            try:
                obj = v8._complete_json(harness.gate, system, {"items": batch})
                raw = obj.get("checks") or []
                if not isinstance(raw, list):
                    raise ValueError("checks is not a list")
                parsed = {str(row.get("id") or ""): row for row in raw if isinstance(row, dict)}
                if set(parsed) != expected:
                    raise ValueError("verifier id contract violation")
                checks = parsed
                break
            except Exception as exc:
                print(f"[v9aa-final-verify] batch={start // batch_size + 1} attempt={attempt + 1} error={type(exc).__name__}", flush=True)
        if checks is None:
            # Contract failure should not silently bless mandatory high-risk items.
            checks = {sid: {"id": sid, "ok": False, "code": "verifier_contract", "confidence": 0.72,
                            "reason": "mandatory verifier failed its exact-id contract; conservative retranslation"}
                      for sid in expected}
        for sid, row in checks.items():
            if type(row.get("ok")) is not bool or row.get("ok") is True:
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            risk_code = str(route_by_id[sid].get("code") or "semantic")
            threshold = 0.58 if risk_code in {"legal_role", "learning", "paired_referent", "livelihood", "word_sense", "specialized_lexicon"} else 0.68
            if confidence < threshold:
                continue
            confirmed.append({
                "id": sid,
                "severity": "major",
                "confidence": min(0.99, max(threshold, confidence)),
                "code": str(row.get("code") or risk_code),
                "source_span": str(by_id[sid].text or "")[:240],
                "target_span": str(translated.get(sid) or "")[:240],
                "reason": v9._norm_text(row.get("reason") or "mandatory final semantic verifier found a defect")[:520],
                "repairability": "local",
            })

    confirmed_ids = []
    for row in confirmed:
        sid = str(row["id"])
        if sid not in confirmed_ids:
            confirmed_ids.append(sid)
    routes = [route_by_id[sid] for sid in confirmed_ids if sid in route_by_id]
    routes.sort(key=lambda r: (-int(r["priority"]), int(r["index"])))
    global _V9AA_STATS
    _V9AA_STATS.update({
        "mandatory_router_selected": len(selected),
        "mandatory_glossary_and_extra_verified": len(mandatory),
        "mandatory_final_confirmed": len(confirmed),
        "mandatory_final_confirmed_ids": confirmed_ids,
    })
    return confirmed, routes, len(mandatory)


def _quality_v9aa(harness, targets, translated, memory):
    stats = dict(_BASE_QUALITY_Z(harness, targets, translated, memory) or {})
    stats.update(_V9AA_STATS)
    stats["quality_mode_v9aa"] = "bookwide-name-canon+advisory-polysemy+reserved-router-quotas+lexical-sense-analysis+mandatory-final-verifier"
    print("[bookai-v9aa] " + json.dumps({**_V9AA_STATS, "quality_mode_v9aa": stats["quality_mode_v9aa"]}, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9aa() -> None:
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9aa-mandatory-semantic-verification",
        "gold_reference_available_to_pipeline": False,
        "entity_policy": "single whole-book source-only canonical registry cached across chapter processes",
        "glossary_policy": "proper names hard; legal/polysemous lexical senses advisory during semantic repair",
        "router_policy": "reserved semantic quotas including four specialist-lexicon slots",
        "word_sense_policy": "source-context lexical denotation analysis before repair",
        "final_verifier": "mandatory all first-wave routes + every glossary mismatch + bounded extra risks",
    }
    data["v9aa_stats"] = dict(_V9AA_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    # Patch v9z at module-global lookup points so its source-only runtime safeguards
    # remain intact while v9aa owns the new policy layers.
    v9z._candidate_bad_source_only = _candidate_bad_source_only_v9aa
    v9z._learn_name_canon = _learn_name_canon_v9aa
    v9z._route_v9z = _route_v9aa
    v9z._repair_direct_v9z = _repair_direct_v9aa
    v9z._blind_risk_verify = _mandatory_verify_v9aa
    v9z._quality_v9z = _quality_v9aa
    try:
        v9z.main()
    finally:
        _annotate_v9aa()


if __name__ == "__main__":
    main()
