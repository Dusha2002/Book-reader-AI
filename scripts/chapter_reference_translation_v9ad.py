from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chapter_reference_translation_v9ac as v9ac

v9ab = v9ac.v9ab
v9aa = v9ab.v9aa
v9z = v9aa.v9z
v9x = v9z.v9x
v9v, v9u, v9t = v9z.v9v, v9z.v9u, v9z.v9t
v9s, v9, v8, v6, v3 = v9z.v9s, v9z.v9, v9z.v8, v9z.v6, v9z.v3

_ORIGINAL_ENSURE_INTELLIGENCE = v9ab._ensure_giga_book_intelligence

# v9ad keeps the successful v9ac cost architecture but removes the parts that
# empirically provided no value. Across the 3-chapter v9ac benchmark the broad
# GigaChat screen produced zero suspects, so it is disabled. DeepSeek remains at
# only a few batched calls per chapter, with a stronger risk-specific contract.
# No Russian reference/gold translation is available at runtime.


def _name_canon_path() -> Path:
    root = Path(os.getenv("BOOKAI_V8_SHARED_CACHE") or ".bookai-cache-v9ad")
    root.mkdir(parents=True, exist_ok=True)
    return root / "giga-name-canon-v9ad.json"


def _learn_giga_name_canon(memory) -> dict[str, str]:
    path = _name_canon_path()
    try:
        if path.exists():
            obj = json.loads(path.read_text("utf-8"))
            canon = {str(k): v9._norm_text(v) for k, v in dict(obj.get("canonicals") or {}).items() if str(k) and v9._norm_text(v)}
            if canon:
                return canon
    except Exception:
        pass

    candidates = v9aa._whole_book_name_items()
    if not candidates:
        return {}
    limit = max(30, int(os.getenv("BOOKAI_V9AD_NAME_MAX") or "60"))
    candidates = candidates[:limit]
    batch_size = max(12, int(os.getenv("BOOKAI_V9AD_NAME_BATCH") or "20"))
    system = """Create a SOURCE-ONLY book-wide Russian spelling canon for fictional proper names.
You receive English candidate names, frequency, and a few English contexts. No Russian reference exists.
Return every candidate that is genuinely a person/place/institution proper name; omit ordinary English words/titles.
For invented names prefer SPELLING-PRESERVING transliteration over guessed pronunciation. Preserve visible source
information: do not collapse a written final -ea to one vowel; preserve an internal/final written vowel sequence;
preserve written -tz- as a corresponding тз cluster rather than silently deleting a consonant; do not add й/я unless
source spelling supports it. Use normal Russian orthography (including soft sign when naturally needed) but never
simplify away source letters just because a pronunciation is plausible. Keep ONE stable dictionary-form spelling for
the whole book. ONLY JSON {"canonicals":[{"source":"...","ru":"...","confidence":0.0}]}.
"""
    canon: dict[str, str] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        try:
            obj = v9ab._giga_json(system, {"candidates": batch}, max_tokens=3400)
            rows = obj.get("canonicals") or []
        except Exception as exc:
            print(f"[v9ad-name-canon] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
            rows = []
        allowed = {row["source"] for row in batch}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source") or "").strip()
            ru = v9._norm_text(row.get("ru") or "")
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if src not in allowed or confidence < 0.72 or not ru:
                continue
            if not any("А" <= c <= "я" or c in "Ёё" for c in ru):
                continue
            canon[src] = ru
    if canon:
        path.write_text(json.dumps({"canonicals": canon}, ensure_ascii=False, indent=2), "utf-8")
    print(f"[v9ad-name-canon] entries={len(canon)} source_only=true", flush=True)
    return canon


def _ensure_intelligence_v9ad(memory) -> dict[str, Any]:
    stats = dict(_ORIGINAL_ENSURE_INTELLIGENCE(memory) or {})
    canon = _learn_giga_name_canon(memory)
    for src, ru in canon.items():
        memory.glossary[src] = ru
        old = str(memory.characters.get(src) or "")
        if old:
            # Replace or insert a deterministic canonical spelling while retaining
            # source-derived gender/role/voice notes from the cheap book bible.
            import re
            if re.search(r"\bru=[^;]+", old):
                memory.characters[src] = re.sub(r"\bru=[^;]+", f"ru={ru}", old, count=1)
            else:
                memory.characters[src] = f"ru={ru};" + old
        else:
            memory.characters[src] = f"ru={ru};role=proper_name"
    stats["bookwide_name_canon"] = len(canon)
    return stats


def _screen_disabled(targets, translated, memory):
    print(f"[v9ad-giga-screen] disabled=true reason=v9ac_zero_suspects segments={len(targets)}", flush=True)
    return []


def _canonicalize_candidate(segment, candidate: str, memory) -> str:
    value = v9._norm_text(candidate)
    if not value:
        return ""
    # Canonicalize actual source-derived named entities AFTER semantic editing.
    # This avoids the old hard-lock failure where ordinary glossary words blocked
    # a correct semantic repair, while still enforcing stable book-wide names.
    try:
        value, _ = v9s._entity_fix_v9s(segment, value, memory)
    except Exception:
        pass
    return value


def _deep_repair_v9ad(harness, targets, translated, memory, routes):
    if not routes:
        return [], 0
    batch_size = max(3, int(os.getenv("BOOKAI_V9AB_DEEP_BATCH") or "7"))
    system = """You are the ONLY expensive independent semantic specialist in a cost-sensitive EN→RU literary pipeline.
For EVERY item, compare English SOURCE and neighboring English context against current_ru. Correct ONLY real meaning
or publication-breaking defects; otherwise keep current_ru. Fidelity is absolute.

Risk-specific rules:
- legal_role: FIRST infer which side/function the speaker actually performs from the surrounding argument. If the
  speaker is arguing that the prisoner/offence is serious and attacking the prisoner's defence, the Russian role noun
  must denote the accusing/prosecuting function; never preserve a dictionary gloss merely because English says
  advocate/counsel. Conversely, do not turn a genuine defender into a prosecutor.
- comparison: preserve who outnumbers whom and every explicit number. "outnumbered five to one" means five opponents
  against one subject, not a generic numerical advantage.
- paired_referent: reconstruct the exact antecedent SET from previous questions/replies. If two semantic slots were
  asked about (for example spouse? children?) and the answer supplies one member of one slot, a later plural/either
  expression may refer to the spouse plus that child; never choose a Russian pronoun that silently turns them into two
  same-category people.
- learning: preserve the explicit fact of learning/acquiring the mental habit, not merely getting used to it.
- livelihood: preserve the need to earn/make a living and need for work.
- word_sense/specialized_lexicon: infer the concrete denotation from syntax and scene before translating. Motion "to"
  a named/circumscribed Wood strongly indicates a woodland/place rather than one tree or material. For target practice,
  preserve the exact progression of physical targets (fruit/species/ring/needle part); the eye of a needle is its hole/
  eye, never an anatomical eye. Keep exact armour/tool/plant/object type rather than a generic substitute.
- ordinal/polarity/request_pragmatics: preserve logical relation and pragmatic force, but use natural Russian.

Glossary hints are advisory and may be wrong. Proper names should remain stable, but grammar may inflect them.
Return exactly one row for every input id. If change=true, corrected_ru must be a COMPLETE translation of that source
segment. ONLY JSON {"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}.
"""
    changed = []
    calls = 0
    by_id = {s.id: s for s in targets}
    for start in range(0, len(routes), batch_size):
        batch_routes = routes[start:start + batch_size]
        items = [v9ab._payload_item(targets, translated, memory, row) for row in batch_routes]
        expected = {x["id"] for x in items}
        try:
            obj = v8._complete_json(harness.gate, system, {"items": items})
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ad-deep-repair] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
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
            strong = {"legal_role", "comparison", "paired_referent", "learning", "livelihood", "word_sense", "specialized_lexicon", "ordinal"}
            threshold = 0.62 if route.get("code") in strong else 0.72
            candidate = _canonicalize_candidate(by_id[sid], row.get("corrected_ru") or "", memory)
            if confidence < threshold or not candidate:
                continue
            current = str(translated.get(sid) or "")
            if v9._fatal_count(by_id[sid], candidate, memory) > v9._fatal_count(by_id[sid], current, memory):
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _deep_verify_v9ad(harness, targets, translated, memory, routes):
    if not routes:
        return [], 0, 0
    cap = max(8, int(os.getenv("BOOKAI_V9AB_FINAL_VERIFY_MAX") or "20"))
    routes = routes[:cap]
    batch_size = max(5, int(os.getenv("BOOKAI_V9AB_VERIFY_BATCH") or "10"))
    system = """FINAL independent EN→RU semantic verification. Check each item against source and neighboring English.
Ignore stylistic taste. Re-evaluate its risk_code from scratch, with special attention to legal FUNCTION, comparison
orientation/numbers, paired antecedents, explicit learning, livelihood, named-place vs common-noun word sense, and
precise specialist object/plant/tool/armour denotation. Glossary hints are not authoritative. If wrong, repair the
complete segment; if semantically correct, ok=true. Return exactly one row per id. ONLY JSON
{"checks":[{"id":"...","ok":true,"corrected_ru":"","confidence":0.0,"reason":"..."}]}.
"""
    changed, confirmed, calls = [], 0, 0
    by_id = {s.id: s for s in targets}
    for start in range(0, len(routes), batch_size):
        batch_routes = routes[start:start + batch_size]
        items = [v9ab._payload_item(targets, translated, memory, row) for row in batch_routes]
        expected = {x["id"] for x in items}
        try:
            obj = v8._complete_json(harness.gate, system, {"items": items})
            raw = obj.get("checks") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ad-final-verify] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
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
            if confidence < 0.66:
                continue
            confirmed += 1
            candidate = _canonicalize_candidate(by_id[sid], row.get("corrected_ru") or "", memory)
            if not candidate:
                continue
            current = str(translated.get(sid) or "")
            if v9._fatal_count(by_id[sid], candidate, memory) > v9._fatal_count(by_id[sid], current, memory):
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, confirmed, calls


def main() -> None:
    # v9ac handles robust Giga JSON + recursion-safe routing. v9ad only replaces
    # empirically wasteful screening and semantic lock/repair behavior.
    v9ab._ensure_giga_book_intelligence = _ensure_intelligence_v9ad
    v9ab._screen_batches = _screen_disabled
    v9ab._deepseek_batched_repair = _deep_repair_v9ad
    v9ab._deepseek_final_verify = _deep_verify_v9ad
    v9ac.main()


if __name__ == "__main__":
    main()
