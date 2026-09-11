from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v6 as v6
import chapter_reference_translation_v7 as v7
import hybrid_reference_translation as hybrid
from bookai.models import BookMemory, Segment
from bookai.quality_v3 import enhanced_candidate_issues


# quality-v8: keep v6/v7's source-only book intelligence + unified QE + self-TM,
# but fix the four failure modes demonstrated by the blind Chapter Twenty test:
#   1) detected quantitative corruption must receive an explicit repair candidate;
#   2) catastrophic omissions prefer complete rescue candidates over broken originals;
#   3) named-entity consistency is inflection-aware and typo-tolerant;
#   4) Russian dialogue typography is normalized deterministically after model edits.
# The benchmark workflow runs Chapters One -> Two -> Three against one shared cache,
# so Chapter Two/Three actually retrieve self-approved translations from prior chapters.

_V8_STATS: dict = {}
_CYR_WORD_RE = re.compile(r"\b[А-ЯЁ][А-Яа-яЁё-]{2,}\b")
_ASCII_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")


def _provider(harness):
    translator = getattr(harness, "translator", None)
    provider = getattr(translator, "provider", None)
    return provider or getattr(harness, "editor", None)


def _complete_json(provider, system: str, payload: dict) -> dict:
    from bookai.llm import extract_json
    raw = provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0)
    obj = extract_json(raw)
    return obj if isinstance(obj, dict) else {}


def _edit_distance(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _inflected_last_forms(word: str) -> set[str]:
    value = word.strip()
    if not value:
        return set()
    low = value.casefold()
    forms = {low}
    if not re.search(r"[а-яё]", low):
        return forms
    if low.endswith("ий"):
        stem = low[:-2]
        forms |= {stem + ending for ending in ("ия", "ию", "ием", "ии")}
    elif low.endswith(("ый", "ой")):
        stem = low[:-2]
        forms |= {stem + ending for ending in ("ого", "ому", "ым", "ом")}
    elif low.endswith("й"):
        stem = low[:-1]
        forms |= {stem + ending for ending in ("я", "ю", "ем", "е")}
    elif low.endswith("а"):
        stem = low[:-1]
        forms |= {stem + ending for ending in ("ы", "и", "е", "у", "ой", "ою")}
    elif low.endswith("я"):
        stem = low[:-1]
        forms |= {stem + ending for ending in ("и", "е", "ю", "ей", "ею")}
    elif low.endswith("ь"):
        stem = low[:-1]
        forms |= {stem + ending for ending in ("я", "ю", "ем", "е", "и", "ью")}
    elif re.search(r"[бвгджзклмнпрстфхцчшщ]$", low):
        forms |= {low + ending for ending in ("а", "у", "ом", "е", "ы", "и")}
    return forms


def _phrase_forms(target: str) -> set[str]:
    words = target.split()
    if not words:
        return set()
    prefix = " ".join(words[:-1]).casefold()
    return {(prefix + " " + form).strip() for form in _inflected_last_forms(words[-1])}


def _entity_map(memory: BookMemory) -> dict[str, str]:
    out: dict[str, str] = {}
    for source, desc in memory.characters.items():
        match = re.search(r"\bru=([^;]+)", str(desc), flags=re.I)
        if match:
            target = match.group(1).strip()
            if source and target:
                out[str(source).strip()] = target
    for source, target in memory.glossary.items():
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target or source in out:
            continue
        named_source = len(source.split()) >= 2 or source[:1].isupper()
        named_target = target[:1].isupper() or len(target.split()) >= 2
        demonym_like = bool(re.search(r"(?:ine|ian|ese|ish)$", source, flags=re.I)) and target[:1].islower()
        if named_source and named_target and not demonym_like:
            out[source] = target
    return out


def _source_mentions(source_text: str, name: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", source_text, flags=re.I))


def _entity_present(candidate: str, target: str) -> bool:
    low = candidate.casefold()
    return any(form in low for form in _phrase_forms(target))


def _fix_near_entity_typos(segment: Segment, candidate: str, memory: BookMemory) -> tuple[str, int]:
    value = str(candidate or "")
    fixes = 0
    for source, target in _entity_map(memory).items():
        if not _source_mentions(segment.text, source) or _entity_present(value, target):
            continue
        words = target.split()
        if not words:
            continue
        canonical_last = words[-1]
        allowed = _inflected_last_forms(canonical_last)
        best = None
        for match in _CYR_WORD_RE.finditer(value):
            token = match.group(0)
            if min(len(token), len(canonical_last)) < 5:
                continue
            options = sorted(((_edit_distance(token, form), form) for form in allowed), key=lambda row: row[0])
            if options and options[0][0] <= 1:
                score, form = options[0]
                if best is None or score < best[0]:
                    best = (score, match.start(), match.end(), form)
        if best is None:
            continue
        _, start, end, replacement = best
        original = value[start:end]
        if original[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        value = value[:start] + replacement + value[end:]
        fixes += 1
    return value, fixes


def _v8_source_only_hard(targets, translations, memory) -> tuple[set[str], dict[str, str]]:
    hard: set[str] = set()
    reasons: dict[str, list[str]] = {}
    entities = _entity_map(memory)
    for segment in targets:
        candidate = str(translations.get(segment.id) or "")
        for issue in enhanced_candidate_issues(segment, candidate, memory, source_segments=targets):
            if issue.severity == "hard":
                hard.add(segment.id)
                reasons.setdefault(segment.id, []).append(f"{issue.code}: {issue.reason}")
        for source, target in entities.items():
            if _source_mentions(segment.text, source) and not _entity_present(candidate, target):
                hard.add(segment.id)
                reasons.setdefault(segment.id, []).append(
                    f"named entity missing/inconsistent: {source} -> {target} (inflection-aware)"
                )
    return hard, {sid: "; ".join(dict.fromkeys(rows)) for sid, rows in reasons.items()}


def _quantity_repair(harness, targets, segment: Segment, current: str, memory, reason: str) -> str:
    provider = _provider(harness)
    if provider is None:
        return ""
    position = {row.id: i for i, row in enumerate(targets)}
    idx = position.get(segment.id, 0)
    payload = {
        "source": segment.text,
        "current_ru": current,
        "detected_quantitative_error": reason,
        "context_before": [row.text for row in targets[max(0, idx - 2):idx]],
        "context_after": [row.text for row in targets[idx + 1:idx + 3]],
        "book_bible": {"glossary": memory.glossary, "characters": memory.characters, "style": asdict(memory.style)},
    }
    system = """You are an EN→RU literary fidelity editor. Repair ONLY the detected quantitative/measurement error while
preserving every other correct fact, clause, joke, relation and stylistic choice. Interpret the English quantity exactly;
never change 1.5x into 2x, a fraction into its denominator, or units into a different magnitude. Keep the paragraph complete
and natural Russian. Return ONLY JSON {\"translation\":\"...\"}."""
    try:
        return str(_complete_json(provider, system, payload).get("translation") or "").strip()
    except Exception as exc:
        print(f"[v8-quantity-repair] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _obligation_rescue_v8(harness, targets, segment: Segment, memory) -> str:
    provider = _provider(harness)
    if provider is None:
        return ""
    position = {row.id: i for i, row in enumerate(targets)}
    idx = position.get(segment.id, 0)
    payload = {
        "source": segment.text,
        "context_before": [row.text for row in targets[max(0, idx - 2):idx]],
        "context_after": [row.text for row in targets[idx + 1:idx + 3]],
        "book_bible": {
            "style": asdict(memory.style), "glossary": memory.glossary,
            "characters": memory.characters, "continuity": str(memory.rolling_summary or "")[:3500],
        },
    }
    system = """The previous EN→RU literary translation catastrophically omitted content. Reconstruct the paragraph from
SOURCE, not from the broken translation. Internally enumerate every atomic obligation: actors/actions/objects, every sentence
and question, numbers/measurements, negation/modality, comparisons, causes, qualifications, images, joke setup/payoff and
referents. Then write ONE complete natural Russian paragraph preserving all obligations and the source's dry literary voice.
Do not summarize. Return ONLY JSON {\"obligations\":[\"short item\",...],\"translation\":\"...\"}."""
    try:
        obj = _complete_json(provider, system, payload)
        translation = str(obj.get("translation") or "").strip()
        obligations = obj.get("obligations") or []
        print(f"[v8-obligation-rescue] id={segment.id} obligations={len(obligations) if isinstance(obligations, list) else 0}", flush=True)
        return translation
    except Exception as exc:
        print(f"[v8-obligation-rescue] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _raw_english_count(text: str) -> int:
    return len(_ASCII_WORD_RE.findall(text or ""))


def _candidate_vector(segment: Segment, candidate: str, memory, targets) -> tuple[int, int, int, int]:
    if not candidate.strip():
        return (1, 1, 999, 999)
    catastrophic = int(v7._catastrophic(segment, candidate))
    quantity = int(v7._quantity_risk(segment, candidate) is not None)
    hard = sum(issue.severity == "hard" for issue in enhanced_candidate_issues(segment, candidate, memory, source_segments=targets))
    return catastrophic, quantity, int(hard), _raw_english_count(candidate)


def _judge_candidates(harness, segment: Segment, candidates: dict[str, str], memory) -> str:
    valid = {name: text for name, text in candidates.items() if str(text).strip()}
    if len(valid) <= 1:
        return next(iter(valid), "")
    labels = list(valid)
    payload = {
        "source": segment.text,
        "book_bible": {"glossary": memory.glossary, "characters": memory.characters, "style": asdict(memory.style)},
        "candidates": {chr(65 + i): valid[name] for i, name in enumerate(labels)},
    }
    system = """Choose the best EN→RU literary translation candidate; do not rewrite. Fidelity dominates style. Reject
omissions, wrong actor/action/object relations, corrupted numbers/measurements, negation changes, invented facts, raw English
and broken Russian. Then prefer natural prose preserving voice and irony. Return ONLY JSON {\"choice\":\"A|B|C\",\"reason\":\"brief\"}."""
    try:
        obj = _complete_json(harness.gate, system, payload)
        choice = str(obj.get("choice") or "").strip().upper()
        index = ord(choice) - 65 if len(choice) == 1 else -1
        if 0 <= index < len(labels):
            return labels[index]
    except Exception as exc:
        print(f"[v8-candidate-judge] id={segment.id} error={type(exc).__name__}", flush=True)
    return labels[0]


def _select_candidate(harness, segment, candidates, memory, targets):
    valid = {name: text for name, text in candidates.items() if str(text).strip()}
    vectors = {name: _candidate_vector(segment, text, memory, targets) for name, text in valid.items()}
    if not valid:
        return "", "", vectors
    best_fatal = min((vec[0], vec[1]) for vec in vectors.values())
    pool = {name: valid[name] for name, vec in vectors.items() if (vec[0], vec[1]) == best_fatal}
    if len(pool) == 1:
        name = next(iter(pool))
        return name, pool[name], vectors
    judged = _judge_candidates(harness, segment, pool, memory)
    if judged in pool:
        return judged, pool[judged], vectors
    name = min(pool, key=lambda key: (vectors[key][2], vectors[key][3]))
    return name, pool[name], vectors


def _normalize_dialogue_v8(segment: Segment, text: str) -> tuple[str, int]:
    value = v6._normalize_typography(segment, text)
    before = value
    source_dialogue = bool(re.search(r"(^|[.!?]\s+)[\"'‘“]", segment.text))
    if source_dialogue and re.search(r"[А-Яа-яЁё]", value):
        value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r"^\s*['\"]\s*", "— ", value)
        value = re.sub(r"([,!?…\.])['\"]\s*(—|-)", r"\1 \2", value)
        value = re.sub(r"([.!?…])\s*['\"]\s*(?=[А-ЯЁ])", r"\1 — ", value)
        value = re.sub(r"['\"]\s*$", "", value)
        value = re.sub(r"['\"]([^'\"\n]{1,160})['\"]", r"«\1»", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
    return value, int(value != before)


def _v8_semantic_qe_repair(harness, targets, translated, memory) -> dict:
    global _V8_STATS
    base_stats = v6._semantic_qe_repair(harness, targets, translated, memory)
    by_id = {segment.id: segment for segment in targets}
    quantity: dict[str, str] = {}
    catastrophic: list[str] = []
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        reason = v7._quantity_risk(segment, current)
        if reason:
            quantity[segment.id] = reason
        if v7._catastrophic(segment, current):
            catastrophic.append(segment.id)

    candidate_ids: list[str] = []
    for sid in catastrophic + list(quantity) + sorted(v6._V6_REMAINING_HARD):
        if sid in by_id and sid not in candidate_ids:
            candidate_ids.append(sid)
    candidate_ids = candidate_ids[: max(4, int(os.getenv("BOOKAI_V8_COUNCIL_MAX") or "12"))]

    council_attempted = council_changed = quantity_repairs = rescue_attempted = rescue_accepted = 0
    for sid in candidate_ids:
        segment = by_id[sid]
        original = str(translated.get(sid) or "")
        fresh = v7._direct_candidate(harness, targets, segment, memory)
        candidates = {"original": original, "fresh": fresh}
        if sid in quantity:
            repaired = _quantity_repair(harness, targets, segment, original, memory, quantity[sid])
            if repaired:
                candidates["quantity_repair"] = repaired
                quantity_repairs += 1
        if sid in catastrophic:
            rescued = _obligation_rescue_v8(harness, targets, segment, memory)
            if rescued:
                candidates["rescue"] = rescued
                rescue_attempted += 1
        if len(candidates) > 3:
            candidates.pop("fresh", None)
        chosen_name, chosen, vectors = _select_candidate(harness, segment, candidates, memory, targets)
        council_attempted += 1
        if chosen and chosen.strip() != original.strip():
            translated[sid] = chosen
            council_changed += 1
            if sid in catastrophic and chosen_name == "rescue":
                rescue_accepted += 1
            print("[v8-council] " + json.dumps({"id": sid, "choice": chosen_name, "vectors": vectors}, ensure_ascii=False), flush=True)

    entity_fixes = dialogue_normalized = 0
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        fixed, count = _fix_near_entity_typos(segment, current, memory)
        entity_fixes += count
        fixed, changed = _normalize_dialogue_v8(segment, fixed)
        dialogue_normalized += changed
        translated[segment.id] = fixed

    hard_after, hard_reasons = _v8_source_only_hard(targets, translated, memory)
    v6._V6_REMAINING_HARD = set(hard_after)
    for sid, reason in hard_reasons.items():
        v6._V6_REASONS[sid] = reason

    _V8_STATS = {
        **dict(base_stats),
        "quantity_flagged": len(quantity),
        "quantity_repair_candidates": quantity_repairs,
        "catastrophic_detected": len(catastrophic),
        "catastrophic_ids": catastrophic,
        "rescue_attempted": rescue_attempted,
        "rescue_accepted": rescue_accepted,
        "council_attempted": council_attempted,
        "council_changed": council_changed,
        "entity_typo_fixes": entity_fixes,
        "dialogue_segments_normalized": dialogue_normalized,
        "remaining_hard_after_v8": len(hard_after),
    }
    hybrid.progress({"phase": "v8_quality_done", **_V8_STATS})
    return dict(_V8_STATS)


class GigaChatLightningV8Backend(v7.GigaChatLightningV7Backend):
    name = "gigachat-3-lightning-v8-source-only-self-tm"


_ORIGINAL_V6_HARD = v6._source_only_hard


def _configure_v8() -> None:
    v7._configure_v7()
    slug = v3.CHAPTER_SLUG
    v3.OUTPUT = Path(f"Devices_and_Desires_RU_{slug}_EVAL_V8.fb2")
    v3.CACHE = Path(os.getenv("BOOKAI_V8_SHARED_CACHE") or ".bookai-cache-first3-v8-shared")
    v3.PROGRESS = Path(f"chapter-v8-{slug}-progress.json")
    v3.PROBE = Path(f"chapter-v8-{slug}-probe.json")
    v3.ROUTING = Path(f"chapter-v8-{slug}-routing.json")
    v3.REPORT = Path(f"chapter-v8-{slug}.json")
    v3.SOURCE_TXT = Path(f"chapter-v8-{slug}-source.txt")
    v3.TRANSLATED_TXT = Path(f"chapter-v8-{slug}-translated.txt")
    v3.MAP_JSON = Path(f"chapter-v8-{slug}-translation-map.json")
    v3.GigaChatLightningV3Backend = GigaChatLightningV8Backend
    v3._semantic_short_repair = _v8_semantic_qe_repair
    v6._source_only_hard = _v8_source_only_hard


def _annotate_v8() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        **dict(report.get("architecture") or {}),
        "version": "quality-v8-source-only-sequential-self-tm",
        "quantitative_repair": True,
        "catastrophic_rescue_priority": True,
        "candidate_selection": "fatal-class filter then bilingual A/B/C judge",
        "entity_integrity": "inflection-aware canonical entities + edit-distance typo repair",
        "dialogue_formatter": "deterministic Russian typography",
        "gold_reference_available_to_pipeline": False,
        "shared_self_tm_cache": str(v3.CACHE),
    }
    report["v8_stats"] = dict(_V8_STATS)
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v8()
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v7._annotate_v7()
        _annotate_v8()
        v6._source_only_hard = _ORIGINAL_V6_HARD


if __name__ == "__main__":
    main()
