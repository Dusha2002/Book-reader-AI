from __future__ import annotations

import json
import re
from typing import Any

import chapter_reference_translation_v9aq_giga_salvage as v9aq


_V9AO = v9aq.v9ap.v9ao
_V9AP = v9aq.v9ap
_BASE_ENHANCED = v9aq._enhanced_v9aq
_BASE_OBLIGATIONS = v9aq._obligations_v9aq
_BASE_REPAIR = v9aq.v9ap._giga_repair_resilient

_STATS: dict[str, Any] = {
    "pre_pass_executions": 0,
    "pre_pass_skips": 0,
    "final_pass_executions": 0,
    "strict_repair_disabled": True,
    "fluency_rejections": [],
    "context_rejections": [],
    "medium_omission_flags": 0,
    "dozen_half_dozen_flags": 0,
    "contextual_flags": 0,
}
_PRE_DONE = False


def _medium_omission(source: str, target: str) -> bool:
    src = str(source or "").strip()
    ru = str(target or "").strip()
    if len(src) < 145:
        return False
    src_beats = v9aq._beat_count(src)
    if src_beats < 3:
        return False
    return len(ru) / max(1, len(src)) < 0.67


def _dozen_mistranslation(source: str, target: str) -> bool:
    src = str(source or "")
    ru = str(target or "").casefold().replace("ё", "е")
    if not re.search(r"\b(?:a|one)\s+dozen\b|\bdozen\b", src, re.I):
        return False
    if re.search(r"\bhalf[ -](?:a[ -])?dozen\b", src, re.I):
        return False
    return bool(re.search(r"\bполдюжин", ru))


def _enhanced_v9ar(segment, target: str) -> list[str]:
    source = str(segment.text or "")
    codes = list(_BASE_ENHANCED(segment, target))
    if _medium_omission(source, target) and "long_omission" not in codes:
        codes.extend(["long_omission", "medium_beat_coverage"])
        _STATS["medium_omission_flags"] = int(_STATS.get("medium_omission_flags") or 0) + 1
    if _dozen_mistranslation(source, target):
        if "numeric" not in codes:
            codes.append("numeric")
        codes.append("quantity:dozen=12_not_half_dozen")
        _STATS["dozen_half_dozen_flags"] = int(_STATS.get("dozen_half_dozen_flags") or 0) + 1
    return list(dict.fromkeys(codes))


def _character_canon(memory) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for src_name, raw in dict(getattr(memory, "characters", {}) or {}).items():
        desc = str(raw or "")
        gender_match = re.search(r"gender=(male|female)", desc, re.I)
        ru_match = re.search(r"ru=([^;]+)", desc, re.I)
        if not gender_match or not ru_match:
            continue
        out.append((str(src_name), ru_match.group(1).strip(), gender_match.group(1).casefold()))
    return out


def _context_codes_for_index(targets, translated, memory, index: int) -> list[str]:
    segment = targets[index]
    source = str(segment.text or "")
    target = str(translated.get(str(segment.id)) or "")
    low = target.casefold().replace("ё", "е")
    before = " ".join(str(x.text or "") for x in targets[max(0, index - 2):index])
    after = " ".join(str(x.text or "") for x in targets[index + 1:index + 3])
    window = f"{before} {source} {after}"
    codes: list[str] = []

    workshop = bool(re.search(r"\b(?:file|chalk|teeth|steel|handle|grease|workshop|bench|carded|treadle|plate)\b", window, re.I))
    if workshop:
        if re.search(r"\bfile\b", source, re.I) and re.search(r"\b(?:досье|файл)\b", low):
            codes.append("context:file_tool")
        if re.search(r"\bchalk\s+it\b", source, re.I) and "мел" not in low:
            codes.append("context:chalk_file")
        if re.search(r"\bcard(?:ed|ing)?\s+it\b", source, re.I) and not re.search(r"прочист|очист|щетк", low):
            codes.append("context:card_file")

    if re.search(r"\ba\s+day\s+and\s+a\s+half\b[^.!?]{0,120}\bwould\s+be\s+finished\b", source, re.I):
        if re.search(r"\bпрошл\w*\s+полтора\s+дн", low):
            codes.append("context:future_completion")

    for src_name, ru_name, gender in _character_canon(memory):
        if not re.search(rf"\b{re.escape(src_name)}\b", source, re.I):
            continue
        ru_norm = ru_name.casefold().replace("ё", "е")
        if ru_norm and ru_norm not in low:
            continue
        if gender == "male" and re.search(r"\b(?:сказала|говорила|ответила|спросила|заметила|подумала)\b", low):
            codes.append("context:male_character_feminine_verb")
        elif gender == "female" and re.search(r"\b(?:сказал|говорил|ответил|спросил|заметил|подумал)\b", low):
            codes.append("context:female_character_masculine_verb")
        if gender == "male" and re.search(r"\bкоролева\b", low) and re.search(r"\bking\b", source, re.I):
            codes.append("context:king_gender")
        if gender == "female" and re.search(r"\bкороль\b", low) and re.search(r"\bqueen\b", source, re.I):
            codes.append("context:queen_gender")
    return list(dict.fromkeys(codes))


def _merge_context_rows(targets, translated, memory, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): dict(row) for row in rows}
    for i, segment in enumerate(targets):
        codes = _context_codes_for_index(targets, translated, memory, i)
        if not codes:
            continue
        sid = str(segment.id)
        row = by_id.get(sid)
        if row is None:
            row = {
                "id": sid,
                "index": i,
                "codes": [],
                "source": str(segment.text or ""),
                "current_ru": str(translated.get(sid) or ""),
                "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
                "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
            }
            by_id[sid] = row
        row["codes"] = list(dict.fromkeys(list(row.get("codes") or []) + codes))
        _STATS["contextual_flags"] = int(_STATS.get("contextual_flags") or 0) + len(codes)
    return sorted(by_id.values(), key=lambda row: int(row.get("index") or 0))


def _obligations_v9ar(row: dict[str, Any]) -> list[str]:
    out = list(_BASE_OBLIGATIONS(row))
    for raw in row.get("codes", []):
        code = str(raw)
        if code == "medium_beat_coverage":
            out.append("restore every missing thought/sentence in this medium-length multi-sentence source")
        elif code == "quantity:dozen=12_not_half_dozen":
            out.append("a dozen = 12; never translate it as полдюжины (6)")
        elif code == "context:file_tool":
            out.append("workshop context: file is a physical напильник, never computer файл or досье")
        elif code == "context:chalk_file":
            out.append("workshop context: Chalk it means apply/rub chalk to the file; preserve мел")
        elif code == "context:card_file":
            out.append("workshop context: card the file means clean the file teeth with a brush/file card")
        elif code == "context:future_completion":
            out.append("A day and a half, and X would be finished is a future estimate: 'ещё полтора дня — и ... будет готово', not elapsed time")
        elif code == "context:male_character_feminine_verb":
            out.append("character canon says this character is male; use masculine Russian agreement")
        elif code == "context:female_character_masculine_verb":
            out.append("character canon says this character is female; use feminine Russian agreement")
        elif code == "context:king_gender":
            out.append("source says King and character is male: use король, never королева")
        elif code == "context:queen_gender":
            out.append("source says Queen and character is female: use королева, never король")
    out.append("return clean publication-ready Russian only; no translator notes, slash-separated alternatives, synonym lists, or explanatory parentheses")
    return list(dict.fromkeys(out))


def _bad_repair_fluency(text: str) -> bool:
    value = str(text or "").strip()
    low = value.casefold().replace("ё", "е")
    if not value:
        return True
    if re.search(r"\([^()]{0,70}/[^()]{0,70}\)", value):
        return True
    if re.search(r"\b(?:или|вариант|буквально|прим\.?\s*перев)\s*[:—-]", low):
        return True
    if re.search(r"\bпосле\s+а\s+за\b|\bа\s+за\s*\(", low):
        return True
    return len(re.findall(r"\b[A-Za-z]{3,}\b", value)) >= 2


def _repair_with_fluency(targets, translated, memory, issues, *, strict: bool = False):
    before = {str(segment.id): str(translated.get(str(segment.id)) or "") for segment in targets}
    changed, calls = _BASE_REPAIR(targets, translated, memory, issues, strict=False)
    index_by_id = {str(segment.id): i for i, segment in enumerate(targets)}
    accepted: list[str] = []
    for sid in changed:
        candidate = str(translated.get(sid) or "")
        bad_fluency = _bad_repair_fluency(candidate)
        contextual = _context_codes_for_index(targets, translated, memory, index_by_id[sid])
        if bad_fluency or contextual:
            translated[sid] = before.get(sid, "")
            key = "fluency_rejections" if bad_fluency else "context_rejections"
            _STATS[key] = sorted(set(_STATS.get(key) or []) | {sid})
            print(f"[v9ar-accept] id={sid} accepted=false fluency={str(bad_fluency).lower()} contextual={contextual}", flush=True)
        else:
            accepted.append(sid)
    return accepted, calls


def _single_pass(targets, translated, memory, *, stage: str) -> dict[str, Any]:
    global _PRE_DONE
    _V9AO._normalize_addresses(targets, translated)
    initial = _merge_context_rows(targets, translated, memory, _V9AO._detect_simple_failures(targets, translated))

    if stage == "pre" and _PRE_DONE:
        _STATS["pre_pass_skips"] = int(_STATS.get("pre_pass_skips") or 0) + 1
        residual = initial
        calls = 0
        changed: list[str] = []
        print("[v9ar-giga-pass] stage=pre skipped=true reason=already_processed", flush=True)
    else:
        if stage == "pre":
            _PRE_DONE = True
            _STATS["pre_pass_executions"] = int(_STATS.get("pre_pass_executions") or 0) + 1
        else:
            _STATS["final_pass_executions"] = int(_STATS.get("final_pass_executions") or 0) + 1
        changed, calls = _repair_with_fluency(targets, translated, memory, initial, strict=False)
        _V9AO._normalize_addresses(targets, translated)
        residual = _merge_context_rows(targets, translated, memory, _V9AO._detect_simple_failures(targets, translated))

    _V9AO._GIGA_STATS["gigachat_calls"] = int(_V9AO._GIGA_STATS.get("gigachat_calls") or 0) + calls
    _V9AO._GIGA_STATS[f"{stage}_detected"] = len(initial)
    _V9AO._GIGA_STATS[f"{stage}_repaired"] = len(set(changed))
    _V9AO._GIGA_STATS[f"{stage}_residual"] = len(residual)
    _V9AO._GIGA_STATS[f"{stage}_initial_ids"] = [row["id"] for row in initial]
    _V9AO._GIGA_STATS[f"{stage}_residual_ids"] = [row["id"] for row in residual]
    _V9AO._GIGA_STATS[f"{stage}_codes"] = {row["id"]: row["codes"] for row in initial}
    result = {
        "stage": stage,
        "detected": len(initial),
        "repaired": len(set(changed)),
        "gigachat_calls": calls,
        "residual": len(residual),
        "residual_ids": [row["id"] for row in residual],
    }
    print("[v9ar-giga-pass] " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _annotate_report() -> None:
    v3 = _V9AO.v9am.v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    arch = dict(data.get("architecture") or {})
    arch.update({
        "experiment": "v9ar-fast-single-pass-giga",
        "giga_simple_policy": "one pre repair pass + one final residual repair pass; no strict duplicate pass; repeated pre route is cached/skipped",
        "repair_fluency_gate": "reject translator-note/alternative/broken salvage residue before accepting Giga repair",
        "context_guards": ["workshop file/chalk/card", "character gender canon", "King/Queen agreement", "future completion estimate"],
        "medium_omission_gate": "145+ chars, >=3 source beats, target/source ratio <0.67",
        "deepseek_policy": "unchanged: one semantic batch target <=10; no simple publication repairs",
        "runtime_sla_seconds": [120, 180],
        "ab_benchmark": "disabled",
    })
    data["architecture"] = arch
    data["v9ar"] = dict(_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    global _PRE_DONE
    _PRE_DONE = False
    old_simple_pass = _V9AO._giga_simple_pass
    old_v9aq_enhanced = v9aq._enhanced_v9aq
    old_v9aq_obligations = v9aq._obligations_v9aq
    old_repair = _V9AP._giga_repair_resilient

    _V9AO._giga_simple_pass = _single_pass
    v9aq._enhanced_v9aq = _enhanced_v9ar
    v9aq._obligations_v9aq = _obligations_v9ar
    _V9AP._giga_repair_resilient = _repair_with_fluency
    try:
        v9aq.main()
    finally:
        _V9AO._giga_simple_pass = old_simple_pass
        v9aq._enhanced_v9aq = old_v9aq_enhanced
        v9aq._obligations_v9aq = old_v9aq_obligations
        _V9AP._giga_repair_resilient = old_repair
        _annotate_report()


if __name__ == "__main__":
    main()
