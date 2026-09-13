from __future__ import annotations

import json
import os
import re
from typing import Any

import chapter_reference_translation_v9ap_precision_giga as v9ap
from bookai.json_salvage import parse_json_with_item_salvage


_BASE_ENHANCED = v9ap._enhanced_simple_failures
_BASE_RATE_JSON = v9ap._rate_resilient_giga_json
_BASE_OBLIGATIONS = v9ap._issue_obligations
_V9AB = v9ap.v9ao.v9am.v9al.v9ah.v9af.v9ab

_STATS: dict[str, Any] = {
    "json_partial_salvages": 0,
    "json_full_fail_format_retries": 0,
    "beat_coverage_flags": 0,
    "invention_flags": 0,
    "extra_technical_flags": 0,
}


def _parse_json_recovering(text: str) -> dict[str, Any]:
    obj = parse_json_with_item_salvage(text)
    if obj.get("_salvaged_partial"):
        _STATS["json_partial_salvages"] = int(_STATS.get("json_partial_salvages") or 0) + 1
        print(
            "[v9aq-json-salvage] "
            + json.dumps({"items": len(obj.get("items") or [])}, ensure_ascii=False),
            flush=True,
        )
    return obj


def _robust_giga_json(system: str, payload: Any, *, max_tokens: int = 5000):
    try:
        return _BASE_RATE_JSON(system, payload, max_tokens=max_tokens)
    except json.JSONDecodeError:
        _STATS["json_full_fail_format_retries"] = int(_STATS.get("json_full_fail_format_retries") or 0) + 1
        retry_system = system + """

FORMAT RECOVERY RETRY: Your previous response was not parseable JSON. Return ONLY one compact JSON object.
No Markdown fences, no commentary, no trailing commas. Keep every corrected_ru concise enough to finish the full array.
Schema exactly: {\"items\":[{\"id\":\"...\",\"corrected_ru\":\"...\"}]}.
"""
        print("[v9aq-json-retry] reason=unrecoverable_json same_batch=true", flush=True)
        return _BASE_RATE_JSON(retry_system, payload, max_tokens=max_tokens)


def _beat_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:['\"»”])?(?:\s|$)", str(text or "")))


def _beat_coverage_failure(source: str, target: str) -> bool:
    src = str(source or "").strip()
    ru = str(target or "").strip()
    if len(src) < 300:
        return False
    src_beats = _beat_count(src)
    ru_beats = _beat_count(ru)
    if src_beats < 5 or src_beats - ru_beats < 2:
        return False
    ratio = len(ru) / max(1, len(src))
    return ratio < 0.78


def _invention_failures(source: str, target: str) -> list[str]:
    src = str(source or "")
    ru = str(target or "").casefold().replace("ё", "е")
    out: list[str] = []

    # Anachronistic method invented from a plain shear-cut operation.
    if re.search(r"\bshear[- ]cut\b", src, re.I) and "лазер" in ru:
        out.append("invented_laser")

    # Vague source duration must not become an invented exact year/month/week.
    if re.search(r"\bnot\s+long\s+after\b", src, re.I):
        exact_duration = re.search(
            r"(?:не\s+прошл\w*\s+и|через|спустя)\s+(?:\w+\s+){0,3}(?:год|года|лет|месяц|месяца|месяцев|недел|дн)\w*",
            ru,
        )
        if exact_duration:
            out.append("invented_exact_duration")

    # English idiom `the other day` means recently, not a part of the clock-day.
    if re.search(r"\bthe\s+other\s+day\b", src, re.I) and re.search(r"середин\w*\s+дн|после\s+полудн", ru):
        out.append("other_day_idiom")

    # `since he was twenty-one` is an age boundary, not a duration of 21 years.
    if re.search(r"\bsince\s+(?:he|she)\s+was\s+twenty[- ]one\b", src, re.I) and re.search(
        r"(?:за|последн\w*)[^.!?]{0,30}двадцат\w*\s+один\w*\s+год", ru
    ):
        out.append("since_age_relation")
    return out


def _extra_technical_failures(source: str, target: str) -> list[str]:
    src = str(source or "")
    ru = str(target or "").casefold().replace("ё", "е")
    out: list[str] = []
    if re.search(r"\bscorpion\s+locks?\b", src, re.I) and "скорпион" not in ru:
        out.append("scorpion_lock")
    if re.search(r"\bshear[- ]cut\s+plate\b", src, re.I) and not re.search(r"ножниц|срез|резан", ru):
        out.append("shear_cut_plate")
    if re.search(r"\biron\s+stock\b", src, re.I) and ("руд" in ru or not re.search(r"заготов|полос|прут|желез", ru)):
        out.append("iron_stock")
    if re.search(r"\bold\s+lists\b", src, re.I) and "списк" in ru:
        out.append("tournament_lists")
    return out


def _enhanced_v9aq(segment, target: str) -> list[str]:
    source = str(segment.text or "")
    codes = list(_BASE_ENHANCED(segment, target))

    if _beat_coverage_failure(source, target) and "long_omission" not in codes:
        codes.append("long_omission")
        codes.append("beat_coverage")
        _STATS["beat_coverage_flags"] = int(_STATS.get("beat_coverage_flags") or 0) + 1

    invention = _invention_failures(source, target)
    if invention:
        codes.append("invented_specific")
        codes.extend("invention:" + item for item in invention)
        _STATS["invention_flags"] = int(_STATS.get("invention_flags") or 0) + len(invention)

    tech = _extra_technical_failures(source, target)
    if tech:
        if "technical_term" not in codes:
            codes.append("technical_term")
        codes.extend("term:" + item for item in tech)
        _STATS["extra_technical_flags"] = int(_STATS.get("extra_technical_flags") or 0) + len(tech)

    return list(dict.fromkeys(codes))


def _obligations_v9aq(row: dict[str, Any]) -> list[str]:
    out = list(_BASE_OBLIGATIONS(row))
    for raw in row.get("codes", []):
        code = str(raw)
        if code == "beat_coverage":
            out.append("the Russian must preserve every source sentence/thought beat; do not drop the opening perspective-change thoughts")
        elif code == "invention:invented_laser":
            out.append("source says shear-cut; NEVER invent a laser; render mechanical cutting with shears")
        elif code == "invention:invented_exact_duration":
            out.append("source says only 'not long after'; do not invent an exact year/month/week duration")
        elif code == "invention:other_day_idiom":
            out.append("'the other day' means recently / на днях, NOT середина дня or после полудня")
        elif code == "invention:since_age_relation":
            out.append("'since he was twenty-one' means с двадцати одного года / с тех пор как ему был 21, not 'за 21 год'")
        elif code == "term:scorpion_lock":
            out.append("scorpion lock is a lock/mechanism of the scorpion engine; preserve скорпион, never скальный")
        elif code == "term:shear_cut_plate":
            out.append("shear-cut plate = plate cut with shears / ножницами; no laser")
        elif code == "term:iron_stock":
            out.append("iron stock in workshop context = iron stock/bar/workpiece (железная заготовка/полоса/пруток), not ore")
        elif code == "term:tournament_lists":
            out.append("old lists in martial/tournament context = старое ристалище/турнирное ограждение, not paper lists")
    return list(dict.fromkeys(out))


def _annotate_report() -> None:
    v3 = v9ap.v9ao.v9am.v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    arch = dict(data.get("architecture") or {})
    arch.update({
        "experiment": "v9aq-gigachat-json-salvage+beat-coverage",
        "giga_structured_recovery": "normal parse -> partial-items salvage -> one compact same-batch format retry",
        "long_omission_gate": "length + source/target sentence-beat coverage for long multi-beat segments",
        "source_grounded_invention_guards": ["shear-cut!=laser", "not-long-after!=exact-duration", "the-other-day idiom", "since-age relation"],
        "extra_technical_guards": ["scorpion locks", "shear-cut plate", "iron stock", "tournament lists"],
        "deepseek_policy": "unchanged v9ap: one semantic batch target <=10; simple repairs remain GigaChat-first",
        "ab_benchmark": "retired; single real chapter audit only",
    })
    data["architecture"] = arch
    data["v9aq"] = dict(_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_parse = _V9AB._parse_json
    old_rate = v9ap._rate_resilient_giga_json
    old_enhanced = v9ap._enhanced_simple_failures
    old_obligations = v9ap._issue_obligations

    _V9AB._parse_json = _parse_json_recovering
    v9ap._rate_resilient_giga_json = _robust_giga_json
    v9ap._enhanced_simple_failures = _enhanced_v9aq
    v9ap._issue_obligations = _obligations_v9aq
    try:
        v9ap.main()
    finally:
        _V9AB._parse_json = old_parse
        v9ap._rate_resilient_giga_json = old_rate
        v9ap._enhanced_simple_failures = old_enhanced
        v9ap._issue_obligations = old_obligations
        _annotate_report()


if __name__ == "__main__":
    main()
