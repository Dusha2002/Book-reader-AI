from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import chapter_reference_translation_v9ao_giga_first as v9ao
from bookai.gigachat_rate_limit import _is_rate_limit_error


_BASE_SIMPLE = v9ao._simple_failures_for_pair
_BASE_GIGA_JSON = v9ao.v9am.v9al.v9ah.v9af.v9ab._giga_json
_BASE_SEMANTIC_ROUTE = v9ao._BASE_SEMANTIC_ROUTE
_STATS: dict[str, Any] = {
    "giga_json_rate_retries": 0,
    "giga_json_rate_exhausted": 0,
    "cheap_changed_ids": [],
    "deepseek_routes_before_filter": 0,
    "deepseek_routes_after_filter": 0,
    "deepseek_routes_dropped_after_cheap_repair": [],
}

_RU_NUM_WORDS = {
    1: r"(?:один|одна|одно|одну|одного|одной|одним|одном)",
    2: r"(?:два|две|двух|двум|двумя)",
    3: r"(?:три|тр[её]х|тр[её]м|тремя)",
    4: r"(?:четыре|четыр[её]х|четыр[её]м|четырьмя)",
    5: r"(?:пять|пяти|пятью)",
    6: r"(?:шесть|шести|шестью)",
    7: r"(?:семь|семи|семью)",
    12: r"(?:двенадцать|двенадцати)",
    40: r"(?:сорок|сорока)",
}

_TECH_RULES: list[tuple[str, re.Pattern[str], tuple[str, ...]]] = [
    ("file_tool", re.compile(r"\b(?:three[- ]square\s+)?file\b", re.I), ("напильник", "напильн")),
    ("cuisses", re.compile(r"\bcuisses?\b", re.I), ("набедрен", "бедренн")),
    ("gorget", re.compile(r"\bgorget\b", re.I), ("горжет", "шейн")),
    ("tinplate", re.compile(r"\btinplate\b", re.I), ("лужен", "оловян")),
    ("treadle_saw", re.compile(r"\btreadle\s+saw\b", re.I), ("ножн", "педал")),
    ("card_file", re.compile(r"\bcard(?:ed|ing)?\b[^.!?]{0,50}\bfile\b|\bfile\b[^.!?]{0,50}\bcard(?:ed|ing)?\b", re.I), ("прочист", "очист", "щетк")),
]


def _has_ru(text: str, stems: tuple[str, ...]) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    return any(stem.replace("ё", "е") in low for stem in stems)


def _technical_obligations(source: str, target: str) -> list[str]:
    rows: list[str] = []
    for code, pattern, stems in _TECH_RULES:
        if pattern.search(source) and not _has_ru(target, stems):
            # `file` is highly polysemous: lock it only in explicit workshop/tool context.
            if code == "file_tool" and not re.search(r"\b(?:teeth|steel|tool|bench|cutting|three[- ]square|grease|handle|workshop)\b", source, re.I):
                continue
            rows.append(code)
    return rows


def _special_quantity_failures(source: str, target: str) -> list[str]:
    src = str(source or "")
    ru = str(target or "").casefold().replace("ё", "е")
    failures: list[str] = []

    if re.search(r"\bhalf[ -](?:a[ -])?dozen\b", src, re.I):
        if not (re.search(r"\b6\b", ru) or re.search(r"\bшест(?:ь|и|ью)\b", ru) or "полдюжин" in ru):
            failures.append("half_dozen=6")
    elif re.search(r"\b(?:a|one)\s+dozen\b|\bdozen\b", src, re.I):
        if not (re.search(r"\b12\b", ru) or "двенадцат" in ru or "дюжин" in ru):
            failures.append("dozen=12")

    # Quarter-inch may stay imperial or be converted accurately to metric (~6.35 mm).
    if re.search(r"\b(?:a\s+)?quarter[- ]inch\b|\b1/4[- ]inch\b", src, re.I):
        imperial_ok = bool(re.search(r"четверт\w*\s+дюйм|1\s*/\s*4\s*дюйм|0[,.]25\s*дюйм", ru))
        metric = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:мм|миллиметр)", ru)
        metric_ok = False
        if metric:
            try:
                value = float(metric.group(1).replace(",", "."))
                metric_ok = 6.2 <= value <= 6.5
            except ValueError:
                pass
        if not imperial_ok and not metric_ok:
            failures.append("quarter_inch=6.35mm")
    return failures


def _augment_target_for_natural_numeric_forms(source: str, target: str) -> str:
    src = str(source or "")
    ru = str(target or "")
    low = ru.casefold().replace("ё", "е")
    synthetic: list[str] = []

    if re.search(r"\bone\s+or\s+two\b|\bor\s+two\b", src, re.I) and re.search(r"\bпар(?:у|а|е|ой)\b", low):
        synthetic.extend(["1", "2"])
    if re.search(r"\b(?:no more than\s+)?a\s+year\b", src, re.I) and re.search(r"\bгод(?:а|у|ом|е)?\b", low):
        synthetic.append("1")
    if re.search(r"\bthree[- ]square\s+file\b", src, re.I) and re.search(r"тр[её]хгран", ru, re.I):
        synthetic.append("3")
    if re.search(r"\bsix\s+months\b", src, re.I) and re.search(r"\bполгода\b", low):
        synthetic.append("6")
    if re.search(r"\bforty\s+thousand\b", src, re.I) and re.search(r"сорока\s+тысяч", low):
        synthetic.append("40000")
    if re.search(r"\btwelve\s+hundred\b", src, re.I) and re.search(r"двенадцат\w*\s+сот", low):
        synthetic.append("1200")
    # Fractions such as four fifths are frequently rendered with an inflected ordinal noun.
    if re.search(r"\bfour\s+fifths\b", src, re.I) and re.search(r"четыр\w*\s+пят", low):
        synthetic.extend(["4", "5"])
    if re.search(r"\bone\s+fifth\b", src, re.I) and re.search(r"одн\w*\s+пят", low):
        synthetic.extend(["1", "5"])
    return ru + (" " + " ".join(synthetic) if synthetic else "")


def _long_omission(source: str, target: str) -> bool:
    src = str(source or "").strip()
    ru = str(target or "").strip()
    if len(src) < 110:
        return False
    src_beats = len(re.findall(r"[.!?](?:['\"])?(?:\s|$)", src))
    if src_beats < 2:
        return False
    ratio = len(ru) / max(1, len(src))
    # Deliberately conservative: ordinary EN→RU compression should not approach 50%.
    return ratio < 0.56


def _actor_role_failure(source: str, target: str) -> bool:
    src = str(source or "")
    ru = str(target or "").casefold().replace("ё", "е")
    m = re.search(r"\bDid\s+(he|she|they)\s+([A-Za-z]+)\s+(you|him|her|them)\s*\?", src, re.I)
    if not m:
        return False
    obj = m.group(3).casefold()
    expected = {
        "you": r"\b(?:тебя|вас)\b",
        "him": r"\bего\b",
        "her": r"\b(?:ее|её)\b",
        "them": r"\bих\b",
    }[obj]
    if re.search(expected, ru):
        return False
    # High precision only: flag when Russian explicitly substitutes a different pronoun object.
    wrong_objects = r"\b(?:его|ее|её|их|тебя|вас)\b"
    return bool(re.search(wrong_objects, ru))


def _time_relation_failure(source: str, target: str) -> bool:
    if not re.search(r"\bit\s+was\s+tomorrow\s+already\b", str(source or ""), re.I):
        return False
    ru = str(target or "").casefold().replace("ё", "е")
    return not bool(re.search(r"(?:завтра\s+уже\s+наступ|уже\s+(?:был|было)\s+следующ|уже\s+наступил\s+следующ|уже\s+было\s+завтра)", ru))


def _enhanced_simple_failures(segment, target: str) -> list[str]:
    source = str(segment.text or "")
    target = str(target or "")
    codes = list(_BASE_SIMPLE(segment, target))

    # Re-run numeric comparison with synthetic values for natural Russian equivalents.
    if "numeric" in codes:
        augmented = _augment_target_for_natural_numeric_forms(source, target)
        if v9ao.compare_numeric_fidelity(source, augmented).get("ok", True):
            codes = [code for code in codes if code != "numeric"]

    for obligation in _special_quantity_failures(source, target):
        if "numeric" not in codes:
            codes.append("numeric")
        codes.append("quantity:" + obligation)
    if _long_omission(source, target) and "short_omission" not in codes:
        codes.append("long_omission")
    if _actor_role_failure(source, target):
        codes.append("actor_role")
    if _time_relation_failure(source, target):
        codes.append("time_relation")
    tech = _technical_obligations(source, target)
    if tech:
        codes.append("technical_term")
        codes.extend("term:" + item for item in tech)
    return list(dict.fromkeys(codes))


def _rate_resilient_giga_json(system: str, payload: Any, *, max_tokens: int = 5000):
    attempts = max(2, int(os.getenv("BOOKAI_V9AP_GIGA_JSON_ATTEMPTS") or "5"))
    base_sleep = max(0.5, float(os.getenv("BOOKAI_V9AP_GIGA_JSON_BACKOFF") or "1.5"))
    max_sleep = max(base_sleep, float(os.getenv("BOOKAI_V9AP_GIGA_JSON_MAX_SLEEP") or "8"))
    for attempt in range(1, attempts + 1):
        try:
            return _BASE_GIGA_JSON(system, payload, max_tokens=max_tokens)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            if attempt >= attempts:
                _STATS["giga_json_rate_exhausted"] = int(_STATS.get("giga_json_rate_exhausted") or 0) + 1
                raise
            _STATS["giga_json_rate_retries"] = int(_STATS.get("giga_json_rate_retries") or 0) + 1
            delay = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            print(f"[v9ap-giga-json-rate] attempt={attempt}/{attempts} sleep={delay:.2f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _issue_obligations(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for code in row.get("codes", []):
        code = str(code)
        if code.startswith("quantity:"):
            out.append(code.split(":", 1)[1])
        elif code.startswith("term:"):
            out.append("technical=" + code.split(":", 1)[1])
        elif code == "actor_role":
            out.append("preserve grammatical actor and object; do not turn active Did X VERB Y? into passive Was Y VERBed?")
        elif code == "long_omission":
            out.append("restore the complete long segment, every sentence/dialogue beat")
        elif code == "time_relation":
            out.append("It was tomorrow already means the next day had already arrived, not future 'уже завтра'")
    return out


def _giga_repair_resilient(targets, translated, memory, issues, *, strict: bool = False):
    if not issues:
        return [], 0
    batch_size = max(10, int(os.getenv("BOOKAI_V9AO_GIGA_BATCH") or "18"))
    cap = max(16, int(os.getenv("BOOKAI_V9AO_GIGA_MAX") or "48"))
    issues = list(issues)[:cap]
    by_id = {str(segment.id): segment for segment in targets}
    system = """You are a FAST EN→RU publication repairer. Every row has one or more PROVEN local defects.
Do not broadly rewrite correct prose. Return the COMPLETE translation of SOURCE and fix the listed defect.

Rules:
- numeric / quantity:*: preserve exact quantities. dozen=12, half-dozen=6, quarter-inch=1/4 inch=6.35 mm.
- latin_leak: remove accidental English prose; established proper names must be Cyrillic.
- short_omission / long_omission: restore every omitted sentence, dialogue beat, attribution, action and object.
- question: preserve all questions.
- material: preserve exact physical material.
- order: preserve front/back/beginning/end ordering; never invert it.
- actor_role: preserve who performs the action and who receives it.
- time_relation: preserve whether a time has already arrived versus lies in the future.
- technical_term / term:*: use workshop/armour meaning from context, not modern-computing/literal false friends. file(tool)=напильник; cuisses=набедренники/набедренная броня; gorget=горжет/защита шеи; tinplate=луженая жесть/железо; treadle saw=ножная/педальная пила; card a file=прочистить напильник щеткой.
Preserve names, polarity, causality and literary tone. BEFORE_EN/AFTER_EN are context only.
ONLY JSON {\"items\":[{\"id\":\"...\",\"corrected_ru\":\"...\"}]}; exactly one item per id."""
    if strict:
        system += "\nSTRICT RETRY: previous output still failed deterministic validation. Resolve every listed obligation."

    changed: list[str] = []
    calls = 0
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        payload = [{
            "id": row["id"], "failed_checks": row["codes"], "obligations": _issue_obligations(row),
            "source": row["source"], "current_ru": row["current_ru"],
            "before_en": row["before_en"], "after_en": row["after_en"],
        } for row in batch]
        try:
            obj = _rate_resilient_giga_json(system, {"items": payload}, max_tokens=6500)
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            calls += 1
            print(f"[v9ap-giga-repair] strict={strict} error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(x.get("id") or ""): x for x in raw if isinstance(x, dict) and str(x.get("id") or "")}
        for row in batch:
            sid = row["id"]
            item = parsed.get(sid) or {}
            candidate = v9ao.v9am.v9al.v9ah.v9._norm_text(item.get("corrected_ru") or "")
            if not candidate:
                continue
            segment = by_id[sid]
            old = str(translated.get(sid) or "")
            try:
                candidate = v9ao.v9am.v9al.v9ah.v9ag.v9ad._canonicalize_candidate(segment, candidate, memory)
            except Exception:
                pass
            if not candidate or _enhanced_simple_failures(segment, candidate):
                continue
            if v9ao.v9am.v9al.v9ah.v9._fatal_count(segment, candidate, memory) > v9ao.v9am.v9al.v9ah.v9._fatal_count(segment, old, memory):
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _giga_first_route_precision(targets, translated, memory):
    before = {str(s.id): str(translated.get(str(s.id)) or "") for s in targets}
    v9ao._GIGA_STATS["pre_passes"] = int(v9ao._GIGA_STATS.get("pre_passes") or 0) + 1
    v9ao._giga_simple_pass(targets, translated, memory, stage="pre")
    cheap_changed = {
        str(s.id) for s in targets
        if str(translated.get(str(s.id)) or "") != before.get(str(s.id), "")
        and not _enhanced_simple_failures(s, str(translated.get(str(s.id)) or ""))
    }
    _STATS["cheap_changed_ids"] = sorted(set(_STATS.get("cheap_changed_ids") or []) | cheap_changed)

    selected, ranked = _BASE_SEMANTIC_ROUTE(targets, translated, memory)
    _STATS["deepseek_routes_before_filter"] = len(selected)
    dropped = [str(row.get("id") or "") for row in selected if str(row.get("id") or "") in cheap_changed]
    selected = [row for row in selected if str(row.get("id") or "") not in cheap_changed]

    # Keep every mandatory v9ah route, then only the strongest medium-risk tail.
    target = max(6, int(os.getenv("BOOKAI_V9AP_DEEP_MAX") or "10"))
    mandatory = [row for row in selected if int(row.get("priority") or 0) >= 22]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in mandatory + selected:
        sid = str(row.get("id") or "")
        if not sid or sid in seen:
            continue
        if len(out) >= target and int(row.get("priority") or 0) < 22:
            continue
        out.append(row); seen.add(sid)
    out.sort(key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)))
    _STATS["deepseek_routes_dropped_after_cheap_repair"] = dropped
    _STATS["deepseek_routes_after_filter"] = len(out)
    v9ao._GIGA_STATS["deepseek_semantic_routes"] = len(out)
    v9ao._GIGA_STATS["deepseek_semantic_route_ids"] = [str(row.get("id") or "") for row in out]
    print("[v9ap-route] " + json.dumps({
        "before": _STATS["deepseek_routes_before_filter"],
        "cheap_dropped": dropped,
        "after": len(out),
        "mandatory": len([r for r in out if int(r.get("priority") or 0) >= 22]),
    }, ensure_ascii=False), flush=True)
    return out, ranked


def _annotate_report() -> None:
    v3 = v9ao.v9am.v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    arch = dict(data.get("architecture") or {})
    arch.update({
        "experiment": "v9ap-precision-gigachat-first",
        "giga_json_429_policy": "same-batch exponential retry before failure",
        "simple_repair_classes_v2": ["numeric", "dozen/fraction/unit", "latin", "short+long omission", "question", "material", "order", "actor_role", "time_relation", "technical_term"],
        "deepseek_target": "mandatory semantic risks plus strongest tail, target <=10; cheap-repaired rows excluded",
        "deepseek_batch_target": 10,
        "chapter_runtime_sla_seconds": [120, 180],
    })
    data["architecture"] = arch
    data["v9ap_precision"] = dict(_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_simple = v9ao._simple_failures_for_pair
    old_repair = v9ao._giga_repair
    old_route = v9ao._giga_first_route
    old_giga_json = v9ao.v9am.v9al.v9ah.v9af.v9ab._giga_json
    old_batch = os.environ.get("BOOKAI_V9AB_DEEP_BATCH")
    v9ao._simple_failures_for_pair = _enhanced_simple_failures
    v9ao._giga_repair = _giga_repair_resilient
    v9ao._giga_first_route = _giga_first_route_precision
    v9ao.v9am.v9al.v9ah.v9af.v9ab._giga_json = _rate_resilient_giga_json
    os.environ["BOOKAI_V9AB_DEEP_BATCH"] = str(max(10, int(os.environ.get("BOOKAI_V9AB_DEEP_BATCH") or "0")))
    try:
        v9ao.main()
    finally:
        v9ao._simple_failures_for_pair = old_simple
        v9ao._giga_repair = old_repair
        v9ao._giga_first_route = old_route
        v9ao.v9am.v9al.v9ah.v9af.v9ab._giga_json = old_giga_json
        if old_batch is None:
            os.environ.pop("BOOKAI_V9AB_DEEP_BATCH", None)
        else:
            os.environ["BOOKAI_V9AB_DEEP_BATCH"] = old_batch
        _annotate_report()


if __name__ == "__main__":
    main()
