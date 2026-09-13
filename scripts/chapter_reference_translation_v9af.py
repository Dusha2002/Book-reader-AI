from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import chapter_reference_translation_v9ae as v9ae

v9ad = v9ae.v9ad
v9ab = v9ae.v9ab
v9s, v9t, v9 = v9ae.v9s, v9ae.v9t, v9ae.v9

_BASE_QUALITY = v9ae._ORIGINAL_QUALITY

# v9af narrows v9ae's detector to high-precision source facts and makes repairs
# mandatory. DeepSeek policy remains IDENTICAL to v9ad: max 3 repair batches +
# 2 verify batches per chapter. All new work is deterministic + GigaChat.

_LATIN_WORD_RE = v9ae._LATIN_WORD_RE
_RU_FORMAL_RE = v9ae._RU_FORMAL_RE
_RU_INFORMAL_RE = v9ae._RU_INFORMAL_RE
_REQUEST_SOURCE_RE = v9ae._REQUEST_SOURCE_RE
_LEGAL_SOURCE_RE = v9ae._LEGAL_SOURCE_RE
_LEGAL_CONTEXT_RE = v9ae._LEGAL_CONTEXT_RE
_RU_DEFENDER_RE = v9ae._RU_DEFENDER_RE

_TIME_FACTS = (
    (
        "this_afternoon",
        re.compile(r"\bthis afternoon\b", re.I),
        re.compile(r"(?:сегодня[^.!?]{0,24}(?:дн[её]м|дн[яе]|после полудня|во второй половине дня)|(?:дн[её]м|после полудня)[^.!?]{0,24}сегодня)", re.I),
        "source says THIS AFTERNOON: preserve both today and afternoon/daytime",
    ),
    (
        "mid_afternoon",
        re.compile(r"\bmid[- ]afternoon\b", re.I),
        re.compile(r"(?:середин\w*[^.!?]{0,16}дн\w*|втор\w* половин\w* дн\w*|после полудня|дн[её]м)", re.I),
        "source says MID-AFTERNOON: do not reduce it to noon/poluden",
    ),
    (
        "tomorrow_night",
        re.compile(r"\btomorrow night\b", re.I),
        re.compile(r"завтра[^.!?]{0,24}(?:вечер\w*|ноч\w*)", re.I),
        "source says TOMORROW NIGHT: preserve tomorrow + evening/night",
    ),
    (
        "tomorrow_morning",
        re.compile(r"\btomorrow morning\b", re.I),
        re.compile(r"завтра[^.!?]{0,24}утр\w*", re.I),
        "source says TOMORROW MORNING: preserve tomorrow + morning",
    ),
    (
        "in_the_morning",
        re.compile(r"\bin the morning\b", re.I),
        re.compile(r"\bутр\w*\b", re.I),
        "source says IN THE MORNING: preserve morning",
    ),
    (
        "tonight",
        re.compile(r"\btonight\b", re.I),
        re.compile(r"(?:сегодня[^.!?]{0,24}(?:вечер\w*|ноч\w*)|(?:вечер\w*|ноч\w*)[^.!?]{0,24}сегодня)", re.I),
        "source says TONIGHT: preserve tonight as this evening/night",
    ),
    (
        "for_the_night",
        re.compile(r"\bfor the night\b", re.I),
        re.compile(r"(?:на ночь|на ночлег|до утра)", re.I),
        "source says FOR THE NIGHT: preserve the overnight meaning",
    ),
)


def _neighbors(targets, translated, i: int, *, source: bool) -> list[str]:
    return v9ae._neighbor_text(targets, translated, i, source=source)


def _formal_score(targets, translated, i: int) -> int:
    return v9ae._formal_neighbor_score(targets, translated, i)


def _objective_issues(targets, translated) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        codes: list[str] = []
        reasons: list[str] = []

        latin = sorted(set(_LATIN_WORD_RE.findall(current)))
        if latin:
            codes.append("latin_leak")
            reasons.append("accidental untranslated Latin words: " + ", ".join(latin[:16]))

        for code, src_re, target_re, reason in _TIME_FACTS:
            if src_re.search(source) and not target_re.search(current):
                codes.append("time_marker:" + code)
                reasons.append(reason)

        context_en = " ".join(_neighbors(targets, translated, i, source=True))
        if (
            _LEGAL_SOURCE_RE.search(source)
            and _RU_DEFENDER_RE.search(current)
            and _LEGAL_CONTEXT_RE.search(source + " " + context_en)
            and re.search(r"\b(?:prisoner|defen[cs]e|pleads? not guilty|offen[cs]e|abomination)\b", source + " " + context_en, re.I)
        ):
            codes.append("legal_function")
            reasons.append("speaker is functioning as the accusing/prosecuting side; Russian 'адвокат/защитник' would misleadingly mean defender")

        if _REQUEST_SOURCE_RE.search(source) and _RU_INFORMAL_RE.search(current) and _formal_score(targets, translated, i) >= 2:
            codes.append("address_register")
            reasons.append("neighboring dialogue establishes formal Вы-register; preserve that relationship")

        if codes:
            rows.append({"id": segment.id, "index": i, "codes": codes, "reason": "; ".join(reasons)})
    return rows


def _mandatory_giga_repair(targets, translated, issues) -> tuple[list[str], int]:
    if not issues:
        return [], 0
    batch_size = max(6, int(os.getenv("BOOKAI_V9AF_SANITIZER_BATCH") or "18"))
    cap = max(10, int(os.getenv("BOOKAI_V9AF_SANITIZER_MAX") or "32"))
    issues = issues[:cap]
    system = """MANDATORY final EN→RU source-fidelity repair. Every input item contains at least one objectively detected defect.
You MUST return a corrected_ru for EVERY id; there is no keep/skip option. Correct only the listed issue codes and any
immediate grammar damage caused by that correction. Do not imitate a reference translation and do not rewrite harmless style.

Requirements:
- latin_leak: remove ALL accidental English/Latin prose fragments by translating them naturally into Russian. Do not leave words such as efficiently, apparently, assumed, expendable, boots, anyway, Butter Pass, etc. Proper names/place names must be rendered consistently in Cyrillic.
- time_marker:*: preserve the exact source temporal fact. this afternoon != morning; mid-afternoon != noon; tomorrow night may be tomorrow evening/night; tonight must remain tonight.
- legal_function: if context shows the titled advocate is attacking the prisoner's defence and arguing guilt/seriousness, render the functional Russian role as обвинитель (or another unmistakably prosecuting term), not адвокат/защитник.
- address_register: preserve the established Вы-register from neighboring Russian dialogue, including pronouns and imperative morphology.
- Preserve propositions, participants, numbers, polarity, names and dialogue intent. Return a COMPLETE translation of exactly SOURCE.

ONLY JSON {"items":[{"id":"...","corrected_ru":"...","reason":"..."}]}; exactly one row for every id.
"""
    changed: list[str] = []
    calls = 0
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        payload = []
        for row in batch:
            i = int(row["index"])
            segment = targets[i]
            payload.append({
                "id": segment.id,
                "issue_codes": row["codes"],
                "issue_reason": row["reason"],
                "source": str(segment.text or ""),
                "current_ru": str(translated.get(segment.id) or ""),
                "before_en": [x.text for x in targets[max(0, i - 2):i]],
                "after_en": [x.text for x in targets[i + 1:i + 3]],
                "before_ru": [str(translated.get(x.id) or "") for x in targets[max(0, i - 2):i]],
                "after_ru": [str(translated.get(x.id) or "") for x in targets[i + 1:i + 3]],
            })
        expected = {x["id"] for x in payload}
        try:
            obj = v9ab._giga_json(system, {"items": payload}, max_tokens=6200)
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            print(f"[v9af-sanitizer] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(x.get("id") or ""): x for x in raw if isinstance(x, dict)}
        for row in batch:
            sid = str(row["id"])
            item = parsed.get(sid)
            if not item or sid not in expected:
                continue
            candidate = v9._norm_text(item.get("corrected_ru") or "")
            if not candidate:
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _deterministic_legal_fallback(targets, translated, issues) -> list[str]:
    """Safe fallback for the one unambiguous Russian-role ambiguity we detect.

    Only runs where source/context independently established a prosecuting function.
    It never fires for ordinary uses of advocate/counsel.
    """
    changed: list[str] = []
    issue_ids = {row["id"] for row in issues if "legal_function" in row.get("codes", [])}
    for segment in targets:
        if segment.id not in issue_ids:
            continue
        current = str(translated.get(segment.id) or "")
        value = re.sub(r"\bАдвокат\b", "Обвинитель", current)
        value = re.sub(r"\bадвокат\b", "обвинитель", value)
        value = re.sub(r"\bЗащитник\b", "Обвинитель", value)
        value = re.sub(r"\bзащитник\b", "обвинитель", value)
        if value != current:
            translated[segment.id] = value
            changed.append(segment.id)
    return changed


def _quality_v9af(harness, targets, translated, memory):
    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    detected = _objective_issues(targets, translated)
    changed, calls = _mandatory_giga_repair(targets, translated, detected)

    # If the model still preserved a defender-labelled title in a source-proven
    # prosecuting context, apply the narrow deterministic role fallback.
    residual_after_giga = _objective_issues(targets, translated)
    legal_fallback = _deterministic_legal_fallback(targets, translated, residual_after_giga)

    name_postfixes = v9ab._apply_name_canon(targets, translated, memory)
    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]

    residual = _objective_issues(targets, translated)

    semantic: dict[str, list[dict[str, Any]]] = {}
    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)
    by_code = Counter(code for row in detected for code in row.get("codes", []))
    residual_by_code = Counter(code for row in residual for code in row.get("codes", []))

    stats.update({
        "quality_mode": "v9ad-sparse-deepseek+v9af-mandatory-source-sanitizer",
        "source_grounded_sanitizer": True,
        "sanitizer_detected": len(detected),
        "sanitizer_detected_by_code": dict(by_code),
        "sanitizer_changed": len(set(changed + legal_fallback)),
        "sanitizer_changed_ids": sorted(set(changed + legal_fallback)),
        "sanitizer_gigachat_calls": calls,
        "sanitizer_legal_fallback_ids": legal_fallback,
        "sanitizer_name_postfixes": name_postfixes,
        "sanitizer_residual": len(residual),
        "sanitizer_residual_by_code": dict(residual_by_code),
        "sanitizer_residual_ids": [row["id"] for row in residual],
        "giga_analyst_usage": dict(v9ab._GIGA_ANALYST_USAGE),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
    })
    v9ab._V9AB_STATS = dict(stats)
    print("[bookai-v9af] " + json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    v9ab._quality_v9ab = _quality_v9af
    try:
        v9ad.main()
    finally:
        v9ab._quality_v9ab = _BASE_QUALITY


if __name__ == "__main__":
    main()
