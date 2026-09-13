from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import chapter_reference_translation_v9ad as v9ad

v9ab = v9ad.v9ab
v9s, v9t, v9 = v9ad.v9s, v9ad.v9t, v9ad.v9

_ORIGINAL_QUALITY = v9ab._quality_v9ab

# v9ae does NOT add DeepSeek work.  It runs after v9ad's sparse DeepSeek stage and
# only repairs objective source-grounded publication defects with deterministic
# checks + cheap GigaChat.  The Russian reference translation is unavailable.

_LATIN_WORD_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{2,}(?:['’-][A-Za-z]+)?(?![A-Za-z])")
_LEGAL_SOURCE_RE = re.compile(r"\b(?:advocate|counsel)\b", re.I)
_LEGAL_CONTEXT_RE = re.compile(r"\b(?:prisoner|defen[cs]e|guilt|guilty|crime|offen[cs]e|court|trial|tribunal|sentence)\b", re.I)
_RU_DEFENDER_RE = re.compile(r"\b(?:адвокат\w*|защитник\w*)\b", re.I)
_REQUEST_SOURCE_RE = re.compile(
    r"\b(?:indulge me|tell me|show me|forgive me|excuse me|would you|could you|will you|do you|are you|have you)\b",
    re.I,
)
_RU_FORMAL_RE = re.compile(r"\b(?:вы|вас|вам|вами|ваш\w*)\b", re.I)
_RU_INFORMAL_RE = re.compile(r"\b(?:ты|тебя|тебе|тобой|твой\w*|сделай|скажи|расскажи|покажи|дай|послушай)\b", re.I)

_TIME_RULES = (
    ("afternoon", re.compile(r"\b(?:this |that |the |in the )?afternoon\b", re.I), re.compile(r"\b(?:дн[её]м|днев\w*|после полудня|вторая половина дня)\b", re.I), re.compile(r"\b(?:утр\w*|вечер\w*|ноч\w*)\b", re.I)),
    ("morning", re.compile(r"\b(?:this |that |the |in the )?morning\b", re.I), re.compile(r"\bутр\w*\b", re.I), re.compile(r"\b(?:вечер\w*|ноч\w*|после полудня)\b", re.I)),
    ("evening", re.compile(r"\b(?:this |that |the |in the )?evening\b", re.I), re.compile(r"\bвечер\w*\b", re.I), re.compile(r"\b(?:утр\w*|после полудня)\b", re.I)),
    ("night", re.compile(r"\b(?:this |that |the |at |in the )?night\b", re.I), re.compile(r"\bноч\w*\b", re.I), re.compile(r"\b(?:утр\w*|дн[её]м|днев\w*)\b", re.I)),
    ("noon", re.compile(r"\b(?:at |by |until )?noon\b", re.I), re.compile(r"\bполуд\w*\b", re.I), re.compile(r"\bполноч\w*\b", re.I)),
    ("midnight", re.compile(r"\b(?:at |by |until )?midnight\b", re.I), re.compile(r"\bполноч\w*\b", re.I), re.compile(r"\bполуд\w*\b", re.I)),
)


def _neighbor_text(targets, translated, i: int, *, source: bool) -> list[str]:
    rows: list[str] = []
    for j in range(max(0, i - 2), min(len(targets), i + 3)):
        if j == i:
            continue
        rows.append(str(targets[j].text if source else translated.get(targets[j].id, "") or ""))
    return rows


def _formal_neighbor_score(targets, translated, i: int) -> int:
    text = " ".join(_neighbor_text(targets, translated, i, source=False))
    return len(_RU_FORMAL_RE.findall(text)) - len(_RU_INFORMAL_RE.findall(text))


def _objective_issues(targets, translated) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for i, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        codes: list[str] = []
        reasons: list[str] = []

        latin = sorted(set(_LATIN_WORD_RE.findall(current)))
        if latin:
            codes.append("latin_leak")
            reasons.append("untranslated Latin words in Russian output: " + ", ".join(latin[:12]))

        for label, src_re, good_re, conflict_re in _TIME_RULES:
            if not src_re.search(source):
                continue
            if not good_re.search(current) or conflict_re.search(current):
                codes.append("time_marker")
                reasons.append(f"source explicitly marks {label}; preserve the same time of day in Russian")
            break

        context_en = " ".join(_neighbor_text(targets, translated, i, source=True))
        if _LEGAL_SOURCE_RE.search(source) and _RU_DEFENDER_RE.search(current) and _LEGAL_CONTEXT_RE.search(source + " " + context_en):
            codes.append("legal_function")
            reasons.append("infer courtroom function from argument/context; Russian role must not misleadingly imply defence if speaker prosecutes")

        if _REQUEST_SOURCE_RE.search(source) and _RU_INFORMAL_RE.search(current) and _formal_neighbor_score(targets, translated, i) >= 2:
            codes.append("address_register")
            reasons.append("surrounding dialogue consistently uses formal Вы-register; avoid an accidental switch to Ты")

        if codes:
            issues.append({
                "id": segment.id,
                "index": i,
                "codes": codes,
                "reason": "; ".join(reasons),
            })
    return issues


def _repair_objective_with_giga(targets, translated, issues) -> tuple[list[str], int]:
    if not issues:
        return [], 0
    batch_size = max(4, int(os.getenv("BOOKAI_V9AE_SANITIZER_BATCH") or "10"))
    cap = max(8, int(os.getenv("BOOKAI_V9AE_SANITIZER_MAX") or "28"))
    issues = issues[:cap]
    system = """You are a FINAL SOURCE-GROUNDED EN→RU publication sanitizer. This stage must NOT imitate any reference translation.
For each item, compare SOURCE and neighboring English against current_ru and repair ONLY the listed objective defect(s).
Do not rewrite harmless literary choices. Return a COMPLETE Russian translation of exactly the source segment when change=true.

Rules:
- latin_leak: Russian prose must contain no accidental untranslated English words/fragments. Translate them naturally. Proper names must use the established Cyrillic spellings already visible in current_ru/neighbors when available.
- time_marker: preserve the exact source time-of-day. afternoon is not morning; morning is not evening; noon is not midnight.
- legal_function: translate the FUNCTION performed in this courtroom context, not a misleading dictionary label. If the speaker argues the prisoner's offence is serious, attacks the defence, and seeks guilt/punishment, Russian wording should denote the accusing/prosecuting side. Do not change a genuine defender.
- address_register: preserve the established Russian Ты/Вы relationship from neighboring dialogue. Do not switch register because an English imperative has no T/V distinction.
- Preserve all propositions, numbers, negation, participants and established Cyrillic proper names.
- Do not “correct” debatable literary choices such as wood/лес/роща/поляна unless the listed defect requires it.

Return exactly one row per id. ONLY JSON:
{"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}.
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
            obj = v9ab._giga_json(system, {"items": payload}, max_tokens=5200)
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ae-sanitizer] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(x.get("id") or ""): x for x in raw if isinstance(x, dict)}
        for row in batch:
            sid = str(row["id"])
            item = parsed.get(sid)
            if not item or sid not in expected or type(item.get("change")) is not bool or not item.get("change"):
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            candidate = v9._norm_text(item.get("corrected_ru") or "")
            if confidence < 0.62 or not candidate:
                continue
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _quality_v9ae(harness, targets, translated, memory):
    stats = dict(_ORIGINAL_QUALITY(harness, targets, translated, memory) or {})

    detected = _objective_issues(targets, translated)
    changed1, calls1 = _repair_objective_with_giga(targets, translated, detected)

    # Re-enforce the same source-only proper-name canon AFTER Giga sanitization.
    name_postfixes = v9ab._apply_name_canon(targets, translated, memory)
    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]

    # One bounded recovery round only for objective defects that are still visible.
    residual1 = _objective_issues(targets, translated)
    changed2, calls2 = _repair_objective_with_giga(targets, translated, residual1)
    if changed2:
        name_postfixes += v9ab._apply_name_canon(targets, translated, memory)
        for segment in targets:
            translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]
    residual2 = _objective_issues(targets, translated)

    semantic: dict[str, list[dict[str, Any]]] = {}
    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)

    by_code = Counter(code for row in detected for code in row.get("codes", []))
    residual_by_code = Counter(code for row in residual2 for code in row.get("codes", []))
    stats.update({
        "quality_mode": "v9ad-sparse-deepseek+source-grounded-gigachat-sanitizer",
        "source_grounded_sanitizer": True,
        "sanitizer_detected": len(detected),
        "sanitizer_detected_by_code": dict(by_code),
        "sanitizer_changed": len(set(changed1 + changed2)),
        "sanitizer_changed_ids": sorted(set(changed1 + changed2)),
        "sanitizer_gigachat_calls": calls1 + calls2,
        "sanitizer_name_postfixes": name_postfixes,
        "sanitizer_residual": len(residual2),
        "sanitizer_residual_by_code": dict(residual_by_code),
        "sanitizer_residual_ids": [row["id"] for row in residual2],
        "giga_analyst_usage": dict(v9ab._GIGA_ANALYST_USAGE),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
    })
    v9ab._V9AB_STATS = dict(stats)
    print("[bookai-v9ae] " + json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    # v9ad still owns the sparse DeepSeek policy. Only its final quality function is
    # wrapped with a zero-DeepSeek objective sanitizer.
    v9ab._quality_v9ab = _quality_v9ae
    try:
        v9ad.main()
    finally:
        v9ab._quality_v9ab = _ORIGINAL_QUALITY


if __name__ == "__main__":
    main()
