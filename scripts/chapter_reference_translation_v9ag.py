from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import chapter_reference_translation_v9af as v9af
from bookai.release_guards import source_grounded_objective_issues

v9ad = v9af.v9ad
v9ab = v9af.v9ab
v9s, v9t, v9 = v9af.v9s, v9af.v9t, v9af.v9

_BASE_QUALITY = v9af._BASE_QUALITY
_LATIN = v9af._LATIN_WORD_RE
_RU_TOKEN_RE = re.compile(r"[А-Яа-яЁё]+")
_REGISTER_PARTICLES = {"не", "ну", "пожалуйста", "просто", "так", "ладно"}


def _objective_issues(targets, translated) -> list[dict[str, Any]]:
    rows = v9af._objective_issues(targets, translated)
    by_id = {row["id"]: row for row in rows}

    # Stable production release guards contribute only narrow SOURCE-proven
    # invariants. Merge them into the existing objective validator so the same
    # Giga sanitizer and DeepSeek emergency fallback can repair/verify them.
    for extra in source_grounded_objective_issues(list(targets), dict(translated)):
        sid = str(extra["id"])
        row = by_id.get(sid)
        if row is None:
            row = {"id": sid, "index": int(extra["index"]), "codes": [], "reason": ""}
            rows.append(row)
            by_id[sid] = row
        for code in extra.get("codes") or []:
            if code not in row["codes"]:
                row["codes"].append(code)
        if extra.get("codes"):
            label = "source-grounded release invariant: " + ", ".join(extra["codes"])
            row["reason"] = (str(row.get("reason") or "") + "; " + label).strip("; ")

    # High-confidence register extension: an English request/imperative embedded
    # between strongly formal Russian turns should not suddenly become informal,
    # even when the chosen Russian imperative (e.g. "удели") is outside our
    # small explicit informal-word list.
    for i, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        if not v9af._REQUEST_SOURCE_RE.search(source):
            continue
        if v9af._formal_score(targets, translated, i) < 3:
            continue
        if v9af._RU_FORMAL_RE.search(current):
            continue
        row = by_id.get(segment.id)
        if row is None:
            row = {"id": segment.id, "index": i, "codes": [], "reason": ""}
            rows.append(row)
            by_id[segment.id] = row
        if "address_register" not in row["codes"]:
            row["codes"].append("address_register")
            extra = "neighboring dialogue strongly establishes formal Вы-register; keep this request formal"
            row["reason"] = (str(row.get("reason") or "") + "; " + extra).strip("; ")
    return rows


def _issue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "segments": len(rows),
        "by_code": dict(Counter(code for row in rows for code in row.get("codes", []))),
        "ids": [row["id"] for row in rows],
    }


def _candidate_clears_objective_issues(targets, translated, sid: str, candidate: str) -> bool:
    if not candidate:
        return False
    probe = dict(translated)
    probe[sid] = candidate
    return not any(str(row.get("id") or "") == sid for row in _objective_issues(targets, probe))


def _formalize_imperative_token(word: str) -> str:
    """Conservative 2sg -> polite/plural imperative rewrite.

    Russian productive imperative morphology normally forms the polite/plural form
    by adding -те. Reflexive imperatives need a small deterministic spelling
    adjustment. This helper is used only after the objective register detector has
    already proved that the turn is a request in strongly formal context.
    """
    lower = word.casefold()
    if lower.endswith("тесь") or lower.endswith("йте") or lower.endswith("ите"):
        return word
    if lower.endswith("ись"):
        replacement = word[:-3] + "итесь"
    elif lower.endswith("ься"):
        replacement = word[:-3] + "ьтесь"
    elif lower.endswith("йся"):
        replacement = word[:-3] + "йтесь"
    elif lower.endswith("ся"):
        replacement = word[:-2] + "тесь"
    else:
        replacement = word + "те"
    return replacement


def _deterministic_register_fallback(targets, translated, issues) -> list[str]:
    """Close only proven formal-register residuals without inventing content.

    We do not guess whether a turn should be formal here: `_objective_issues` has
    already established that from the English request plus neighboring dialogue.
    The rewrite changes only the first imperative-like Russian token and is accepted
    only if the same objective validator becomes clean for that segment.
    """
    changed: list[str] = []
    for row in issues:
        if "address_register" not in (row.get("codes") or []):
            continue
        i = int(row["index"])
        seg = targets[i]
        sid = str(seg.id)
        if not v9af._REQUEST_SOURCE_RE.search(str(seg.text or "")):
            continue
        current = str(translated.get(sid) or "")
        matches = list(_RU_TOKEN_RE.finditer(current))
        target_match = None
        for match in matches:
            if match.group(0).casefold() in _REGISTER_PARTICLES:
                continue
            target_match = match
            break
        if target_match is None:
            continue
        original = target_match.group(0)
        formal = _formalize_imperative_token(original)
        if formal == original:
            continue
        candidate = current[: target_match.start()] + formal + current[target_match.end() :]
        if _candidate_clears_objective_issues(targets, translated, sid, candidate):
            translated[sid] = candidate
            changed.append(sid)
            print(
                f"[v9ag-register-fallback] id={sid} {original!r}->{formal!r}",
                flush=True,
            )
    return changed


def _strict_retry_batch(targets, translated, issues) -> tuple[list[str], int]:
    if not issues:
        return [], 0
    cap = max(4, int(os.getenv("BOOKAI_V9AG_RETRY_MAX") or "16"))
    issues = issues[:cap]
    payload = []
    for row in issues:
        i = int(row["index"])
        seg = targets[i]
        current = str(translated.get(seg.id) or "")
        payload.append({
            "id": seg.id,
            "failed_checks": row["codes"],
            "remaining_ascii": sorted(set(_LATIN.findall(current))),
            "source": str(seg.text or ""),
            "failed_ru": current,
            "before_ru": [str(translated.get(x.id) or "") for x in targets[max(0, i - 2):i]],
            "after_ru": [str(translated.get(x.id) or "") for x in targets[i + 1:i + 3]],
        })
    system = """Your previous EN→RU correction FAILED deterministic validation. Fix every row again.
This is not optional. Return a COMPLETE translation of SOURCE for every id.

Hard postconditions:
1. latin_leak: corrected_ru must contain ZERO Latin-script words matching [A-Za-z]{2,}. Translate ordinary English words; render ALL proper/place names in Cyrillic too. Do not copy any ASCII word from failed_ru.
2. time_marker:this_afternoon => explicitly preserve today + afternoon/daytime (e.g. сегодня днём / сегодня после полудня).
3. time_marker:mid_afternoon => preserve mid-afternoon, not noon.
4. time_marker:tomorrow_night => preserve tomorrow + evening/night.
5. time_marker:tomorrow_morning => preserve tomorrow + morning.
6. time_marker:in_the_morning => preserve morning.
7. time_marker:tonight => preserve this evening/tonight.
8. time_marker:for_the_night => preserve overnight/for the night.
9. legal_function => use an unmistakably prosecuting Russian role (normally обвинитель), never адвокат/защитник, when the supplied source is attacking the prisoner's defence.
10. address_register => this turn must use formal Вы-register consistent with the supplied neighboring Russian turns; use formal imperative morphology where needed.
11. armor_terminology => SOURCE 'brigandine' must stay that armour type: use бригантина/бригандин, never кольчуга or generic armour.
12. needle_eye_idiom => SOURCE 'eye of a darning-needle' is the needle's eye: use natural Russian 'ушко штопальной иглы'.
13. penultimate_idiom => SOURCE 'last lesson but one' means the penultimate lesson: explicitly use 'предпоследний/предпоследним'.

Do not alter unrelated meaning, numbers, polarity or established Cyrillic names. No reference translation exists.
ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}; exactly one item per input id.
"""
    try:
        obj = v9ab._giga_json(system, {"items": payload}, max_tokens=6500)
    except Exception as exc:
        print(f"[v9ag-retry] error={type(exc).__name__}", flush=True)
        return [], 1
    parsed = {str(x.get("id") or ""): x for x in (obj.get("items") or []) if isinstance(x, dict)}
    changed: list[str] = []
    for row in issues:
        sid = str(row["id"])
        item = parsed.get(sid)
        candidate = v9._norm_text((item or {}).get("corrected_ru") or "")
        if candidate and _candidate_clears_objective_issues(targets, translated, sid, candidate):
            translated[sid] = candidate
            changed.append(sid)
    return changed, 1


def _quality_v9ag(harness, targets, translated, memory):
    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    initial = _objective_issues(targets, translated)
    changed1, calls1 = v9af._mandatory_giga_repair(targets, translated, initial)
    legal1 = v9af._deterministic_legal_fallback(targets, translated, _objective_issues(targets, translated))

    residual1 = _objective_issues(targets, translated)
    changed2, calls2 = _strict_retry_batch(targets, translated, residual1)
    legal2 = v9af._deterministic_legal_fallback(targets, translated, _objective_issues(targets, translated))

    # One final bounded retry only if an objective postcondition still fails.
    residual2 = _objective_issues(targets, translated)
    changed3, calls3 = _strict_retry_batch(targets, translated, residual2)
    legal3 = v9af._deterministic_legal_fallback(targets, translated, _objective_issues(targets, translated))
    register3 = _deterministic_register_fallback(targets, translated, _objective_issues(targets, translated))

    name_postfixes = v9ab._apply_name_canon(targets, translated, memory)
    for segment in targets:
        translated[segment.id] = v9s._format_dialogue_v9s(segment, translated.get(segment.id, ""))[0]

    final_residual = _objective_issues(targets, translated)

    semantic: dict[str, list[dict[str, Any]]] = {}
    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for rows in final_map.values() for row in rows)

    changed_all = sorted(set(changed1 + changed2 + changed3 + legal1 + legal2 + legal3 + register3))
    stats.update({
        "quality_mode": "v9ad-sparse-deepseek+v9ag-validated-source-sanitizer",
        "source_grounded_sanitizer": True,
        "sanitizer_initial": _issue_summary(initial),
        "sanitizer_after_first": _issue_summary(residual1),
        "sanitizer_after_retry": _issue_summary(residual2),
        "sanitizer_final_residual": _issue_summary(final_residual),
        "sanitizer_changed": len(changed_all),
        "sanitizer_changed_ids": changed_all,
        "sanitizer_gigachat_calls": calls1 + calls2 + calls3,
        "sanitizer_legal_fallback_ids": sorted(set(legal1 + legal2 + legal3)),
        "sanitizer_register_fallback_ids": sorted(set(register3)),
        "sanitizer_name_postfixes": name_postfixes,
        "giga_analyst_usage": dict(v9ab._GIGA_ANALYST_USAGE),
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
    })
    v9ab._V9AB_STATS = dict(stats)
    print("[bookai-v9ag] " + json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    v9ab._quality_v9ab = _quality_v9ag
    try:
        v9ad.main()
    finally:
        v9ab._quality_v9ab = _BASE_QUALITY


if __name__ == "__main__":
    main()
