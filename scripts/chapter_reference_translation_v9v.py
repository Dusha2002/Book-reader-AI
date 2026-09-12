from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9u as v9u

v9t = v9u.v9t
v9s, v9r, v9, v8, v6, v3 = v9u.v9s, v9u.v9r, v9u.v9, v9u.v8, v9u.v6, v9u.v3
_BASE_QUALITY = v9t._quality_v9t
_V9V_STATS: dict[str, Any] = {}

# v9v keeps v9t's fast full-QE path, then spends extra LLM calls only on
# source segments whose surface form is genuinely risky. This is intentionally
# high-recall at the cheap deterministic stage and high-precision after the
# LLM verifier. No Russian gold/reference is available to the runtime.

_RATIO_RE = re.compile(
    r"\b(?:outnumber(?:ed|ing)?|odds?|ratio|twice|thrice|half|double|triple|"
    r"more\s+than|less\s+than|fewer\s+than|at\s+least|at\s+most|"
    r"no\s+more\s+than|no\s+less\s+than)\b",
    re.I,
)
_POLARITY_RE = re.compile(
    r"\b(?:not|never|neither|nor|unless|hardly|scarcely|barely|without|"
    r"can(?:not|'t)\s+be\s+too|wouldn(?:'t|\s+not)|couldn(?:'t|\s+not)|"
    r"shouldn(?:'t|\s+not)|mustn(?:'t|\s+not))\b",
    re.I,
)
_REFERENT_RE = re.compile(
    r"\b(?:either|neither|both|former|latter|the\s+other|each\s+other|"
    r"one\s+another|them|themselves|his|her|their)\b",
    re.I,
)
_IDIOM_RE = re.compile(
    r"\b(?:indulge\s+me|make\s+a\s+living|living\s+to\s+make|"
    r"takes?\s+(?:a\s+)?(?:little\s+)?(?:while|time)\s+to|"
    r"last\s+[^.!?]{0,40}\s+but\s+one|you\s+did|i\s+should\s+do|"
    r"said\s+yes|good\s+idea\s+at\s+the\s+time|with\s+it)\b",
    re.I,
)
_ROLE_RE = re.compile(
    r"\b(?:advocate|counsel|prosecutor|attorney|clerk|secretary|magistrate|"
    r"master|warden|guard|officer|captain|judge|engineer)\b",
    re.I,
)
_AUX_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|do|does|did|have|has|had|can|could|"
    r"shall|should|will|would|may|might|must)\b",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|I've|I'd|I'll|I'm)\b", re.I)
_RU_NAME_RE = re.compile(r"(?<![А-Яа-яЁё])([А-ЯЁ][а-яё]{2,})(?![А-Яа-яЁё])")
_RU_STOP_NAMES = {
    "Он", "Она", "Они", "Это", "Если", "Когда", "Потом", "Теперь", "Тогда",
    "Да", "Нет", "Но", "Или", "Так", "Что", "Как", "Кто", "Где", "Вот",
    "Уже", "Ещё", "Можно", "Нельзя", "Глава", "Город", "Города",
}


def _blocking(row: dict[str, Any]) -> bool:
    return v9s._blocking(row)


def _risk_score(segment, current_ru: str) -> tuple[int, list[str]]:
    source = str(segment.text or "")
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", source)
    score = 0
    reasons: list[str] = []

    if _IDIOM_RE.search(source):
        score += 7
        reasons.append("idiom_or_ellipsis_cue")
    if _RATIO_RE.search(source):
        score += 6
        reasons.append("comparison_or_ratio")
    if _POLARITY_RE.search(source):
        score += 4
        reasons.append("polarity_or_scope")
    if _ROLE_RE.search(source):
        score += 4
        reasons.append("role_or_word_sense")
    if _REFERENT_RE.search(source):
        score += 3
        reasons.append("referent")
    if v9._is_dialogue(source) and len(words) <= 10 and (_AUX_RE.search(source) or _FIRST_PERSON_RE.search(source)):
        score += 5
        reasons.append("short_dialogue_ellipsis")
    if _FIRST_PERSON_RE.search(source) and re.search(r"\b(?:сказала|была|сделала|думала|знала|хотела|могла|должна)\b", current_ru, re.I):
        score += 6
        reasons.append("possible_speaker_gender")
    if re.search(r"[A-Za-z]{3,}", current_ru):
        score += 8
        reasons.append("raw_english")
    if current_ru.count("«") != current_ru.count("»"):
        score += 5
        reasons.append("broken_quotes")
    return score, reasons


def _prefilter(targets, translated):
    ranked = []
    for i, segment in enumerate(targets):
        current = str(translated.get(segment.id) or "")
        score, reasons = _risk_score(segment, current)
        if score >= 4:
            ranked.append((score, i, reasons))

    fraction = max(0.08, min(0.30, float(os.getenv("BOOKAI_V9V_RISK_FRACTION") or "0.18")))
    cap = max(8, int(os.getenv("BOOKAI_V9V_RISK_MAX") or "44"))
    wanted = min(cap, max(12, math.ceil(len(targets) * fraction)))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    ranked = ranked[:wanted]

    ids = {targets[i].id for _, i, _ in ranked}
    heuristic_rows = []
    for score, i, reasons in ranked:
        s = targets[i]
        confidence = min(0.94, 0.72 + score * 0.02)
        heuristic_rows.append({
            "id": s.id,
            "severity": "major",
            "confidence": confidence,
            "code": "risk_prefilter",
            "source_span": s.text[:220],
            "target_span": str(translated.get(s.id) or "")[:220],
            "reason": ", ".join(reasons),
            "repairability": "local",
            "risk_score": score,
        })
    return ids, heuristic_rows


def _scan_selected(harness, targets, translated, selected_ids: set[str]):
    if not selected_ids:
        return []
    items = [v9u._ctx(targets, translated, i) for i, s in enumerate(targets) if s.id in selected_ids]
    cap = max(6000, int(os.getenv("BOOKAI_V9V_SCAN_BATCH_CHARS") or "12000"))
    batches, cur, size = [], [], 0
    for item in items:
        weight = len(item["source"]) + len(item["current_ru"]) + sum(map(len, item["before_en"] + item["after_en"]))
        if cur and size + weight > cap:
            batches.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += weight
    if cur:
        batches.append(cur)

    system = """You are a HIGH-PRECISION challenge scanner for EN→RU literary translation.
The deterministic prefilter already selected risky segments. Do not rewrite. Flag only a clear publication-level meaning defect supported by SOURCE and local context: idiom/pragmatics; reversed comparison or ratio; polarity/scope/modality; short-dialogue ellipsis; wrong actor/role/word sense; referent/gender; technical term corruption; meaning-changing calque; broken grammar/quotes. Do not flag stylistic taste or a merely different but faithful rendering. Max one issue per item. ONLY JSON {\"issues\":[{\"id\":\"...\",\"code\":\"idiom|comparison|polarity|ellipsis|role|referent|gender|term|calque|grammar\",\"severity\":\"critical|major\",\"confidence\":0.0,\"source_span\":\"...\",\"target_span\":\"...\",\"reason\":\"...\"}]}"""

    def one(batch):
        try:
            return v8._complete_json(harness.gate, system, {"items": batch}).get("issues") or []
        except Exception as exc:
            print(f"[v9v-scan] {type(exc).__name__}", flush=True)
            return []

    raw = []
    workers = max(1, min(3, int(os.getenv("BOOKAI_V9V_SCAN_WORKERS") or "3"), len(batches) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9v-scan") as pool:
        for future in as_completed([pool.submit(one, b) for b in batches]):
            raw.extend(future.result())

    valid = {s.id for s in targets}
    out = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        if sid not in valid:
            continue
        severity = str(row.get("severity") or "major").lower()
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0
        # High precision is intentional: v9u over-confirmed too many stylistic cases.
        threshold = 0.78 if severity == "critical" else 0.84
        if severity not in {"critical", "major"} or confidence < threshold:
            continue
        out.append({
            "id": sid,
            "severity": severity,
            "confidence": min(1.0, max(0.0, confidence)),
            "code": str(row.get("code") or "challenge"),
            "source_span": v9._norm_text(row.get("source_span") or "")[:240],
            "target_span": v9._norm_text(row.get("target_span") or "")[:240],
            "reason": v9._norm_text(row.get("reason") or "clear source-faithfulness defect")[:520],
            "repairability": "local",
        })
    return out


def _high_precision(rows):
    out = []
    for row in rows:
        if not _blocking(row):
            continue
        sev = str(row.get("severity") or "major")
        try:
            conf = float(row.get("confidence") or 0)
        except Exception:
            conf = 0
        if sev == "critical" and conf >= 0.78:
            out.append(dict(row))
        elif sev == "major" and conf >= 0.86:
            out.append(dict(row))
    return out


def _recurring_name_tokens(targets, translated):
    counts = Counter()
    for s in targets:
        text = str(translated.get(s.id) or "")
        for token in _RU_NAME_RE.findall(text):
            if token not in _RU_STOP_NAMES and len(token) >= 4:
                counts[token.casefold()] += 1
    return {token for token, count in counts.items() if count >= 3}


def _locks_for(segment, current: str, memory, recurring_names: set[str]):
    names = []
    for token in _RU_NAME_RE.findall(current):
        if token.casefold() in recurring_names and token not in _RU_STOP_NAMES:
            names.append(token)

    glossary = []
    source_low = segment.text.casefold()
    current_low = current.casefold()
    for src, ru in dict(getattr(memory, "glossary", {}) or {}).items():
        src = v9._norm_text(src)
        ru = v9._norm_text(ru)
        if len(src) < 3 or len(ru) < 3:
            continue
        if src.casefold() in source_low and ru.casefold() in current_low:
            glossary.append(ru)
        if len(glossary) >= 10:
            break
    return sorted(set(names)), sorted(set(glossary))


def _preserves_locks(candidate: str, names: list[str], glossary: list[str]) -> bool:
    low = candidate.casefold()
    for name in names:
        if not re.search(rf"(?<![А-Яа-яЁё]){re.escape(name.casefold())}(?![А-Яа-яЁё])", low):
            return False
    for term in glossary:
        if term.casefold() not in low:
            return False
    return True


def _two_candidates(harness, payload):
    system = """Retranslate ONE EN segment into publication-quality Russian because independent QE confirmed a likely meaning defect. Do not edit the old Russian word-by-word. Return exactly two COMPLETE fresh alternatives: (1) faithful_literary — exact actor/relation/polarity/numbers/idiom with natural professional Russian; (2) contextual_literary — same exact meaning, but resolve ellipsis/referents/speaker intent from neighboring context and freely restructure English syntax. Preserve established proper names and valid glossary terminology exactly. Never add facts. ONLY JSON {\"faithful_literary\":\"...\",\"contextual_literary\":\"...\"}."""
    try:
        obj = v8._complete_json(harness.gate, system, payload)
    except Exception as exc:
        print(f"[v9v-candidates] {payload.get('id')} {type(exc).__name__}", flush=True)
        return []
    out = []
    for key in ("faithful_literary", "contextual_literary"):
        value = v9._norm_text(obj.get(key) or "")
        if value and value not in out:
            out.append(value)
    return out


def _repair(harness, targets, translated, memory, finding_map, risk_scores):
    pos = {s.id: i for i, s in enumerate(targets)}
    recurring = _recurring_name_tokens(targets, translated)

    candidates = []
    for s in targets:
        rows = [r for r in finding_map.get(s.id, []) if _blocking(r) and r.get("code") not in {"entity", "thread"}]
        if not rows:
            continue
        severity_rank = 0 if any(str(r.get("severity")) == "critical" for r in rows) else 1
        confidence = max(float(r.get("confidence") or 0) for r in rows)
        risk = risk_scores.get(s.id, 0)
        candidates.append((severity_rank, -confidence, -risk, pos[s.id], s.id))

    candidates.sort()
    cap = max(1, int(os.getenv("BOOKAI_V9V_RETRANSLATE_MAX") or "10"))
    ids = [row[-1] for row in candidates[:cap]]
    workers = max(1, min(5, int(os.getenv("BOOKAI_V9V_RETRANSLATE_WORKERS") or "5"), len(ids) or 1))

    def one(sid):
        i = pos[sid]
        seg = targets[i]
        current = str(translated.get(sid) or "")
        rows = [r for r in finding_map.get(sid, []) if _blocking(r) and r.get("code") not in {"entity", "thread"}]
        payload = v9u._payload(targets, translated, memory, i, rows)
        names, glossary = _locks_for(seg, current, memory, recurring)
        payload["hard_locks"] = {"proper_names": names, "glossary_terms": glossary}

        fresh = _two_candidates(harness, payload)
        safe = [current]
        for value in fresh:
            if v9._fatal_count(seg, value, memory) > v9._fatal_count(seg, current, memory):
                continue
            if not _preserves_locks(value, names, glossary):
                print(f"[v9v-lock-reject] id={sid}", flush=True)
                continue
            safe.append(value)

        dedup, seen = [], set()
        for value in safe:
            key = v9._norm_text(value)
            if key and key not in seen:
                seen.add(key)
                dedup.append(value)
        if len(dedup) < 2:
            return sid, ""

        chosen = dedup[v9u._judge(harness, payload, dedup)]
        if v9._norm_text(chosen) == v9._norm_text(current):
            return sid, ""
        return sid, chosen

    results = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9v-repair") as pool:
        for future in as_completed([pool.submit(one, sid) for sid in ids]):
            sid, value = future.result()
            if value:
                results[sid] = value

    changed = []
    for sid in ids:
        if sid in results:
            translated[sid] = results[sid]
            changed.append(sid)
    return changed, ids


def _quality_v9v(harness, targets, translated, memory):
    global _V9V_STATS

    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})

    # Reverify every blocker from v9t, including priority_invariant. In v9v it is
    # evidence, never an axiom.
    base = {sid: v9t._semantic_rows(rows) for sid, rows in v9._V9_FINAL_FINDINGS.items()}
    old_blockers = [r for rows in base.values() for r in rows if _blocking(r)]
    old_verified = _high_precision(v9u._verify(harness, targets, translated, old_blockers))
    semantic = v9u._merge_verified(base, old_verified)

    selected_ids, heuristic = _prefilter(targets, translated)
    risk_scores = {str(r["id"]): int(r.get("risk_score") or 0) for r in heuristic}
    scanned = _scan_selected(harness, targets, translated, selected_ids)

    # The scanner must agree with an independent verifier. Deterministic rows are
    # supplied as extra evidence, but only high-confidence verified defects block.
    alleged = [*scanned, *heuristic]
    confirmed = _high_precision(v9u._verify(harness, targets, translated, alleged))
    merged_confirmed = v9._merge_findings(confirmed)
    for sid, rows in merged_confirmed.items():
        semantic.setdefault(sid, []).extend(rows)

    finding_map, scores, det_before = v9t._rebuild_full_map(targets, translated, memory, semantic)
    changed, repair_selected = _repair(harness, targets, translated, memory, finding_map, risk_scores)

    changed_set = set(changed)
    for sid in changed_set:
        seg = next((s for s in targets if s.id == sid), None)
        if seg is not None:
            translated[sid] = v9s._format_dialogue_v9s(seg, translated.get(sid, ""))[0]

    remaining = []
    if changed_set:
        fresh = v9t._selected_audit(harness, targets, translated, memory, changed_set)
        refreshed = {}
        for row in fresh:
            refreshed.setdefault(str(row.get("id") or ""), []).append(dict(row))
        for sid in changed_set:
            semantic[sid] = v9t._semantic_rows(refreshed.get(sid, []))

        alleged_after = [r for sid in changed_set for r in semantic.get(sid, []) if _blocking(r)]
        remaining = _high_precision(v9u._verify(harness, targets, translated, alleged_after))
        verified_by = {}
        for row in remaining:
            verified_by.setdefault(str(row.get("id") or ""), []).append(dict(row))
        for sid in changed_set:
            semantic[sid] = [r for r in semantic.get(sid, []) if not _blocking(r)] + verified_by.get(sid, [])

    for s in targets:
        translated[s.id] = v9s._format_dialogue_v9s(s, translated.get(s.id, ""))[0]

    final_map, final_scores, det_final = v9t._rebuild_full_map(targets, translated, memory, semantic)
    critical, major = v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(r.get("severity") for rows in final_map.values() for r in rows)

    _V9V_STATS = {
        "risk_prefilter_selected": len(selected_ids),
        "risk_prefilter_fraction": round(len(selected_ids) / max(1, len(targets)), 3),
        "scanner_findings": len(scanned),
        "challenge_confirmed_high_precision": len(confirmed),
        "existing_blockers_reverified": len(old_blockers),
        "existing_blockers_confirmed": len(old_verified),
        "priority_invariants_are_allegations": True,
        "two_candidate_selected": len(repair_selected),
        "two_candidate_changed": len(changed),
        "two_candidate_changed_ids": changed,
        "hard_entity_glossary_locks": True,
        "delta_remaining_verified": len(remaining),
        "deterministic_before_retranslation": det_before,
        "final_deterministic": det_final,
        "remaining_critical": len(critical),
        "remaining_major_high_confidence": len(major),
        "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
        "severity_counts": dict(counts),
        "quality_mode": "v9t-fast+risk-prefilter+selected-challenge+consensus+2way-retranslation+hard-locks+delta-reaudit",
    }
    stats.update(_V9V_STATS)
    print("[bookai-v9v] " + json.dumps(_V9V_STATS, ensure_ascii=False), flush=True)
    return stats


def _annotate_v9v():
    if not v3.REPORT.exists():
        return
    try:
        data = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9v-risk-filtered-selective-mbr",
        "gold_reference_available_to_pipeline": False,
        "risk_policy": "deterministic high-recall prefilter; LLM challenge scan only on selected risk segments",
        "priority_policy": "priority_invariant is an allegation requiring independent verification",
        "repair_policy": "2 fresh literary candidates + current translation + source-aware judge",
        "entity_policy": "hard-lock recurring established RU names and active glossary terms before candidate acceptance",
        "performance": "v9t fast path plus bounded selected scan and max-10 repairs/chapter",
        "publication_gate": "high-precision confirmed critical/major after delta re-audit blocks completion",
    }
    data["v9v_stats"] = dict(_V9V_STATS)
    v3.REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main():
    v9._configure_v9()
    v9._MATERIALS["coal"] = ("угл",)
    v9s._deterministic_v9s = v9u._deterministic
    v9._deterministic_findings = v9u._deterministic
    v9._thread_findings = v9s._thread_v9s
    v8._fix_near_entity_typos = v9s._entity_fix_v9s
    v9._format_dialogue_v9 = v9s._format_dialogue_v9s
    v8._normalize_dialogue_v8 = v9s._format_dialogue_v9s
    v3._semantic_short_repair = _quality_v9v
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v9r.v9j._annotate_v9j()
        v9r.v9k._annotate_v9k()
        v9r._annotate_v9r()
        v9s._annotate_v9s()
        v9t._annotate_v9t()
        _annotate_v9v()


if __name__ == "__main__":
    main()
