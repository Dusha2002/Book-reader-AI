from __future__ import annotations

import json
import os
from typing import Any

import chapter_reference_translation_v9am_numeric_fast as v9am
from bookai.address_fidelity import normalize_numbered_address_literals
from bookai.numeric_fidelity import compare_numeric_fidelity
from bookai.semantic_fidelity import (
    compare_material_fidelity,
    compare_order_fidelity,
    compare_question_fidelity,
    compare_short_omission_fidelity,
)


_BASE_SEMANTIC_ROUTE = v9am._BASE_ROUTE
_GIGA_STATS: dict[str, Any] = {
    "pre_passes": 0,
    "pre_detected": 0,
    "pre_repaired": 0,
    "pre_residual": 0,
    "final_detected": 0,
    "final_repaired": 0,
    "final_residual": 0,
    "gigachat_calls": 0,
    "deepseek_simple_repairs": 0,
    "address_replacements": 0,
}


def _normalize_addresses(targets, translated) -> list[str]:
    changed: list[str] = []
    replacements = 0
    for segment in targets:
        sid = str(segment.id)
        current = str(translated.get(sid) or "")
        if not current:
            continue
        fixed, count = normalize_numbered_address_literals(str(segment.text or ""), current)
        if count > 0 and fixed != current:
            translated[sid] = fixed
            changed.append(sid)
            replacements += int(count)
    _GIGA_STATS["address_replacements"] = int(_GIGA_STATS.get("address_replacements") or 0) + replacements
    return changed


def _latin_codes_for_pair(segment, target: str) -> list[str]:
    rows = v9am.v9al.v9ah.v9af._objective_issues(
        [segment], {str(segment.id): str(target or "")}
    )
    codes: list[str] = []
    for row in rows:
        for code in row.get("codes", []):
            code = str(code)
            if code.startswith("latin") and code not in codes:
                codes.append(code)
    return codes


def _simple_failures_for_pair(segment, target: str) -> list[str]:
    source = str(segment.text or "")
    target = str(target or "")
    codes: list[str] = []

    numeric = compare_numeric_fidelity(source, target)
    if not numeric.get("ok", True):
        codes.append("numeric")

    questions = compare_question_fidelity(source, target)
    if not questions.get("ok", True):
        codes.append("question")

    materials = compare_material_fidelity(source, target)
    if not materials.get("ok", True):
        codes.append("material")

    order = compare_order_fidelity(source, target)
    if not order.get("ok", True):
        codes.append("order")

    short = compare_short_omission_fidelity(source, target)
    if not short.get("ok", True):
        codes.append("short_omission")

    if _latin_codes_for_pair(segment, target):
        codes.append("latin_leak")

    return codes


def _detect_simple_failures(targets, translated) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, segment in enumerate(targets):
        sid = str(segment.id)
        current = str(translated.get(sid) or "")
        if not current:
            continue
        codes = _simple_failures_for_pair(segment, current)
        if not codes:
            continue
        rows.append(
            {
                "id": sid,
                "index": i,
                "codes": codes,
                "source": str(segment.text or ""),
                "current_ru": current,
                "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
                "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
            }
        )
    return rows


def _giga_repair(targets, translated, memory, issues, *, strict: bool = False) -> tuple[list[str], int]:
    if not issues:
        return [], 0

    batch_size = max(8, int(os.getenv("BOOKAI_V9AO_GIGA_BATCH") or "18"))
    cap = max(16, int(os.getenv("BOOKAI_V9AO_GIGA_MAX") or "48"))
    issues = list(issues)[:cap]
    by_id = {str(segment.id): segment for segment in targets}

    system = """You are a fast EN→RU publication repairer. These rows have SIMPLE, PROVEN defects.
Do not perform a broad literary rewrite. Return a complete Russian translation of SOURCE, fixing the listed checks only.

Checks:
- numeric: preserve every source number/quantity/address value exactly.
- latin_leak: remove accidental English/Latin prose; render names in established Cyrillic. For rhyme/wordplay, recreate it naturally in Russian Cyrillic.
- short_omission: restore every omitted sentence/dialogue beat, attribution, action, adjective and object.
- question: preserve every explicit source question.
- material: preserve the physical material fact (steel/iron/copper/brass/bronze/aluminium) together with its object.
- order: preserve front/back/beginning/end ordering. Never reverse `at the front` into `напоследок` or the opposite.

Always preserve actors, actions, objects, negation, chronology, causality and established names. BEFORE_EN/AFTER_EN are context only.
ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}; exactly one item per input id.
"""
    if strict:
        system += "\nSTRICT RETRY: the previous candidate still failed deterministic validation. Do not leave any listed defect unresolved."

    changed: list[str] = []
    calls = 0
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        payload = [
            {
                "id": row["id"],
                "failed_checks": row["codes"],
                "source": row["source"],
                "current_ru": row["current_ru"],
                "before_en": row["before_en"],
                "after_en": row["after_en"],
            }
            for row in batch
        ]
        try:
            obj = v9am.v9al.v9ah.v9af.v9ab._giga_json(system, {"items": payload}, max_tokens=6200)
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            calls += 1
            print(f"[v9ao-giga-repair] strict={strict} error={type(exc).__name__}", flush=True)
            continue

        parsed = {
            str(row.get("id") or ""): row
            for row in raw
            if isinstance(row, dict) and str(row.get("id") or "")
        }
        for row in batch:
            sid = row["id"]
            item = parsed.get(sid) or {}
            candidate = v9am.v9al.v9ah.v9._norm_text(item.get("corrected_ru") or "")
            if not candidate:
                continue
            segment = by_id[sid]
            old = str(translated.get(sid) or "")
            try:
                candidate = v9am.v9al.v9ah.v9ag.v9ad._canonicalize_candidate(segment, candidate, memory)
            except Exception:
                pass
            if not candidate:
                continue
            if _simple_failures_for_pair(segment, candidate):
                continue
            if v9am.v9al.v9ah.v9._fatal_count(segment, candidate, memory) > v9am.v9al.v9ah.v9._fatal_count(segment, old, memory):
                continue
            translated[sid] = candidate
            changed.append(sid)

    return changed, calls


def _giga_simple_pass(targets, translated, memory, *, stage: str) -> dict[str, Any]:
    _normalize_addresses(targets, translated)
    initial = _detect_simple_failures(targets, translated)
    changed1, calls1 = _giga_repair(targets, translated, memory, initial, strict=False)
    _normalize_addresses(targets, translated)
    residual1 = _detect_simple_failures(targets, translated)
    changed2, calls2 = _giga_repair(targets, translated, memory, residual1, strict=True)
    _normalize_addresses(targets, translated)
    residual2 = _detect_simple_failures(targets, translated)

    _GIGA_STATS["gigachat_calls"] = int(_GIGA_STATS.get("gigachat_calls") or 0) + calls1 + calls2
    _GIGA_STATS[f"{stage}_detected"] = len(initial)
    _GIGA_STATS[f"{stage}_repaired"] = len(set(changed1) | set(changed2))
    _GIGA_STATS[f"{stage}_residual"] = len(residual2)
    _GIGA_STATS[f"{stage}_initial_ids"] = [row["id"] for row in initial]
    _GIGA_STATS[f"{stage}_residual_ids"] = [row["id"] for row in residual2]
    _GIGA_STATS[f"{stage}_codes"] = {
        row["id"]: row["codes"] for row in initial
    }
    result = {
        "stage": stage,
        "detected": len(initial),
        "repaired": len(set(changed1) | set(changed2)),
        "gigachat_calls": calls1 + calls2,
        "residual": len(residual2),
        "residual_ids": [row["id"] for row in residual2],
    }
    print("[v9ao-giga-first] " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _giga_first_route(targets, translated, memory):
    """Repair simple objective defects with GigaChat before DeepSeek routing.

    DeepSeek receives only v9ah's original semantic-risk router output; v9am's
    numeric/question/material contract promotion is intentionally bypassed.
    """
    _GIGA_STATS["pre_passes"] = int(_GIGA_STATS.get("pre_passes") or 0) + 1
    _giga_simple_pass(targets, translated, memory, stage="pre")
    selected, ranked = _BASE_SEMANTIC_ROUTE(targets, translated, memory)
    _GIGA_STATS["deepseek_semantic_routes"] = len(selected)
    _GIGA_STATS["deepseek_semantic_route_ids"] = [str(row.get("id") or "") for row in selected]
    return selected, ranked


def _giga_final_cleanup(harness, targets, translated, memory) -> dict[str, Any]:
    """Replace v9al's final DeepSeek evidence pass with GigaChat + proof."""
    result = _giga_simple_pass(targets, translated, memory, stage="final")
    return {
        "candidates": result["detected"],
        "candidate_ids": list(_GIGA_STATS.get("final_initial_ids") or []),
        "deepseek_calls": 0,
        "gigachat_calls": result["gigachat_calls"],
        "changed": result["repaired"],
        "accepted_ids": sorted(set(_GIGA_STATS.get("final_initial_ids") or []) - set(result["residual_ids"])),
        "residual_candidates": result["residual"],
        "residual_ids": result["residual_ids"],
        "repair_model": "GigaChat-3-Lightning",
    }


def _annotate_report() -> None:
    v3 = v9am.v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    architecture = dict(data.get("architecture") or {})
    architecture.update(
        {
            "experiment": "v9ao-gigachat-first-simple-repairs",
            "simple_repair_model": "GigaChat-3-Lightning",
            "simple_repair_classes": ["numeric", "latin_leak", "short_omission", "question", "material", "order"],
            "simple_repair_acceptance": "deterministic contract validation + no fatal regression",
            "deepseek_role": "v9ah semantic-risk specialist only; no simple publication repair/evidence pass",
            "final_evidence_model": "GigaChat-3-Lightning",
            "gigachat_ultra_used": False,
        }
    )
    data["architecture"] = architecture
    data["v9ao_giga_first"] = dict(_GIGA_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_route = v9am._contract_route_v9am
    old_evidence = v9am.v9al._evidence_deepseek_cleanup
    v9am._contract_route_v9am = _giga_first_route
    v9am.v9al._evidence_deepseek_cleanup = _giga_final_cleanup
    try:
        v9am.main()
    finally:
        v9am._contract_route_v9am = old_route
        v9am.v9al._evidence_deepseek_cleanup = old_evidence
        _annotate_report()


if __name__ == "__main__":
    main()
