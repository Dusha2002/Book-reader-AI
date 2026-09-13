from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9al_fast_deepseek as v9al
from bookai.numeric_fidelity import compare_numeric_fidelity
from bookai.semantic_fidelity import (
    compare_material_fidelity,
    compare_question_fidelity,
    lexicalized_technical_compound_preserved,
)


_BASE_HARD_ROWS = v9al._hard_issue_rows
_BASE_ISSUE_SCORE = v9al._issue_score
_BASE_ROUTE = v9al.v9ah._adaptive_route_v9ah
_BASE_DEEP_REPAIR = v9al.v9ah._BASE_DEEP_REPAIR
_NUMERIC_STATS: dict[str, Any] = {}
_ROUTER_STATS: dict[str, Any] = {}


def _contract_checks(source: str, target: str) -> dict[str, dict[str, Any]]:
    return {
        "numeric": compare_numeric_fidelity(source, target),
        "question": compare_question_fidelity(source, target),
        "material": compare_material_fidelity(source, target),
    }


def _contract_failures(source: str, target: str) -> list[tuple[str, dict[str, Any]]]:
    checks = _contract_checks(source, target)
    return [(name, check) for name, check in checks.items() if not check.get("ok", True)]


def _contract_reason(name: str, check: dict[str, Any]) -> str:
    if name == "numeric":
        return (
            "DETERMINISTIC numeric fidelity failure: source numeric values="
            + json.dumps(check.get("source_values") or [], ensure_ascii=False)
            + ", current_ru numeric values="
            + json.dumps(check.get("target_values") or [], ensure_ascii=False)
            + ", missing="
            + json.dumps(check.get("missing") or [], ensure_ascii=False)
            + ". Corrected Russian MUST preserve every missing source value exactly."
        )
    if name == "question":
        return (
            f"DETERMINISTIC question fidelity failure: source has {int(check.get('source_questions') or 0)} explicit questions, "
            f"current_ru has {int(check.get('target_questions') or 0)}. Corrected Russian MUST preserve every source interrogative."
        )
    return (
        "DETERMINISTIC material fidelity failure: physical material(s) missing from current_ru="
        + json.dumps(check.get("missing") or [], ensure_ascii=False)
        + ". Preserve the material fact together with its physical object; do not literalize unrelated lexicalized technical compounds."
    )


def _contract_route_v9am(targets, translated, memory):
    """Inject proven objective failures into v9ah's FIRST specialist repair.

    This does not broaden semantic speculation. Only deterministic numeric,
    explicit-question and physical-material failures are promoted. Priority 25
    makes them mandatory under v9ah's existing >=22 budget rule.
    """
    selected, ranked = _BASE_ROUTE(targets, translated, memory)
    index = {str(segment.id): i for i, segment in enumerate(targets)}
    by_id = {str(segment.id): segment for segment in targets}
    ranked_by_id = {str(row.get("id") or ""): dict(row) for row in ranked}
    selected_ids = {str(row.get("id") or "") for row in selected}
    contract_ids: list[str] = []
    contract_codes: dict[str, list[str]] = {}

    for sid, segment in by_id.items():
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")
        failures = _contract_failures(source, current)
        if not failures:
            continue
        contract_ids.append(sid)
        contract_codes[sid] = [name for name, _ in failures]
        reasons = [_contract_reason(name, check) for name, check in failures]
        row = ranked_by_id.get(sid) or {
            "id": sid,
            "index": index[sid],
            "priority": 25,
            "code": "contract_fidelity",
            "reason": "",
            "glossary_hits": [],
        }
        row = dict(row)
        row["priority"] = max(25, int(row.get("priority") or 0))
        # Preserve an existing specialist code when one already gives DeepSeek a
        # useful semantic lens; otherwise make the deterministic contract explicit.
        if not str(row.get("code") or "").strip():
            row["code"] = "contract_fidelity"
        prior = str(row.get("reason") or "").strip()
        row["reason"] = "; ".join([x for x in [prior, *reasons] if x])
        row.setdefault("glossary_hits", [])
        ranked_by_id[sid] = row

    if not contract_ids:
        _ROUTER_STATS.clear()
        _ROUTER_STATS.update({
            "pre_repair_contract_routes": 0,
            "pre_repair_contract_ids": [],
            "pre_repair_contract_codes": {},
        })
        return selected, ranked

    # Rebuild ranked with the enhanced mandatory rows, then guarantee every
    # contract route is present in selected regardless of v9ah's soft budget.
    ranked_out: list[dict[str, Any]] = []
    seen_ranked: set[str] = set()
    for raw in ranked:
        sid = str(raw.get("id") or "")
        if not sid or sid in seen_ranked:
            continue
        ranked_out.append(dict(ranked_by_id.get(sid) or raw))
        seen_ranked.add(sid)
    for sid in contract_ids:
        if sid not in seen_ranked:
            ranked_out.append(dict(ranked_by_id[sid]))
            seen_ranked.add(sid)
    ranked_out.sort(key=lambda row: (-int(row.get("priority") or 0), int(row.get("index") or 0)))

    selected_out: list[dict[str, Any]] = []
    seen_selected: set[str] = set()
    for raw in selected:
        sid = str(raw.get("id") or "")
        if not sid or sid in seen_selected:
            continue
        selected_out.append(dict(ranked_by_id.get(sid) or raw))
        seen_selected.add(sid)
    for sid in contract_ids:
        if sid not in seen_selected:
            selected_out.append(dict(ranked_by_id[sid]))
            seen_selected.add(sid)
    selected_out.sort(key=lambda row: (-int(row.get("priority") or 0), int(row.get("index") or 0)))

    _ROUTER_STATS.clear()
    _ROUTER_STATS.update({
        "pre_repair_contract_routes": len(contract_ids),
        "pre_repair_contract_ids": contract_ids,
        "pre_repair_contract_codes": contract_codes,
        "selected_before_contracts": len(selected),
        "selected_after_contracts": len(selected_out),
        "contract_routes_added_to_selected": len([sid for sid in contract_ids if sid not in selected_ids]),
    })
    print("[v9am-contract-router] " + json.dumps(_ROUTER_STATS, ensure_ascii=False), flush=True)
    return selected_out, ranked_out


def _contract_validated_deep_repair(harness, targets, translated, memory, routes):
    """Run the existing v9ad repair, but reject objective-contract regressions."""
    before = {str(row.get("id") or ""): str(translated.get(str(row.get("id") or "")) or "") for row in routes}
    changed, calls = _BASE_DEEP_REPAIR(harness, targets, translated, memory, routes)
    by_id = {str(segment.id): segment for segment in targets}
    accepted: list[str] = []
    rejected: list[str] = []

    for sid in changed:
        segment = by_id.get(str(sid))
        if segment is None:
            continue
        candidate = str(translated.get(str(sid)) or "")
        if _contract_failures(str(segment.text or ""), candidate):
            translated[str(sid)] = before.get(str(sid), "")
            rejected.append(str(sid))
        else:
            accepted.append(str(sid))

    if rejected:
        print(
            "[v9am-contract-repair-reject] "
            + json.dumps({"rejected_ids": rejected, "accepted_ids": accepted}, ensure_ascii=False),
            flush=True,
        )
    _ROUTER_STATS["first_repair_contract_rejected_ids"] = rejected
    _ROUTER_STATS["first_repair_changed_after_contract_validation"] = len(accepted)
    return accepted, calls


def _row_priority(row: dict[str, Any]) -> tuple[int, int]:
    codes = [str(code) for code in row.get("codes", [])]
    if any(code.startswith("spelled_number_fidelity:") for code in codes):
        priority = 0
    elif any(code.startswith("question_fidelity:") for code in codes):
        priority = 1
    elif any(code.startswith("material_fidelity:") for code in codes):
        priority = 2
    elif "short_omission" in codes:
        priority = 3
    elif "latin_leak" in codes or "latin_residue" in codes:
        priority = 4
    elif "v9_qe_unresolved" in codes:
        priority = 5
    elif "numbered_entity_exactness" in codes:
        priority = 6
    else:
        priority = 7
    return priority, int(row.get("index") or 0)


def _contract_hard_rows(targets, translated, memory) -> list[dict[str, Any]]:
    raw_rows = [dict(row) for row in _BASE_HARD_ROWS(targets, translated, memory)]
    numeric_checks: dict[str, dict[str, Any]] = {}
    target_by_id = {str(segment.id): segment for segment in targets}
    heuristic_dropped: list[str] = []
    lexicalized_qe_dropped: list[str] = []
    rows: list[dict[str, Any]] = []

    for row in raw_rows:
        sid = str(row.get("id") or "")
        segment = target_by_id.get(sid)
        if segment is None:
            continue
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")
        check = compare_numeric_fidelity(source, current)
        numeric_checks[sid] = check
        codes = [str(code) for code in row.get("codes", [])]
        if "numbered_entity_exactness" in codes and check.get("ok", True):
            codes = [code for code in codes if code != "numbered_entity_exactness"]
            heuristic_dropped.append(sid)
        if (
            "v9_qe_unresolved" in codes
            and lexicalized_technical_compound_preserved(source, current)
            and all(code in {"v9_qe_unresolved", "numbered_entity_exactness"} for code in codes)
        ):
            codes = [code for code in codes if code != "v9_qe_unresolved"]
            lexicalized_qe_dropped.append(sid)
        if not codes:
            continue
        row["codes"] = codes
        rows.append(row)

    by_id = {str(row.get("id") or ""): row for row in rows}
    numeric_mismatches: list[dict[str, Any]] = []
    question_mismatches: list[dict[str, Any]] = []
    material_mismatches: list[dict[str, Any]] = []

    def add_contract_row(i: int, segment, code: str) -> None:
        sid = str(segment.id)
        current_row = by_id.get(sid)
        if current_row is not None:
            codes = list(current_row.get("codes") or [])
            if code not in codes:
                codes.append(code)
            current_row["codes"] = codes
            return
        row = {
            "id": sid,
            "index": i,
            "codes": [code],
            "source": str(segment.text or ""),
            "current_ru": str(translated.get(sid) or ""),
            "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
            "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
        }
        rows.append(row)
        by_id[sid] = row

    for i, segment in enumerate(targets):
        sid = str(segment.id)
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")

        numeric = numeric_checks.get(sid)
        if numeric is None:
            numeric = compare_numeric_fidelity(source, current)
            numeric_checks[sid] = numeric
        if not numeric.get("ok", True):
            detail = {
                "source_values": numeric.get("source_values") or [],
                "target_values": numeric.get("target_values") or [],
                "missing": numeric.get("missing") or [],
            }
            numeric_mismatches.append({"id": sid, **detail})
            add_contract_row(
                i,
                segment,
                "spelled_number_fidelity:" + json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
            )

        questions = compare_question_fidelity(source, current)
        if not questions.get("ok", True):
            detail = {
                "source_questions": questions.get("source_questions", 0),
                "target_questions": questions.get("target_questions", 0),
                "missing_questions": questions.get("missing_questions", 0),
            }
            question_mismatches.append({"id": sid, **detail})
            add_contract_row(
                i,
                segment,
                "question_fidelity:" + json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
            )

        materials = compare_material_fidelity(source, current)
        if not materials.get("ok", True):
            detail = {"required": materials.get("required") or [], "missing": materials.get("missing") or []}
            material_mismatches.append({"id": sid, **detail})
            add_contract_row(
                i,
                segment,
                "material_fidelity:" + json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
            )

    rows.sort(key=_row_priority)
    soft_cap = max(4, int(__import__("os").getenv("BOOKAI_FAST_EVIDENCE_MAX") or "16"))
    mandatory_ids = {row["id"] for row in [*numeric_mismatches, *question_mismatches, *material_mismatches]}
    effective_cap = max(soft_cap, len(mandatory_ids))
    selected = rows[:effective_cap]
    selected_ids = {str(row.get("id") or "") for row in selected}
    dropped_mandatory = sorted(mandatory_ids - selected_ids)

    _NUMERIC_STATS.clear()
    _NUMERIC_STATS.update({
        "mismatch_count": len(numeric_mismatches),
        "mismatch_ids": [row["id"] for row in numeric_mismatches],
        "details": numeric_mismatches[:32],
        "question_mismatch_count": len(question_mismatches),
        "question_mismatch_ids": [row["id"] for row in question_mismatches],
        "question_details": question_mismatches[:32],
        "material_mismatch_count": len(material_mismatches),
        "material_mismatch_ids": [row["id"] for row in material_mismatches],
        "material_details": material_mismatches[:32],
        "heuristic_number_rows_dropped": heuristic_dropped,
        "lexicalized_qe_rows_dropped": lexicalized_qe_dropped,
        "soft_cap": soft_cap,
        "effective_cap": effective_cap,
        "evidence_rows_after_cap": len(selected),
        "selected_ids": [str(row.get("id") or "") for row in selected],
        "dropped_numeric_ids": [row["id"] for row in numeric_mismatches if row["id"] not in selected_ids],
        "dropped_mandatory_ids": dropped_mandatory,
    })
    if numeric_mismatches or question_mismatches or material_mismatches or heuristic_dropped or lexicalized_qe_dropped:
        print("[v9am-fidelity-contracts] " + json.dumps(_NUMERIC_STATS, ensure_ascii=False), flush=True)
    if dropped_mandatory:
        raise RuntimeError("publication fidelity obligations were dropped from evidence queue")
    return selected


def _contract_issue_score(targets, translated, memory, sid: str) -> tuple[int, list[str]]:
    base_score, codes = _BASE_ISSUE_SCORE(targets, translated, memory, sid)
    segment = next(segment for segment in targets if str(segment.id) == str(sid))
    source = str(segment.text or "")
    current = str(translated.get(sid) or "")

    numeric = compare_numeric_fidelity(source, current)
    if not numeric.get("ok", True):
        detail = "spelled_number_fidelity:" + json.dumps({
            "source_values": numeric.get("source_values") or [],
            "target_values": numeric.get("target_values") or [],
            "missing": numeric.get("missing") or [],
        }, ensure_ascii=False, separators=(",", ":"))
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    questions = compare_question_fidelity(source, current)
    if not questions.get("ok", True):
        detail = "question_fidelity:" + json.dumps({
            "source_questions": questions.get("source_questions", 0),
            "target_questions": questions.get("target_questions", 0),
            "missing_questions": questions.get("missing_questions", 0),
        }, ensure_ascii=False, separators=(",", ":"))
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    materials = compare_material_fidelity(source, current)
    if not materials.get("ok", True):
        detail = "material_fidelity:" + json.dumps({
            "required": materials.get("required") or [],
            "missing": materials.get("missing") or [],
        }, ensure_ascii=False, separators=(",", ":"))
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    return base_score, codes


def _filter_resolved_stale_qe(data: dict[str, Any]) -> int:
    v3 = v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    mapping_path = getattr(v3, "MAP_JSON", None)
    if mapping_path is None or not mapping_path.exists():
        return 0
    try:
        mapping = json.loads(mapping_path.read_text("utf-8"))
    except Exception:
        return 0
    by_id = {
        str(row.get("id") or ""): row
        for row in mapping
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    kept = []
    removed = 0
    for row in list(data.get("v6_remaining_hard") or []):
        if not isinstance(row, dict):
            kept.append(row)
            continue
        sid = str(row.get("id") or "")
        reason = str(row.get("reason") or "")
        pair = by_id.get(sid) or {}
        source = str(pair.get("source") or "")
        target = str(pair.get("translation") or "")
        material = compare_material_fidelity(source, target)
        resolved_material = bool(material.get("required")) and material.get("ok", True)
        lexicalized = lexicalized_technical_compound_preserved(source, target)
        if "material concept" in reason.casefold() and (resolved_material or lexicalized):
            removed += 1
            continue
        kept.append(row)
    if not removed:
        return 0
    data["v6_remaining_hard"] = kept
    counts = dict(data.get("post_export_issue_counts") or {})
    key = "hard:v9_qe_unresolved"
    if key in counts:
        value = max(0, int(counts.get(key) or 0) - removed)
        if value:
            counts[key] = value
        else:
            counts.pop(key, None)
        data["post_export_issue_counts"] = counts
    final_quality = dict(data.get("final_quality") or {})
    if final_quality:
        final_quality["hard_issues"] = sum(
            int(value or 0) for name, value in counts.items() if str(name).startswith("hard:")
        )
        data["final_quality"] = final_quality
    return removed


def _annotate_numeric_report() -> None:
    v3 = v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    stale_removed = _filter_resolved_stale_qe(data)
    architecture = dict(data.get("architecture") or {})
    architecture.update({
        "experiment": "v9am-fast-deepseek-fidelity-contracts",
        "numeric_fidelity": "span-aware deterministic EN/RU numeric parser; objective mismatches outrank the soft evidence budget",
        "question_fidelity": "explicit source question count is a deterministic publication obligation",
        "material_fidelity": (
            "physical material+noun facts are locked; lexicalized technical compounds such as lead-screw -> ходовой винт "
            "are recognized rather than forced into literal token matching"
        ),
        "pre_repair_contract_routing": "numeric/question/material failures are mandatory priority-25 routes in the first DeepSeek repair",
        "gigachat_ultra_used": False,
    })
    data["architecture"] = architecture
    stats = dict(_NUMERIC_STATS)
    stats["stale_material_qe_rows_removed"] = stale_removed
    stats["pre_repair_router"] = dict(_ROUTER_STATS)
    data["v9am_numeric_fidelity"] = stats
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_rows = v9al._hard_issue_rows
    old_score = v9al._issue_score
    old_route = v9al.v9ah._adaptive_route_v9ah
    old_deep_repair = v9al.v9ah._BASE_DEEP_REPAIR
    v9al._hard_issue_rows = _contract_hard_rows
    v9al._issue_score = _contract_issue_score
    v9al.v9ah._adaptive_route_v9ah = _contract_route_v9am
    v9al.v9ah._BASE_DEEP_REPAIR = _contract_validated_deep_repair
    try:
        v9al.main()
    finally:
        v9al._hard_issue_rows = old_rows
        v9al._issue_score = old_score
        v9al.v9ah._adaptive_route_v9ah = old_route
        v9al.v9ah._BASE_DEEP_REPAIR = old_deep_repair
        _annotate_numeric_report()


if __name__ == "__main__":
    main()
