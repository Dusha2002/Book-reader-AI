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
_NUMERIC_STATS: dict[str, Any] = {}


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

    # v9al predates the deterministic numeric contract and conservatively routed
    # many numbered-looking rows even when their value was already correct. Once
    # we can prove the value is preserved, that heuristic no longer buys quality.
    # Also suppress stale QE complaints for lexicalized technical compounds whose
    # correct Russian term is demonstrably non-literal (lead-screw -> ходовой винт).
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
                "spelled_number_fidelity:" + json.dumps(
                    detail, ensure_ascii=False, separators=(",", ":")
                ),
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
                "question_fidelity:" + json.dumps(
                    detail, ensure_ascii=False, separators=(",", ":")
                ),
            )

        materials = compare_material_fidelity(source, current)
        if not materials.get("ok", True):
            detail = {
                "required": materials.get("required") or [],
                "missing": materials.get("missing") or [],
            }
            material_mismatches.append({"id": sid, **detail})
            add_contract_row(
                i,
                segment,
                "material_fidelity:" + json.dumps(
                    detail, ensure_ascii=False, separators=(",", ":")
                ),
            )

    # Deterministic contract mismatches are publication obligations, not a soft
    # cost heuristic. Put them ahead of generic evidence and allow them to exceed
    # the ordinary evidence cap if necessary.
    rows.sort(key=_row_priority)
    soft_cap = max(4, int(__import__("os").getenv("BOOKAI_FAST_EVIDENCE_MAX") or "16"))
    mandatory_ids = {
        row["id"]
        for row in [*numeric_mismatches, *question_mismatches, *material_mismatches]
    }
    effective_cap = max(soft_cap, len(mandatory_ids))
    selected = rows[:effective_cap]
    selected_ids = {str(row.get("id") or "") for row in selected}
    dropped_mandatory = sorted(mandatory_ids - selected_ids)

    _NUMERIC_STATS.clear()
    _NUMERIC_STATS.update(
        {
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
        }
    )
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
        detail = "spelled_number_fidelity:" + json.dumps(
            {
                "source_values": numeric.get("source_values") or [],
                "target_values": numeric.get("target_values") or [],
                "missing": numeric.get("missing") or [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    questions = compare_question_fidelity(source, current)
    if not questions.get("ok", True):
        detail = "question_fidelity:" + json.dumps(
            {
                "source_questions": questions.get("source_questions", 0),
                "target_questions": questions.get("target_questions", 0),
                "missing_questions": questions.get("missing_questions", 0),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    materials = compare_material_fidelity(source, current)
    if not materials.get("ok", True):
        detail = "material_fidelity:" + json.dumps(
            {
                "required": materials.get("required") or [],
                "missing": materials.get("missing") or [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if detail not in codes:
            codes.append(detail)
        return 1_000_000, codes

    return base_score, codes


def _filter_resolved_stale_qe(data: dict[str, Any]) -> int:
    """Remove stale material-QE rows only when final text proves them resolved."""
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
            int(value or 0)
            for name, value in counts.items()
            if str(name).startswith("hard:")
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
    architecture.update(
        {
            "experiment": "v9am-fast-deepseek-fidelity-contracts",
            "numeric_fidelity": (
                "span-aware deterministic EN/RU numeric parser; objective mismatches outrank the soft evidence budget"
            ),
            "question_fidelity": "explicit source question count is a deterministic publication obligation",
            "material_fidelity": (
                "physical material+noun facts are locked; lexicalized technical compounds such as lead-screw -> ходовой винт "
                "are recognized rather than forced into literal token matching"
            ),
            "gigachat_ultra_used": False,
        }
    )
    data["architecture"] = architecture
    stats = dict(_NUMERIC_STATS)
    stats["stale_material_qe_rows_removed"] = stale_removed
    data["v9am_numeric_fidelity"] = stats
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_rows = v9al._hard_issue_rows
    old_score = v9al._issue_score
    v9al._hard_issue_rows = _contract_hard_rows
    v9al._issue_score = _contract_issue_score
    try:
        v9al.main()
    finally:
        v9al._hard_issue_rows = old_rows
        v9al._issue_score = old_score
        _annotate_numeric_report()


if __name__ == "__main__":
    main()
