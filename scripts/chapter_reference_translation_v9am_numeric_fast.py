from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9al_fast_deepseek as v9al
from bookai.numeric_fidelity import compare_numeric_fidelity


_BASE_HARD_ROWS = v9al._hard_issue_rows
_BASE_ISSUE_SCORE = v9al._issue_score
_NUMERIC_STATS: dict[str, Any] = {}


def _row_priority(row: dict[str, Any]) -> tuple[int, int]:
    codes = [str(code) for code in row.get("codes", [])]
    if any(code.startswith("spelled_number_fidelity:") for code in codes):
        priority = 0
    elif "short_omission" in codes:
        priority = 1
    elif "latin_leak" in codes or "latin_residue" in codes:
        priority = 2
    elif "v9_qe_unresolved" in codes:
        priority = 3
    elif "numbered_entity_exactness" in codes:
        priority = 4
    else:
        priority = 5
    return priority, int(row.get("index") or 0)


def _numeric_hard_rows(targets, translated, memory) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _BASE_HARD_ROWS(targets, translated, memory)]
    by_id = {str(row.get("id") or ""): row for row in rows}
    mismatches: list[dict[str, Any]] = []

    for i, segment in enumerate(targets):
        sid = str(segment.id)
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")
        check = compare_numeric_fidelity(source, current)
        if check.get("ok", True):
            continue

        detail = {
            "source_values": check.get("source_values") or [],
            "target_values": check.get("target_values") or [],
            "missing": check.get("missing") or [],
        }
        mismatches.append({"id": sid, **detail})
        code = "spelled_number_fidelity:" + json.dumps(detail, ensure_ascii=False, separators=(",", ":"))

        current_row = by_id.get(sid)
        if current_row is not None:
            codes = list(current_row.get("codes") or [])
            if code not in codes:
                codes.append(code)
            current_row["codes"] = codes
            continue

        row = {
            "id": sid,
            "index": i,
            "codes": [code],
            "source": source,
            "current_ru": current,
            "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
            "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
        }
        rows.append(row)
        by_id[sid] = row

    # Objective numeric mismatches are publication obligations, not a soft cost
    # heuristic. Put them ahead of generic evidence and allow them to exceed the
    # ordinary evidence cap if necessary. This prevents a late-book 30→35 error
    # from being silently dropped because 16 earlier heuristic rows filled budget.
    rows.sort(key=_row_priority)
    soft_cap = max(4, int(__import__("os").getenv("BOOKAI_FAST_EVIDENCE_MAX") or "16"))
    effective_cap = max(soft_cap, len(mismatches))
    selected = rows[:effective_cap]
    selected_ids = {str(row.get("id") or "") for row in selected}
    dropped_numeric = [row["id"] for row in mismatches if row["id"] not in selected_ids]

    _NUMERIC_STATS.clear()
    _NUMERIC_STATS.update(
        {
            "mismatch_count": len(mismatches),
            "mismatch_ids": [row["id"] for row in mismatches],
            "details": mismatches[:32],
            "soft_cap": soft_cap,
            "effective_cap": effective_cap,
            "evidence_rows_after_cap": len(selected),
            "selected_ids": [str(row.get("id") or "") for row in selected],
            "dropped_numeric_ids": dropped_numeric,
        }
    )
    if mismatches:
        print("[v9am-numeric-fidelity] " + json.dumps(_NUMERIC_STATS, ensure_ascii=False), flush=True)
    if dropped_numeric:
        raise RuntimeError("numeric publication obligations were dropped from evidence queue")
    return selected


def _numeric_issue_score(targets, translated, memory, sid: str) -> tuple[int, list[str]]:
    score, codes = _BASE_ISSUE_SCORE(targets, translated, memory, sid)
    segment = next(segment for segment in targets if str(segment.id) == str(sid))
    check = compare_numeric_fidelity(
        str(segment.text or ""),
        str(translated.get(sid) or ""),
    )
    if not check.get("ok", True):
        detail = "spelled_number_fidelity:" + json.dumps(
            {
                "source_values": check.get("source_values") or [],
                "target_values": check.get("target_values") or [],
                "missing": check.get("missing") or [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if detail not in codes:
            codes.append(detail)
        # Bigger than any single generic hard finding, so a candidate cannot be
        # accepted merely by polishing style while leaving the number wrong.
        score += 12
    return score, codes


def _annotate_numeric_report() -> None:
    v3 = v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
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
            "experiment": "v9am-fast-deepseek-numeric-contract",
            "numeric_fidelity": (
                "span-aware deterministic EN/RU numeric parser; generic lexical ordinals are excluded; "
                "objective quantity/numbered-entity mismatches outrank the soft evidence budget and "
                "a repair is rejected until the numeric fact is preserved"
            ),
            "gigachat_ultra_used": False,
        }
    )
    data["architecture"] = architecture
    data["v9am_numeric_fidelity"] = dict(_NUMERIC_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_rows = v9al._hard_issue_rows
    old_score = v9al._issue_score
    v9al._hard_issue_rows = _numeric_hard_rows
    v9al._issue_score = _numeric_issue_score
    try:
        v9al.main()
    finally:
        v9al._hard_issue_rows = old_rows
        v9al._issue_score = old_score
        _annotate_numeric_report()


if __name__ == "__main__":
    main()
