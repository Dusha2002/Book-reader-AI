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
    raw_rows = [dict(row) for row in _BASE_HARD_ROWS(targets, translated, memory)]
    checks: dict[str, dict[str, Any]] = {}
    target_by_id = {str(segment.id): segment for segment in targets}
    heuristic_dropped: list[str] = []
    rows: list[dict[str, Any]] = []

    # v9al predates the deterministic numeric contract and conservatively routed
    # many numbered-looking rows even when their value was already correct. Once
    # we can prove the value is preserved, that heuristic no longer buys quality
    # and only bloats the DeepSeek evidence batch.
    for row in raw_rows:
        sid = str(row.get("id") or "")
        segment = target_by_id.get(sid)
        if segment is None:
            continue
        check = compare_numeric_fidelity(
            str(segment.text or ""), str(translated.get(sid) or "")
        )
        checks[sid] = check
        codes = [str(code) for code in row.get("codes", [])]
        if "numbered_entity_exactness" in codes and check.get("ok", True):
            codes = [code for code in codes if code != "numbered_entity_exactness"]
            heuristic_dropped.append(sid)
        if not codes:
            continue
        row["codes"] = codes
        rows.append(row)

    by_id = {str(row.get("id") or ""): row for row in rows}
    mismatches: list[dict[str, Any]] = []

    for i, segment in enumerate(targets):
        sid = str(segment.id)
        source = str(segment.text or "")
        current = str(translated.get(sid) or "")
        check = checks.get(sid)
        if check is None:
            check = compare_numeric_fidelity(source, current)
            checks[sid] = check
        if check.get("ok", True):
            continue

        detail = {
            "source_values": check.get("source_values") or [],
            "target_values": check.get("target_values") or [],
            "missing": check.get("missing") or [],
        }
        mismatches.append({"id": sid, **detail})
        code = "spelled_number_fidelity:" + json.dumps(
            detail, ensure_ascii=False, separators=(",", ":")
        )

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
    # ordinary evidence cap if necessary.
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
            "heuristic_number_rows_dropped": heuristic_dropped,
            "soft_cap": soft_cap,
            "effective_cap": effective_cap,
            "evidence_rows_after_cap": len(selected),
            "selected_ids": [str(row.get("id") or "") for row in selected],
            "dropped_numeric_ids": dropped_numeric,
        }
    )
    if mismatches or heuristic_dropped:
        print("[v9am-numeric-fidelity] " + json.dumps(_NUMERIC_STATS, ensure_ascii=False), flush=True)
    if dropped_numeric:
        raise RuntimeError("numeric publication obligations were dropped from evidence queue")
    return selected


def _numeric_issue_score(targets, translated, memory, sid: str) -> tuple[int, list[str]]:
    base_score, codes = _BASE_ISSUE_SCORE(targets, translated, memory, sid)
    segment = next(segment for segment in targets if str(segment.id) == str(sid))
    check = compare_numeric_fidelity(
        str(segment.text or ""), str(translated.get(sid) or "")
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
        # While the objective number is still wrong, return a constant blocking
        # score. Fixing unrelated style/hard issues cannot make a numerically-wrong
        # candidate appear "better" and slip through acceptance.
        return 1_000_000, codes
    return base_score, codes


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
                "span-aware deterministic EN/RU numeric parser; Russian compact numeric compounds are recognized; "
                "repeated references compare by value presence rather than duplicate count; objective mismatches "
                "outrank the soft evidence budget and cannot be accepted until the value is repaired"
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
