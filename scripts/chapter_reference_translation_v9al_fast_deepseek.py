from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9ak as v9ak


# Fast-path principle: keep v9ah's proven repair selection exactly as-is, but do
# not ask DeepSeek to verify the same repaired rows a second time. Instead, after
# the existing source-grounded sanitizer, spend at most ONE DeepSeek batch on
# rows that still have concrete evidence of a publication defect.
_FAST_STATS: dict[str, Any] = {}
_ORIGINAL_BASE_QUALITY = v9ah._BASE_QUALITY

_NUMBER_ADDRESS_RE = re.compile(
    r"\b(?:number\s+)?(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth))?"
    r"(?:\s+(?:street|road|avenue|house|door|address))?\b",
    re.I,
)


def _fast_no_second_verify(harness, targets, translated, memory, routes):
    """Skip the redundant second DeepSeek verification pass after repair."""
    would_verify = v9ah._verify_routes(targets, routes)
    _FAST_STATS.update(
        {
            "second_model_verify_enabled": False,
            "would_verify_segments": len(would_verify),
            "avoided_verify_ids": [str(row.get("id") or "") for row in would_verify],
        }
    )
    v9ah._ADAPTIVE_STATS.update(
        {
            "fast_path_second_verify_skipped": True,
            "fast_path_would_verify": len(would_verify),
            "verify_batches_actual": 0,
            "verify_changed": 0,
            "verify_confirmed_defects": 0,
        }
    )
    print(
        "[v9al-fast-verify] "
        + json.dumps(
            {
                "skipped": True,
                "would_verify_segments": len(would_verify),
                "reason": "v9ah repair already applied; evidence-based postcondition owns residual defects",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return [], 0, 0


def _hard_issue_rows(targets, translated, memory) -> list[dict[str, Any]]:
    v3 = v9ah.v9ag.v9ad.v9ac.v9ab.v3
    issue_map: dict[str, set[str]] = {}

    # Existing generic hard QA: question loss, digit changes, omissions, mixed
    # script, entity consistency, etc.
    for issue in v3.enhanced_batch_issues(
        targets, translated, memory, source_segments=targets
    ):
        if str(getattr(issue, "severity", "")) == "hard":
            issue_map.setdefault(str(issue.id), set()).add(str(issue.code))

    # v9ag's narrower source-grounded blockers (notably residual Latin prose).
    for row in v9ah.v9af._objective_issues(targets, translated):
        sid = str(row.get("id") or "")
        if sid:
            issue_map.setdefault(sid, set()).update(str(x) for x in row.get("codes", []))

    by_id = {segment.id: (i, segment) for i, segment in enumerate(targets)}

    # Cheap short-omission detector. v3 intentionally ignored very short rows to
    # avoid false positives; here we only flag extreme shrinkage with evidence of
    # multiple source clauses/dialogue beats.
    for i, segment in enumerate(targets):
        source = str(segment.text or "").strip()
        current = str(translated.get(segment.id) or "").strip()
        if not source or not current:
            continue
        ratio = len(current) / max(1, len(source))
        multi_beat = source.count(".") >= 2 or source.count("'") >= 4 or source.count('"') >= 4
        if 20 <= len(source) < 100 and ratio < 0.42 and multi_beat:
            issue_map.setdefault(segment.id, set()).add("short_omission")

        # Numbered addresses/streets are rare but semantically brittle. Route only
        # these explicit number-entity constructions instead of every number in the
        # chapter; DeepSeek must preserve every value exactly.
        if _NUMBER_ADDRESS_RE.search(source):
            issue_map.setdefault(segment.id, set()).add("numbered_entity_exactness")

    rows: list[dict[str, Any]] = []
    for sid, codes in issue_map.items():
        found = by_id.get(sid)
        if found is None:
            continue
        i, segment = found
        rows.append(
            {
                "id": sid,
                "index": i,
                "codes": sorted(codes),
                "source": str(segment.text or ""),
                "current_ru": str(translated.get(sid) or ""),
                "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
                "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
            }
        )
    rows.sort(key=lambda row: int(row["index"]))
    cap = max(4, int(__import__("os").getenv("BOOKAI_FAST_EVIDENCE_MAX") or "16"))
    return rows[:cap]


def _issue_score(targets, translated, memory, sid: str) -> tuple[int, list[str]]:
    v3 = v9ah.v9ag.v9ad.v9ac.v9ab.v3
    by_id = {segment.id: segment for segment in targets}
    segment = by_id[sid]
    candidate = str(translated.get(sid) or "")
    codes: list[str] = []
    score = 0
    for issue in v3.enhanced_candidate_issues(
        segment, candidate, memory, source_segments=targets
    ):
        if str(getattr(issue, "severity", "")) == "hard":
            codes.append(str(issue.code))
            score += 3
    for row in v9ah.v9af._objective_issues(targets, translated):
        if str(row.get("id") or "") == sid:
            for code in row.get("codes", []):
                code = str(code)
                if code not in codes:
                    codes.append(code)
                score += 4
            break
    return score, codes


def _evidence_deepseek_cleanup(harness, targets, translated, memory) -> dict[str, Any]:
    rows = _hard_issue_rows(targets, translated, memory)
    if not rows:
        return {
            "candidates": 0,
            "deepseek_calls": 0,
            "changed": 0,
            "accepted_ids": [],
            "residual_candidates": 0,
        }

    payload = [
        {
            "id": row["id"],
            "failed_checks": row["codes"],
            "source": row["source"],
            "current_ru": row["current_ru"],
            "before_en": row["before_en"],
            "after_en": row["after_en"],
        }
        for row in rows
    ]
    system = """You are the final evidence-based EN→RU literary repair specialist.
Every supplied row has a CONCRETE deterministic/source-grounded failure. This is NOT a general style rewrite.
Repair only the listed defect while preserving all correct meaning and established Russian names.

Hard requirements:
- Translate the COMPLETE SOURCE; never shorten, summarize, or omit a dialogue beat.
- Preserve every person/actor/action/object, negation, question, chronology and causal relation.
- Preserve every number and numbered street/house/address EXACTLY. Never turn 30 into 33 or 28 into another value.
- Remove accidental Latin-script English prose. For quoted wordplay/rhymes, recreate the literary device in natural Russian Cyrillic rather than leaving English words.
- Keep dialogue attribution and punctuation natural in literary Russian.
- Do not import facts from BEFORE_EN/AFTER_EN; they are context only.
Return exactly one complete correction for every id.
ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}.
"""

    try:
        obj = v9ah.v8._complete_json(harness.gate, system, {"items": payload})
        raw = obj.get("items") or []
        calls = 1
    except Exception as exc:
        print(f"[v9al-evidence-repair] error={type(exc).__name__}", flush=True)
        return {
            "candidates": len(rows),
            "candidate_ids": [row["id"] for row in rows],
            "deepseek_calls": 1,
            "changed": 0,
            "accepted_ids": [],
            "error": type(exc).__name__,
        }

    parsed = {
        str(row.get("id") or ""): row
        for row in raw
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    accepted: list[str] = []
    rejected: list[str] = []

    for row in rows:
        sid = row["id"]
        item = parsed.get(sid) or {}
        candidate = v9ah.v9._norm_text(item.get("corrected_ru") or "")
        if not candidate:
            rejected.append(sid)
            continue

        old_text = str(translated.get(sid) or "")
        before_score, _ = _issue_score(targets, translated, memory, sid)
        translated[sid] = candidate
        after_score, _ = _issue_score(targets, translated, memory, sid)

        # Accept only an objectively better state. A numbered-entity row may not
        # have had a generic QA score, so accept a non-truncated complete candidate
        # unless it introduces a new hard defect.
        if after_score < before_score or (
            "numbered_entity_exactness" in row["codes"]
            and after_score == 0
            and len(candidate) >= max(8, int(len(row["source"]) * 0.45))
        ):
            accepted.append(sid)
        else:
            translated[sid] = old_text
            rejected.append(sid)

    if accepted:
        v9ah.v9ab._apply_name_canon(targets, translated, memory)
        for segment in targets:
            translated[segment.id] = v9ah.v9af.v9s._format_dialogue_v9s(
                segment, translated.get(segment.id, "")
            )[0]

    residual = _hard_issue_rows(targets, translated, memory)
    result = {
        "candidates": len(rows),
        "candidate_ids": [row["id"] for row in rows],
        "candidate_codes": dict(Counter(code for row in rows for code in row["codes"])),
        "deepseek_calls": calls,
        "changed": len(accepted),
        "accepted_ids": accepted,
        "rejected_ids": rejected,
        "residual_candidates": len(residual),
        "residual_ids": [row["id"] for row in residual],
    }
    print("[v9al-evidence-repair] " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _quality_fast(harness, targets, translated, memory):
    stats = dict(_ORIGINAL_BASE_QUALITY(harness, targets, translated, memory) or {})
    evidence = _evidence_deepseek_cleanup(harness, targets, translated, memory)

    # Rebuild the final v9 state from the actual post-repair text so the exported
    # quality report is not stale.
    semantic: dict[str, list[dict[str, Any]]] = {}
    final_map, final_scores, det_final = v9ah.v9af.v9t._rebuild_full_map(
        targets, translated, memory, semantic
    )
    critical, major = v9ah.v9af.v9s._publish_final_state(targets, final_map, final_scores)
    counts = Counter(row.get("severity") for values in final_map.values() for row in values)
    stats.update(
        {
            "fast_evidence_repair": evidence,
            "final_deterministic": det_final,
            "remaining_critical": len(critical),
            "remaining_major_high_confidence": len(major),
            "mean_quality_score": round(sum(final_scores.values()) / max(1, len(final_scores)), 2),
            "severity_counts": dict(counts),
            "quality_mode": "v9ah-fast-deepseek-evidence-only",
        }
    )
    return stats


def _annotate_report() -> None:
    v3 = v9ah.v9ag.v9ad.v9ac.v9ab.v3
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
            "experiment": "v9al-fast-deepseek-v3",
            "base": "v9ah adaptive sparse DeepSeek",
            "primary_translation": "GigaChat-3-Lightning",
            "semantic_specialist": "DeepSeek Flash",
            "gigachat_ultra_used": False,
            "repair_router": "unchanged v9ah adaptive router and budget",
            "second_model_verify": False,
            "final_specialist": "one bounded evidence-only DeepSeek batch when hard postconditions exist",
            "fast_batch_recovery": (
                "same-batch plain-JSON retry before recursive split for degenerate GigaChat structured output"
            ),
            "post_translation_policy": (
                "never re-judge clean text; repair only deterministic/source-grounded failures"
            ),
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    data["v9al_fast_path"] = dict(_FAST_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_verify = v9ah._adaptive_deep_verify
    old_backend = v9ah.v9.GigaChatLightningV9Backend
    old_base_quality = v9ah._BASE_QUALITY

    # Keep v9ah routing untouched. Reuse only v9ak's transport-level recovery,
    # which has zero quality-policy effect when a normal GigaChat batch succeeds.
    v9ah._adaptive_deep_verify = _fast_no_second_verify
    v9ah.v9.GigaChatLightningV9Backend = v9ak.GigaChatLightningV9AKBackend
    v9ah._BASE_QUALITY = _quality_fast

    try:
        v9ah.main()
    finally:
        v9ah._adaptive_deep_verify = old_verify
        v9ah.v9.GigaChatLightningV9Backend = old_backend
        v9ah._BASE_QUALITY = old_base_quality
        _annotate_report()


if __name__ == "__main__":
    main()
