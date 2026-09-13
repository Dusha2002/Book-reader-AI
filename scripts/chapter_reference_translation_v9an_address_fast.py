from __future__ import annotations

import json
import os
from typing import Any

import chapter_reference_translation_v9am_numeric_fast as v9am
from bookai.address_fidelity import normalize_numbered_address_literals
from bookai.semantic_fidelity import compare_short_omission_fidelity


_BASE_CONTRACT_ROUTE = v9am._contract_route_v9am
_BASE_CONTRACT_REPAIR = v9am._contract_validated_deep_repair
_BASE_HARD_ROWS = v9am._contract_hard_rows
_ADDRESS_STATS: dict[str, Any] = {
    "normalization_passes": 0,
    "segments_changed": 0,
    "replacements": 0,
    "changed_ids": [],
}
_PRE_REPAIR_STATS: dict[str, Any] = {}


def _normalize_address_literals(targets, translated) -> dict[str, Any]:
    _ADDRESS_STATS["normalization_passes"] = int(_ADDRESS_STATS.get("normalization_passes") or 0) + 1
    changed_ids = set(str(x) for x in _ADDRESS_STATS.get("changed_ids") or [])
    replacements = 0
    newly_changed = 0
    for segment in targets:
        sid = str(segment.id)
        current = str(translated.get(sid) or "")
        if not current:
            continue
        fixed, count = normalize_numbered_address_literals(str(segment.text or ""), current)
        if count <= 0 or fixed == current:
            continue
        translated[sid] = fixed
        replacements += int(count)
        if sid not in changed_ids:
            newly_changed += 1
            changed_ids.add(sid)
    _ADDRESS_STATS["replacements"] = int(_ADDRESS_STATS.get("replacements") or 0) + replacements
    _ADDRESS_STATS["segments_changed"] = len(changed_ids)
    _ADDRESS_STATS["changed_ids"] = sorted(changed_ids)
    if replacements:
        print(
            "[v9an-address-normalize] "
            + json.dumps(
                {
                    "replacements_this_pass": replacements,
                    "new_segments_this_pass": newly_changed,
                    "changed_ids": sorted(changed_ids),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return dict(_ADDRESS_STATS)


def _latin_issue_map(targets, translated) -> dict[str, list[str]]:
    """Return only the high-precision Latin leak codes already proven by v9af."""
    result: dict[str, list[str]] = {}
    for row in v9am.v9al.v9ah.v9af._objective_issues(targets, translated):
        sid = str(row.get("id") or "")
        codes = [str(code) for code in row.get("codes", []) if str(code).startswith("latin")]
        if sid and codes:
            result[sid] = codes
    return result


def _extra_failure_codes(segment, target: str, *, latin_codes: list[str] | None = None) -> list[str]:
    codes: list[str] = []
    short = compare_short_omission_fidelity(str(segment.text or ""), str(target or ""))
    if not short.get("ok", True):
        codes.append("short_omission")
    if latin_codes is None:
        rows = v9am.v9al.v9ah.v9af._objective_issues([segment], {str(segment.id): str(target or "")})
        latin_codes = [
            str(code)
            for row in rows
            for code in row.get("codes", [])
            if str(code).startswith("latin")
        ]
    for code in latin_codes:
        if code not in codes:
            codes.append(code)
    return codes


def _extra_reason(codes: list[str]) -> str:
    reasons: list[str] = []
    if "short_omission" in codes:
        reasons.append(
            "DETERMINISTIC short omission: current_ru collapses a short multi-beat source. "
            "Corrected Russian MUST translate the complete source, including every dialogue beat and attribution."
        )
    if any(code.startswith("latin") for code in codes):
        reasons.append(
            "PUBLICATION Latin leak: remove accidental English/Latin prose. Translate or recreate wordplay/rhyme in natural "
            "Russian Cyrillic while preserving meaning and the literary device."
        )
    return " ".join(reasons)


def _address_aware_route(targets, translated, memory):
    """Normalize cheap address facts, then promote all proven draft defects early."""
    _normalize_address_literals(targets, translated)
    selected, ranked = _BASE_CONTRACT_ROUTE(targets, translated, memory)

    by_id = {str(segment.id): segment for segment in targets}
    index = {str(segment.id): i for i, segment in enumerate(targets)}
    ranked_by_id = {str(row.get("id") or ""): dict(row) for row in ranked}
    selected_ids = {str(row.get("id") or "") for row in selected}
    latin_map = _latin_issue_map(targets, translated)

    extra_ids: list[str] = []
    extra_codes: dict[str, list[str]] = {}
    for sid, segment in by_id.items():
        codes = _extra_failure_codes(
            segment,
            str(translated.get(sid) or ""),
            latin_codes=latin_map.get(sid, []),
        )
        if not codes:
            continue
        extra_ids.append(sid)
        extra_codes[sid] = codes
        row = ranked_by_id.get(sid) or {
            "id": sid,
            "index": index[sid],
            "priority": 25,
            "code": "publication_contract",
            "reason": "",
            "glossary_hits": [],
        }
        row = dict(row)
        row["priority"] = max(25, int(row.get("priority") or 0))
        if not str(row.get("code") or "").strip():
            row["code"] = "publication_contract"
        prior = str(row.get("reason") or "").strip()
        reason = _extra_reason(codes)
        row["reason"] = "; ".join(x for x in [prior, reason] if x)
        row.setdefault("glossary_hits", [])
        ranked_by_id[sid] = row

    if not extra_ids:
        _PRE_REPAIR_STATS.clear()
        _PRE_REPAIR_STATS.update(
            {
                "extra_contract_routes": 0,
                "extra_contract_ids": [],
                "extra_contract_codes": {},
                "selected_before_extra": len(selected),
                "selected_after_extra": len(selected),
                "extra_routes_added": 0,
            }
        )
        return selected, ranked

    ranked_out: list[dict[str, Any]] = []
    seen_ranked: set[str] = set()
    for raw in ranked:
        sid = str(raw.get("id") or "")
        if not sid or sid in seen_ranked:
            continue
        ranked_out.append(dict(ranked_by_id.get(sid) or raw))
        seen_ranked.add(sid)
    for sid in extra_ids:
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
    for sid in extra_ids:
        if sid not in seen_selected:
            selected_out.append(dict(ranked_by_id[sid]))
            seen_selected.add(sid)
    selected_out.sort(key=lambda row: (-int(row.get("priority") or 0), int(row.get("index") or 0)))

    _PRE_REPAIR_STATS.clear()
    _PRE_REPAIR_STATS.update(
        {
            "extra_contract_routes": len(extra_ids),
            "extra_contract_ids": extra_ids,
            "extra_contract_codes": extra_codes,
            "selected_before_extra": len(selected),
            "selected_after_extra": len(selected_out),
            "extra_routes_added": len([sid for sid in extra_ids if sid not in selected_ids]),
        }
    )
    print("[v9an-pre-repair] " + json.dumps(_PRE_REPAIR_STATS, ensure_ascii=False), flush=True)
    return selected_out, ranked_out


def _strict_publication_retry(harness, targets, translated, memory, rejected_ids, before) -> tuple[list[str], int]:
    """One bounded retry only for first-wave rows that still violate hard proof.

    This replaces the old chain where Giga sanitizer retried Latin wordplay several
    times and the final DeepSeek evidence pass retried it yet again. The retry is
    source-grounded, small, and accepted only if every deterministic contract is
    actually clean afterwards.
    """
    if not rejected_ids:
        return [], 0
    by_id = {str(segment.id): segment for segment in targets}
    index = {str(segment.id): i for i, segment in enumerate(targets)}
    payload = []
    for sid in rejected_ids:
        segment = by_id.get(str(sid))
        if segment is None:
            continue
        i = index[str(sid)]
        current = str(before.get(str(sid)) or translated.get(str(sid)) or "")
        codes = _extra_failure_codes(segment, current)
        if not codes:
            continue
        payload.append(
            {
                "id": str(sid),
                "failed_checks": codes,
                "source": str(segment.text or ""),
                "current_ru": current,
                "before_en": [str(x.text or "") for x in targets[max(0, i - 2):i]],
                "after_en": [str(x.text or "") for x in targets[i + 1:i + 3]],
            }
        )
    if not payload:
        return [], 0

    system = """You are a STRICT final EN→RU publication-contract repairer. Every input row already failed a deterministic check.
Return a COMPLETE Russian translation of exactly SOURCE for every id. Do not merely edit one word if that leaves the
source incomplete.

Mandatory rules:
- latin_leak: corrected_ru must contain NO accidental English/Latin prose. Render proper names in established Cyrillic.
  If the source contains quoted rhyme, pun, or wordplay, recreate the literary device in natural Russian Cyrillic;
  do not leave English words and do not use Latin transliteration as a substitute.
- short_omission: preserve every source sentence/dialogue beat, speaker attribution, action, adjective and object.
- Preserve all numbers, polarity, questions, chronology, roles and causal relations exactly.
- BEFORE_EN/AFTER_EN are context only; never import their facts into corrected_ru.

ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}; exactly one row per input id.
"""
    try:
        obj = v9am.v9al.v9ah.v8._complete_json(harness.gate, system, {"items": payload})
        raw = obj.get("items") or []
    except Exception as exc:
        print(f"[v9an-strict-retry] error={type(exc).__name__}", flush=True)
        return [], 1

    parsed = {
        str(row.get("id") or ""): row
        for row in raw
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    recovered: list[str] = []
    for item in payload:
        sid = item["id"]
        segment = by_id[sid]
        row = parsed.get(sid) or {}
        candidate = v9am.v9al.v9ah.v9._norm_text(row.get("corrected_ru") or "")
        if not candidate:
            continue
        try:
            candidate = v9am.v9al.v9ah.v9ag.v9ad._canonicalize_candidate(segment, candidate, memory)
        except Exception:
            pass
        if not candidate:
            continue
        if v9am._contract_failures(str(segment.text or ""), candidate):
            continue
        if _extra_failure_codes(segment, candidate):
            continue
        old = str(before.get(sid) or "")
        if v9am.v9al.v9ah.v9._fatal_count(segment, candidate, memory) > v9am.v9al.v9ah.v9._fatal_count(segment, old, memory):
            continue
        translated[sid] = candidate
        recovered.append(sid)

    print(
        "[v9an-strict-retry] "
        + json.dumps(
            {
                "requested_ids": [item["id"] for item in payload],
                "recovered_ids": recovered,
                "unrecovered_ids": [item["id"] for item in payload if item["id"] not in recovered],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return recovered, 1


def _extra_validated_repair(harness, targets, translated, memory, routes):
    """Keep v9am validation; strict-retry only failed short/Latin obligations."""
    before = {
        str(row.get("id") or ""): str(translated.get(str(row.get("id") or "")) or "")
        for row in routes
    }
    changed, calls = _BASE_CONTRACT_REPAIR(harness, targets, translated, memory, routes)
    by_id = {str(segment.id): segment for segment in targets}
    accepted: list[str] = []
    rejected: list[str] = []

    for sid in changed:
        sid = str(sid)
        segment = by_id.get(sid)
        if segment is None:
            continue
        candidate = str(translated.get(sid) or "")
        extra = _extra_failure_codes(segment, candidate)
        if extra:
            translated[sid] = before.get(sid, "")
            rejected.append(sid)
        else:
            accepted.append(sid)

    recovered, retry_calls = _strict_publication_retry(
        harness, targets, translated, memory, rejected, before
    )
    accepted.extend(sid for sid in recovered if sid not in accepted)
    rejected_final = [sid for sid in rejected if sid not in set(recovered)]

    _PRE_REPAIR_STATS["extra_validation_rejected_initial_ids"] = rejected
    _PRE_REPAIR_STATS["strict_retry_calls"] = retry_calls
    _PRE_REPAIR_STATS["strict_retry_recovered_ids"] = recovered
    _PRE_REPAIR_STATS["extra_validation_rejected_final_ids"] = rejected_final
    _PRE_REPAIR_STATS["repair_changed_after_extra_validation"] = len(accepted)
    if rejected_final:
        print(
            "[v9an-extra-repair-reject] "
            + json.dumps({"rejected_ids": rejected_final, "accepted_ids": accepted}, ensure_ascii=False),
            flush=True,
        )
    return accepted, calls + retry_calls


def _address_aware_hard_rows(targets, translated, memory):
    _normalize_address_literals(targets, translated)
    return _BASE_HARD_ROWS(targets, translated, memory)


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
            "experiment": "v9an-fast-deepseek-address-contracts",
            "numbered_address_cleanup": (
                "deterministic source-proven English ordinal/cardinal address labels are rendered in Russian before specialist routing"
            ),
            "pre_repair_publication_contracts": (
                "short multi-beat omissions and v9af-proven Latin leaks are mandatory first-wave DeepSeek routes"
            ),
            "failed_publication_contract_retry": (
                "one strict bounded DeepSeek retry replaces repeated Giga sanitizer plus final generic evidence when first-wave output still fails proof"
            ),
            "deepseek_repair_batch_minimum": 16,
            "gigachat_ultra_used": False,
        }
    )
    data["architecture"] = architecture
    data["v9an_address_fidelity"] = dict(_ADDRESS_STATS)
    data["v9an_pre_repair_contracts"] = dict(_PRE_REPAIR_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_route = v9am._contract_route_v9am
    old_rows = v9am._contract_hard_rows
    old_repair = v9am._contract_validated_deep_repair
    old_batch = os.environ.get("BOOKAI_V9AB_DEEP_BATCH")

    v9am._contract_route_v9am = _address_aware_route
    v9am._contract_hard_rows = _address_aware_hard_rows
    v9am._contract_validated_deep_repair = _extra_validated_repair
    try:
        try:
            current_batch = int(os.environ.get("BOOKAI_V9AB_DEEP_BATCH") or "0")
        except ValueError:
            current_batch = 0
        if current_batch < 16:
            os.environ["BOOKAI_V9AB_DEEP_BATCH"] = "16"
        v9am.main()
    finally:
        v9am._contract_route_v9am = old_route
        v9am._contract_hard_rows = old_rows
        v9am._contract_validated_deep_repair = old_repair
        if old_batch is None:
            os.environ.pop("BOOKAI_V9AB_DEEP_BATCH", None)
        else:
            os.environ["BOOKAI_V9AB_DEEP_BATCH"] = old_batch
        _annotate_report()


if __name__ == "__main__":
    main()
