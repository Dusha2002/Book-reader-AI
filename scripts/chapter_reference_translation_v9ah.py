from __future__ import annotations

import json
import math
import os
from typing import Any

import chapter_reference_translation_v9ag as v9ag

v9af = v9ag.v9af
v9ad = v9ag.v9ad
v9ac = v9ad.v9ac
v9ab = v9ag.v9ab
v9 = v9ag.v9
v8 = v9ad.v8

# Freeze the implementations before monkey-patching the runtime chain.
_BASE_AB_ROUTE = v9ac._ORIGINAL_AB_ROUTE
_BASE_DEEP_REPAIR = v9ad._deep_repair_v9ad
_BASE_DEEP_VERIFY = v9ad._deep_verify_v9ad
_BASE_QUALITY = v9ag._quality_v9ag

_LAST_REPAIR_CHANGED: set[str] = set()
_ADAPTIVE_STATS: dict[str, Any] = {}

# High-confidence source patterns. v9ab/v9aa promote these to priority >=22.
# They are never discarded because a chapter happened to exceed a cost budget.
_MANDATORY_PRIORITY = 22


def _source_chars(targets) -> int:
    return sum(len(str(getattr(s, "text", "") or "")) for s in targets)


def _repair_target(chars: int, candidate_count: int) -> tuple[int, float, int]:
    """Return (soft_target, density_per_10k, soft_ceiling).

    Length contributes slowly; semantic-risk density contributes faster. This keeps
    normal chapters cheap while allowing genuinely difficult/long chapters to grow.
    The ceiling is SOFT: mandatory high-confidence routes can exceed it.
    """
    units = max(1.0, chars / 10000.0)
    density = candidate_count / units

    # ~11 slots around a clean 50k-char chapter before density adjustment.
    length_slots = max(7, math.ceil(chars / 9000.0) + 5)
    density_bonus = max(0, min(8, int(round((density - 4.0) * 1.5))))
    target = length_slots + density_bonus

    # Cost guard for medium risks only. Roughly 17-20 slots for our 50-60k chapters,
    # scaling to ~40 for a 120k chapter. Mandatory routes may exceed this.
    soft_ceiling = max(10, min(48, math.ceil(chars / 3000.0)))
    target = max(8, min(target, soft_ceiling))
    return target, round(density, 2), soft_ceiling


def _adaptive_route_v9ah(targets, translated, memory):
    # v9ac temporarily exposes the frozen pre-v9ab router while calling us, so the
    # saved v9ab implementation is recursion-safe and gives the full ranked list.
    _, ranked = _BASE_AB_ROUTE(targets, translated, memory)
    ranked = [dict(row) for row in ranked]
    chars = _source_chars(targets)
    target, density, soft_ceiling = _repair_target(chars, len(ranked))

    mandatory = [row for row in ranked if int(row.get("priority") or 0) >= _MANDATORY_PRIORITY]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Never cut strong semantic obligations for budget reasons.
    for row in mandatory:
        sid = str(row.get("id") or "")
        if sid and sid not in seen:
            selected.append(row)
            seen.add(sid)

    # Fill the soft budget with the best remaining medium-risk candidates.
    desired = max(target, len(selected))
    for row in ranked:
        if len(selected) >= desired:
            break
        sid = str(row.get("id") or "")
        if not sid or sid in seen:
            continue
        selected.append(row)
        seen.add(sid)

    selected.sort(key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)))
    _ADAPTIVE_STATS.update({
        "chapter_chars": chars,
        "router_candidates": len(ranked),
        "risk_density_per_10k": density,
        "mandatory_routes": len(mandatory),
        "repair_soft_target": target,
        "repair_soft_ceiling": soft_ceiling,
        "repair_selected": len(selected),
        "mandatory_budget_override": max(0, len(mandatory) - soft_ceiling),
    })
    print("[v9ah-budget] " + json.dumps(_ADAPTIVE_STATS, ensure_ascii=False), flush=True)
    return selected, ranked


def _adaptive_deep_repair(harness, targets, translated, memory, routes):
    global _LAST_REPAIR_CHANGED
    changed, calls = _BASE_DEEP_REPAIR(harness, targets, translated, memory, routes)
    _LAST_REPAIR_CHANGED = set(changed)
    _ADAPTIVE_STATS["repair_routes_actual"] = len(routes)
    _ADAPTIVE_STATS["repair_batches_actual"] = calls
    _ADAPTIVE_STATS["repair_changed"] = len(changed)
    return changed, calls


def _verify_routes(targets, routes):
    if not routes:
        return []
    chars = _source_chars(targets)
    route_count = len(routes)
    units = max(1.0, chars / 10000.0)
    density = route_count / units

    mandatory_ids = {
        str(row.get("id") or "")
        for row in routes
        if int(row.get("priority") or 0) >= _MANDATORY_PRIORITY
    }
    must_verify = mandatory_ids | _LAST_REPAIR_CHANGED

    # Usually verify ~55% of routed items, with a small complexity bonus. This means
    # a clean short chapter often uses one verify batch, while a difficult chapter
    # automatically gets two or more. All changed/mandatory items bypass the target.
    target = max(6, math.ceil(route_count * 0.55))
    if density >= 7.0:
        target += 2
    elif density >= 5.5:
        target += 1
    soft_ceiling = max(8, min(32, math.ceil(chars / 6000.0) + 4))
    target = min(target, soft_ceiling)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in routes:
        sid = str(row.get("id") or "")
        if sid in must_verify and sid not in seen:
            selected.append(row)
            seen.add(sid)
    desired = max(target, len(selected))
    for row in routes:
        if len(selected) >= desired:
            break
        sid = str(row.get("id") or "")
        if sid and sid not in seen:
            selected.append(row)
            seen.add(sid)

    selected.sort(key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)))
    _ADAPTIVE_STATS.update({
        "verify_mandatory_or_changed": len(must_verify),
        "verify_soft_target": target,
        "verify_soft_ceiling": soft_ceiling,
        "verify_selected": len(selected),
    })
    return selected


def _adaptive_deep_verify(harness, targets, translated, memory, routes):
    selected = _verify_routes(targets, routes)
    if not selected:
        return [], 0, 0

    # Reuse v9ad's proven verifier, but give it a temporary cap equal to the dynamic
    # selection so its old fixed BOOKAI_V9AB_FINAL_VERIFY_MAX cannot truncate it.
    old_cap = os.environ.get("BOOKAI_V9AB_FINAL_VERIFY_MAX")
    os.environ["BOOKAI_V9AB_FINAL_VERIFY_MAX"] = str(max(8, len(selected)))
    try:
        changed, confirmed, calls = _BASE_DEEP_VERIFY(harness, targets, translated, memory, selected)
    finally:
        if old_cap is None:
            os.environ.pop("BOOKAI_V9AB_FINAL_VERIFY_MAX", None)
        else:
            os.environ["BOOKAI_V9AB_FINAL_VERIFY_MAX"] = old_cap
    _ADAPTIVE_STATS["verify_batches_actual"] = calls
    _ADAPTIVE_STATS["verify_changed"] = len(changed)
    _ADAPTIVE_STATS["verify_confirmed_defects"] = confirmed
    print("[v9ah-verify-budget] " + json.dumps(_ADAPTIVE_STATS, ensure_ascii=False), flush=True)
    return changed, confirmed, calls


def _quality_v9ah(harness, targets, translated, memory):
    stats = dict(_BASE_QUALITY(harness, targets, translated, memory) or {})
    stats["adaptive_deepseek_budget"] = dict(_ADAPTIVE_STATS)
    stats["quality_mode"] = "v9ah-adaptive-sparse-deepseek+validated-source-sanitizer"
    v9ab._V9AB_STATS = dict(stats)
    print("[bookai-v9ah] " + json.dumps({
        "adaptive_deepseek_budget": _ADAPTIVE_STATS,
        "deep_repair_batches": stats.get("deep_repair_batches"),
        "deep_verify_batches": stats.get("deep_verify_batches"),
        "sanitizer_final_residual": stats.get("sanitizer_final_residual"),
    }, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    # v9ac owns recursion-safe routing. Replace only the implementation it calls.
    old_route = v9ac._ORIGINAL_AB_ROUTE
    old_repair = v9ad._deep_repair_v9ad
    old_verify = v9ad._deep_verify_v9ad
    old_quality = v9ag._quality_v9ag
    v9ac._ORIGINAL_AB_ROUTE = _adaptive_route_v9ah
    v9ad._deep_repair_v9ad = _adaptive_deep_repair
    v9ad._deep_verify_v9ad = _adaptive_deep_verify
    v9ag._quality_v9ag = _quality_v9ah
    try:
        v9ag.main()
    finally:
        v9ac._ORIGINAL_AB_ROUTE = old_route
        v9ad._deep_repair_v9ad = old_repair
        v9ad._deep_verify_v9ad = old_verify
        v9ag._quality_v9ag = old_quality


if __name__ == "__main__":
    main()
