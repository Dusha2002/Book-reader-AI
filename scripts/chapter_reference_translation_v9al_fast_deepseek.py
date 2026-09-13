from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9ak as v9ak


# This experiment deliberately keeps v9ah's original DeepSeek specialist.
# It changes only the transport/routing around it and removes the redundant
# second model verification pass. Objective source-grounded sanitizers remain.
_BASE_VERIFY = v9ah._adaptive_deep_verify
_BASE_ROUTE = v9ah._BASE_AB_ROUTE
_BASE_BACKEND = v9ah.v9.GigaChatLightningV9Backend

_FAST_STATS: dict[str, Any] = {}


def _fast_no_second_verify(harness, targets, translated, memory, routes):
    """Skip the second DeepSeek verification pass after DeepSeek repair.

    v9ah already sends semantically risky rows to the specialist and then runs
    deterministic/source-grounded postconditions through v9ag. Re-asking a model
    to verify most of the same repaired rows adds serial latency and can rewrite
    already-correct literary text. We still compute which rows v9ah *would* have
    verified so the benchmark can report exactly what was avoided.
    """
    would_verify = v9ah._verify_routes(targets, routes)
    _FAST_STATS.update(
        {
            "second_model_verify_enabled": False,
            "would_verify_segments": len(would_verify),
            "avoided_verify_ids": [str(row.get("id") or "") for row in would_verify],
            "avoided_verify_batches": None,
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
                "reason": "repair-on-evidence then deterministic postconditions",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return [], 0, 0


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
            "experiment": "v9al-fast-deepseek",
            "base": "v9ah adaptive sparse DeepSeek",
            "primary_translation": "GigaChat-3-Lightning",
            "semantic_specialist": "DeepSeek Flash",
            "gigachat_ultra_used": False,
            "second_model_verify": False,
            "fast_batch_recovery": (
                "same-batch plain-JSON retry before recursive split for degenerate structured output"
            ),
            "risk_router": (
                "v9ah adaptive budget + generic semantic categories + cross-segment duplicate guard"
            ),
            "post_translation_policy": (
                "do not rewrite accepted text without evidence; deterministic/source-grounded "
                "sanitizer remains authoritative"
            ),
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    data["v9al_fast_path"] = dict(_FAST_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_verify = v9ah._adaptive_deep_verify
    old_route = v9ah._BASE_AB_ROUTE
    old_backend = v9ah.v9.GigaChatLightningV9Backend

    # Reuse only v9ak's model-free/general routing and fast batch recovery.
    # Do NOT invoke v9ak.main(): that would add its later quality/postcondition
    # stack and muddy a clean v9ah-vs-fast-path comparison.
    v9ah._adaptive_deep_verify = _fast_no_second_verify
    v9ah._BASE_AB_ROUTE = v9ak._route_v9ak
    v9ah.v9.GigaChatLightningV9Backend = v9ak.GigaChatLightningV9AKBackend

    try:
        v9ah.main()
    finally:
        v9ah._adaptive_deep_verify = old_verify
        v9ah._BASE_AB_ROUTE = old_route
        v9ah.v9.GigaChatLightningV9Backend = old_backend
        _annotate_report()


if __name__ == "__main__":
    main()
