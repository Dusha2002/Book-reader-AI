from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9ak as v9ak


# Fast-path principle: keep v9ah's proven repair selection exactly as-is, but do
# not ask DeepSeek to verify the same repaired rows a second time. The existing
# deterministic/source-grounded sanitizer remains the publication safety net.
_FAST_STATS: dict[str, Any] = {}


def _fast_no_second_verify(harness, targets, translated, memory, routes):
    """Skip the redundant second DeepSeek verification pass after repair.

    We still compute which rows baseline v9ah would have verified so the A/B
    report records the work avoided. No translation is changed here.
    """
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
                "reason": "v9ah repair already applied; deterministic sanitizer owns residual hard failures",
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
            "experiment": "v9al-fast-deepseek-v2",
            "base": "v9ah adaptive sparse DeepSeek",
            "primary_translation": "GigaChat-3-Lightning",
            "semantic_specialist": "DeepSeek Flash",
            "gigachat_ultra_used": False,
            "repair_router": "unchanged v9ah adaptive router and budget",
            "second_model_verify": False,
            "fast_batch_recovery": (
                "same-batch plain-JSON retry before recursive split for degenerate GigaChat structured output"
            ),
            "post_translation_policy": (
                "repair only v9ah-selected evidence; do not re-judge repaired text unless objective "
                "source-grounded postconditions fail"
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

    # Keep v9ah routing untouched. Reuse only v9ak's transport-level recovery,
    # which has zero quality-policy effect when a normal GigaChat batch succeeds.
    v9ah._adaptive_deep_verify = _fast_no_second_verify
    v9ah.v9.GigaChatLightningV9Backend = v9ak.GigaChatLightningV9AKBackend

    try:
        v9ah.main()
    finally:
        v9ah._adaptive_deep_verify = old_verify
        v9ah.v9.GigaChatLightningV9Backend = old_backend
        _annotate_report()


if __name__ == "__main__":
    main()
