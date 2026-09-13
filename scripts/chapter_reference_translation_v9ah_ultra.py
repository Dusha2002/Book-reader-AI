from __future__ import annotations

import json

import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9y as v9y
from bookai.gigachat_runtime_guard import install_gigachat_runtime_guard
from bookai.gigachat_ultra_provider import GigaChatUltraProvider


_BASE_ADAPT = v9y.adapt_harness


def _adapt_harness_with_ultra(harness):
    # Keep v9y's direct-adapter setup for all legacy fallback roles, then replace
    # ONLY the sparse semantic gate after that adapter has finished. Patching the
    # earlier builder is insufficient because v9y.adapt_harness rewrites gate.
    harness = _BASE_ADAPT(harness)
    old_model = getattr(harness.gate, "model", "unknown")
    harness.gate = GigaChatUltraProvider(role="sparse_semantic_specialist")
    print(
        f"[v9ah-ultra] gate_swap_after_adapt from={old_model} to={harness.gate.model} "
        "scope=sparse-repair+final-verifier only",
        flush=True,
    )
    return harness


def _annotate() -> None:
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
            "experiment": "v9ah-ultra-specialist-promoted",
            "base": "v9ah unchanged",
            "primary_translation": "GigaChat-3-Lightning",
            "sparse_semantic_specialist": "GigaChat-3-Ultra",
            "deepseek_specialist_replaced": True,
            "deepseek_legacy_fallbacks_retained": True,
            "gigachat_rate_limit_guard": True,
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    install_gigachat_runtime_guard()
    v9y.adapt_harness = _adapt_harness_with_ultra
    try:
        v9ah.main()
    finally:
        v9y.adapt_harness = _BASE_ADAPT
        _annotate()


if __name__ == "__main__":
    main()
