from __future__ import annotations

import json

import hybrid_reference_translation as hybrid
import chapter_reference_translation_v9ah as v9ah
from bookai.gigachat_ultra_provider import GigaChatUltraProvider


_BASE_BUILD = hybrid.build_reference_harness


def _build_ultra_harness():
    harness = _BASE_BUILD()
    old_model = getattr(harness.gate, "model", "unknown")
    harness.gate = GigaChatUltraProvider(role="sparse_semantic_specialist")
    print(
        f"[v9ah-ultra] gate_swap from={old_model} to={harness.gate.model} "
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
            "experiment": "v9ah-ultra-specialist-ab",
            "base": "v9ah unchanged",
            "primary_translation": "GigaChat-3-Lightning",
            "sparse_semantic_specialist": "GigaChat-3-Ultra",
            "deepseek_specialist_replaced": True,
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    hybrid.build_reference_harness = _build_ultra_harness
    try:
        v9ah.main()
    finally:
        hybrid.build_reference_harness = _BASE_BUILD
        _annotate()


if __name__ == "__main__":
    main()
