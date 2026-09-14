from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import chapter_reference_translation_v9ah as legacy_kernel
import chapter_reference_translation_v9y as legacy_transport
from bookai.gigachat_runtime_guard import install_gigachat_runtime_guard
from bookai.gigachat_ultra_provider import GigaChatUltraProvider
from bookai.release_guards import specialist_guard_routes


@dataclass
class LiteraryTranslationStrategy:
    """Stable production facade over the proven v9ah translation kernel.

    New production changes belong here instead of creating v9ai/v9aj/... scripts.
    The legacy experiment modules remain importable for reproducibility, but the
    workflow and compatibility entrypoint use this single named strategy.
    """

    name: str = "literary-production-v1"
    primary: str = "GigaChat-3-Lightning"
    specialist: str = "GigaChat-3-Ultra"
    fallback: str = "DeepSeek legacy emergency fallback"
    _base_route: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _guard_stats: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def guarded_route(self, targets, translated, memory):
        if self._base_route is None:
            raise RuntimeError("production strategy route used before installation")
        selected, ranked = self._base_route(targets, translated, memory)
        guard_rows = specialist_guard_routes(list(targets), dict(translated))
        self._guard_stats = {
            "guard_candidates": len(guard_rows),
            "polarity_scope": sum(row.get("code") == "polarity_scope" for row in guard_rows),
            "source_contamination": sum(row.get("code") == "source_contamination" for row in guard_rows),
        }

        by_id = {str(row.get("id")): dict(row) for row in ranked}
        for guard in guard_rows:
            sid = str(guard["id"])
            existing = by_id.get(sid)
            if existing is None:
                by_id[sid] = dict(guard)
                continue
            existing["priority"] = max(int(existing.get("priority") or 0), int(guard.get("priority") or 0))
            existing["code"] = guard.get("code") or existing.get("code")
            existing["guard_codes"] = sorted(
                set(existing.get("guard_codes") or []) | set(guard.get("guard_codes") or [])
            )
            existing["reason"] = (
                str(existing.get("reason") or "").strip()
                + "; PRODUCTION GUARD: "
                + str(guard.get("reason") or "").strip()
            ).strip("; ")
            by_id[sid] = existing

        ranked_out = sorted(
            by_id.values(),
            key=lambda row: (-int(row.get("priority") or 0), int(row.get("index") or 0)),
        )
        selected_ids = {str(row.get("id")) for row in selected}
        for row in ranked_out:
            if int(row.get("priority") or 0) >= 22:
                selected_ids.add(str(row.get("id")))
        selected_out = [row for row in ranked_out if str(row.get("id")) in selected_ids]
        return selected_out, ranked_out

    def adapt_harness(self, harness):
        harness = legacy_transport.adapt_harness(harness)
        previous = getattr(harness.gate, "model", "unknown")
        harness.gate = GigaChatUltraProvider(role="sparse_semantic_specialist")
        print(
            f"[production-strategy] specialist_swap from={previous} to={harness.gate.model} "
            "scope=mandatory-risk-repair+final-verification",
            flush=True,
        )
        return harness

    def annotate(self) -> None:
        v3 = legacy_kernel.v9ag.v9ad.v9ac.v9ab.v3
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
                "production_strategy": self.name,
                "entrypoint": "scripts/production_literary_translation.py",
                "primary_translation": self.primary,
                "sparse_semantic_specialist": self.specialist,
                "legacy_emergency_fallback": self.fallback,
                "guard_routing": ["polarity_scope", "source_contamination"],
                "guard_stats": dict(self._guard_stats),
                "gold_reference_available_to_pipeline": False,
                "versioned_scripts_policy": "legacy experiment history only; production changes go through LiteraryTranslationStrategy",
            }
        )
        data["architecture"] = architecture
        report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def run(self) -> None:
        install_gigachat_runtime_guard()
        original_adapt = legacy_transport.adapt_harness
        original_route = legacy_kernel._BASE_AB_ROUTE
        self._base_route = original_route
        legacy_transport.adapt_harness = self.adapt_harness
        legacy_kernel._BASE_AB_ROUTE = self.guarded_route
        try:
            legacy_kernel.main()
        finally:
            legacy_kernel._BASE_AB_ROUTE = original_route
            legacy_transport.adapt_harness = original_adapt
            self.annotate()


def main() -> None:
    LiteraryTranslationStrategy().run()


if __name__ == "__main__":
    main()
