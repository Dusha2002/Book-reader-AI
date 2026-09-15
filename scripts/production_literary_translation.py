from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import chapter_reference_translation_v9ah as legacy_kernel
import chapter_reference_translation_v9y as legacy_transport
from bookai.gigachat_runtime_guard import install_gigachat_runtime_guard
from bookai.gigachat_ultra_provider import GigaChatUltraProvider
from bookai.release_guards import source_fingerprint, specialist_guard_routes

# IMPORTANT: freeze the direct DeepSeek adapter before production monkey-patches
# v9y's lookup point. This keeps legacy emergency providers on direct transport.
_BASE_DIRECT_ADAPT = legacy_transport.adapt_harness


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
    _base_deep_repair: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _base_deep_verify: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _specialist_provider: GigaChatUltraProvider | None = field(default=None, init=False, repr=False)
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
        """Keep legacy emergency providers on direct transport.

        Ultra is intentionally NOT installed here. v9ah freezes its repair/verify
        callables during import, so builder-level gate replacement is both unreliable
        and broader than necessary. Production installs Ultra only at the two frozen
        sparse-specialist callsites below.
        """
        harness = _BASE_DIRECT_ADAPT(harness)
        print(
            f"[production-strategy] base_harness gate={getattr(harness.gate, 'model', 'unknown')} "
            "specialist_scope=frozen-repair+verify-callsites",
            flush=True,
        )
        return harness

    def _specialist(self) -> GigaChatUltraProvider:
        if self._specialist_provider is None:
            provider = GigaChatUltraProvider(role="sparse_semantic_specialist")
            if provider.model != self.specialist:
                raise RuntimeError(
                    f"production specialist invariant failed: expected={self.specialist!r} actual={provider.model!r}"
                )
            self._specialist_provider = provider
        return self._specialist_provider

    def specialist_repair(self, harness, targets, translated, memory, routes):
        """Run the frozen v9ah repair implementation with Ultra as its gate."""
        if self._base_deep_repair is None:
            raise RuntimeError("production specialist repair used before installation")
        provider = self._specialist()
        previous = harness.gate
        started = time.perf_counter()
        print(
            f"[production-specialist] phase=repair start routes={len(routes)} model={provider.model}",
            flush=True,
        )
        harness.gate = provider
        try:
            result = self._base_deep_repair(harness, targets, translated, memory, routes)
            changed, calls = result
            print(
                f"[production-specialist] phase=repair done routes={len(routes)} changed={len(changed)} "
                f"batches={calls} elapsed={time.perf_counter()-started:.2f}s",
                flush=True,
            )
            return result
        except BaseException as exc:
            print(
                f"[production-specialist] phase=repair failed routes={len(routes)} "
                f"elapsed={time.perf_counter()-started:.2f}s error={type(exc).__name__}: {str(exc)[:220]}",
                flush=True,
            )
            raise
        finally:
            harness.gate = previous

    def specialist_verify(self, harness, targets, translated, memory, routes):
        """Run the frozen v9ah final verifier with the same serialized Ultra client."""
        if self._base_deep_verify is None:
            raise RuntimeError("production specialist verify used before installation")
        provider = self._specialist()
        previous = harness.gate
        started = time.perf_counter()
        print(
            f"[production-specialist] phase=verify start routes={len(routes)} model={provider.model}",
            flush=True,
        )
        harness.gate = provider
        try:
            result = self._base_deep_verify(harness, targets, translated, memory, routes)
            changed, confirmed, calls = result
            print(
                f"[production-specialist] phase=verify done routes={len(routes)} changed={len(changed)} "
                f"confirmed={confirmed} batches={calls} elapsed={time.perf_counter()-started:.2f}s",
                flush=True,
            )
            return result
        except BaseException as exc:
            print(
                f"[production-specialist] phase=verify failed routes={len(routes)} "
                f"elapsed={time.perf_counter()-started:.2f}s error={type(exc).__name__}: {str(exc)[:220]}",
                flush=True,
            )
            raise
        finally:
            harness.gate = previous

    @staticmethod
    def _write_provenance(v3) -> str | None:
        """Persist source id/hash/context ids next to every production translation map."""
        map_path = getattr(v3, "MAP_JSON", None)
        if map_path is None or not map_path.exists():
            return None
        try:
            rows = json.loads(map_path.read_text("utf-8"))
        except Exception:
            return None
        if not isinstance(rows, list):
            return None

        usable = [row for row in rows if isinstance(row, dict) and str(row.get("id") or "")]
        artifact = []
        for index, row in enumerate(usable):
            sid = str(row.get("id") or "")
            source = str(row.get("source") or "")
            context_ids = [
                str(other.get("id") or "")
                for other in usable[max(0, index - 2) : index]
                + usable[index + 1 : index + 3]
                if str(other.get("id") or "")
            ]
            artifact.append(
                {
                    "source_id": sid,
                    "source_hash": source_fingerprint(source),
                    "context_ids": context_ids,
                    "stage": "production-final",
                }
            )

        name = map_path.stem.replace("-translation-map", "") + "-provenance.json"
        out = map_path.with_name(name)
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), "utf-8")
        return str(out)

    def annotate(self) -> None:
        v3 = legacy_kernel.v9ag.v9ad.v9ac.v9ab.v3
        provenance_file = self._write_provenance(v3)
        report = getattr(v3, "REPORT", None)
        if report is None or not report.exists():
            return
        try:
            data = json.loads(report.read_text("utf-8"))
        except Exception:
            return
        architecture = dict(data.get("architecture") or {})
        specialist_usage = dict(getattr(self._specialist_provider, "usage", {}) or {})
        architecture.update(
            {
                "production_strategy": self.name,
                "entrypoint": "scripts/production_literary_translation.py",
                "primary_translation": self.primary,
                "sparse_semantic_specialist": self.specialist,
                "specialist_installation": "frozen v9ah repair+verify callsites only",
                "specialist_usage": specialist_usage,
                "legacy_emergency_fallback": self.fallback,
                "guard_routing": ["polarity_scope", "source_contamination"],
                "guard_stats": dict(self._guard_stats),
                "provenance_artifact": provenance_file,
                "provenance_fields": ["source_id", "source_hash", "context_ids", "stage"],
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
        original_deep_repair = legacy_kernel._BASE_DEEP_REPAIR
        original_deep_verify = legacy_kernel._BASE_DEEP_VERIFY
        self._base_route = original_route
        self._base_deep_repair = original_deep_repair
        self._base_deep_verify = original_deep_verify

        # Keep old fallback transports intact, but install production policy exactly
        # where v9ah dereferences its already-frozen specialist functions.
        legacy_transport.adapt_harness = self.adapt_harness
        legacy_kernel._BASE_AB_ROUTE = self.guarded_route
        legacy_kernel._BASE_DEEP_REPAIR = self.specialist_repair
        legacy_kernel._BASE_DEEP_VERIFY = self.specialist_verify
        try:
            legacy_kernel.main()
        finally:
            legacy_kernel._BASE_DEEP_VERIFY = original_deep_verify
            legacy_kernel._BASE_DEEP_REPAIR = original_deep_repair
            legacy_kernel._BASE_AB_ROUTE = original_route
            legacy_transport.adapt_harness = original_adapt
            self.annotate()


def main() -> None:
    LiteraryTranslationStrategy().run()


if __name__ == "__main__":
    main()
