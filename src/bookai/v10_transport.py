from __future__ import annotations

import os
from typing import Any

from .models import BookMemory, Segment
from .v10 import GigaPrimaryTransport, _giga_json, _norm


class RobustTaggedPrimaryTransport(GigaPrimaryTransport):
    """Complete tagged transport with bounded Giga-only recovery.

    Normal path stays one tagged request per logical batch. If a response is
    truncated, missing ids are retried in fixed micro-batches (default 4), not as
    one large repeated batch and not through recursive split cascades. A tiny JSON
    fallback is allowed only for the final residual ids. DeepSeek is never used to
    recover missing primary translation.
    """

    name = "gigachat-3-lightning-v10-tagged-complete"

    def __init__(self) -> None:
        super().__init__()
        self.transport_stats: dict[str, Any] = {
            "logical_batches": 0,
            "tagged_calls": 0,
            "micro_recovery_calls": 0,
            "json_fallback_calls": 0,
            "first_pass_missing": 0,
            "final_missing": 0,
        }

    def _tagged(self, batch: list[Segment], memory: BookMemory, source_segments, *, retry: bool) -> dict[str, str]:
        self.transport_stats["tagged_calls"] += 1
        return self._call_tagged(batch, memory, source_segments, retry=retry)

    def _json_recover(self, batch: list[Segment], memory: BookMemory, source_segments) -> dict[str, str]:
        if not batch:
            return {}
        context = self._context_for_batch(batch, source_segments) or "нет"
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        system = """Recover ONLY the missing EN→RU literary translations below.
Return a complete faithful Russian translation for every id. Preserve every proposition,
actor/action/object relation, number, negation, chronology, technical denotation and name.
No commentary. ONLY JSON {"items":[{"id":"s000001","ru":"..."}]} with exactly one row per supplied id."""
        payload = {
            "context_only": context,
            "glossary": glossary,
            "characters": characters,
            "items": [{"id": s.id, "source": s.text} for s in batch],
        }
        self.transport_stats["json_fallback_calls"] += 1
        try:
            obj = _giga_json(self, system, payload, max_tokens=3600)
        except Exception as exc:
            print(f"[v10-primary-json-recovery] error={type(exc).__name__} segments={len(batch)}", flush=True)
            return {}
        expected = {s.id for s in batch}
        out: dict[str, str] = {}
        for row in obj.get("items") or []:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "")
            ru = _norm(row.get("ru") or "")
            if sid in expected and ru:
                out[sid] = ru
        return out

    def translate_many(self, segments: list[Segment], memory: BookMemory, *, source_segments: list[Segment] | None = None) -> tuple[dict[str, str], dict[str, str]]:
        result: dict[str, str] = {}
        regular: list[Segment] = []
        for segment in segments:
            heading = self._deterministic_heading(segment)
            if heading:
                result[segment.id] = heading
            else:
                regular.append(segment)

        micro = max(2, min(6, int(os.getenv("BOOKAI_V10_RECOVERY_BATCH") or "4")))
        json_batch = max(1, min(4, int(os.getenv("BOOKAI_V10_JSON_RECOVERY_BATCH") or "3")))

        for batch in self._batches(regular):
            self.transport_stats["logical_batches"] += 1
            try:
                rows = self._tagged(batch, memory, source_segments, retry=False)
            except Exception as exc:
                rows = {}
                print(f"[v10-primary] batch_error={type(exc).__name__} segments={len(batch)}", flush=True)
            result.update(rows)

            missing = [s for s in batch if s.id not in result]
            self.transport_stats["first_pass_missing"] += len(missing)
            for start in range(0, len(missing), micro):
                part = missing[start:start + micro]
                self.transport_stats["micro_recovery_calls"] += 1
                try:
                    result.update(self._tagged(part, memory, source_segments, retry=True))
                except Exception as exc:
                    print(f"[v10-primary] micro_retry_error={type(exc).__name__} segments={len(part)}", flush=True)

            residual = [s for s in batch if s.id not in result]
            for start in range(0, len(residual), json_batch):
                result.update(self._json_recover(residual[start:start + json_batch], memory, source_segments))

        final_missing = [s for s in regular if s.id not in result]
        # Last bounded Giga-only single-id recovery. This is intentionally rare;
        # it guarantees DeepSeek never receives an untranslated primary row.
        for segment in final_missing:
            result.update(self._json_recover([segment], memory, source_segments))

        final_missing = [s for s in regular if s.id not in result]
        self.transport_stats["final_missing"] = len(final_missing)
        errors = {s.id: "missing after bounded Giga-only recovery" for s in final_missing}
        print("[v10-primary-transport] " + __import__("json").dumps(self.transport_stats, ensure_ascii=False), flush=True)
        return result, errors
