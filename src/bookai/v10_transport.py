from __future__ import annotations

import os
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import GigaPrimaryTransport, _giga_json, _norm, _usage


_START_TAG_RE = re.compile(r"<s\s+id=[\"']?(s\d{6})[\"']?\s*>", re.I)
_CLOSE_TAG_RE = re.compile(r"</s\s*>", re.I)
_ANY_S_TAG_RE = re.compile(r"</?s(?:\s|>)", re.I)
_PROMPT_LABELS = (
    "CONTEXT_ONLY:", "CHARACTERS:", "GLOSSARY:", "SOURCE:", "TARGETS:",
    "VOICE:", "RHYTHM:", "DIALOGUE:", "HUMOR:", "Глоссарий:", "Источник:",
)


def _looks_like_prompt_leak(value: str) -> bool:
    text = str(value or "")
    if not text:
        return False
    if re.search(r"\bru\s*=\s*[^;\n]{0,100};\s*gender\s*=", text, re.I):
        return True
    if re.search(r"<src\s+id=", text, re.I):
        return True
    labels = sum(label.casefold() in text.casefold() for label in _PROMPT_LABELS)
    return labels >= 2


class RobustTaggedPrimaryTransport(GigaPrimaryTransport):
    """Complete tagged transport with bounded Giga-only recovery.

    A malformed tagged response must never contaminate a neighboring segment. Each
    block is accepted only when its closing </s> occurs before the next opening <s>.
    Corrupt/truncated ids and prompt echoes are treated as missing and recovered
    with Giga. DeepSeek is never a transport fallback.
    """

    name = "gigachat-3-lightning-v10-tagged-complete"

    def __init__(self) -> None:
        super().__init__()
        self.transport_stats: dict[str, Any] = {
            "logical_batches": 0,
            "tagged_calls": 0,
            "micro_recovery_calls": 0,
            "json_fallback_calls": 0,
            "plain_single_calls": 0,
            "first_pass_missing": 0,
            "corrupt_tag_blocks": 0,
            "final_missing": 0,
        }

    @staticmethod
    def parse_tagged(text: str, expected: set[str]) -> dict[str, str]:
        """Parse only independently well-formed blocks; never span into a next block."""
        raw = str(text or "")
        starts = list(_START_TAG_RE.finditer(raw))
        out: dict[str, str] = {}
        for pos, match in enumerate(starts):
            sid = match.group(1).casefold()
            if sid not in expected:
                continue
            body_start = match.end()
            next_start = starts[pos + 1].start() if pos + 1 < len(starts) else len(raw)
            close = _CLOSE_TAG_RE.search(raw, body_start, next_start)
            if close is None:
                continue
            value = raw[body_start:close.start()].strip()
            # Reject protocol residue and prompt echoes even if the outer block closed.
            if not value or _ANY_S_TAG_RE.search(value) or _looks_like_prompt_leak(value):
                continue
            out[sid] = value
        return out

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
            if sid in expected and ru and not _ANY_S_TAG_RE.search(ru) and not _looks_like_prompt_leak(ru):
                out[sid] = ru
        return out

    def _plain_single_recover(self, segment: Segment, memory: BookMemory, source_segments) -> str:
        context = self._context_for_batch([segment], source_segments) or "нет"
        glossary = self._relevant_glossary([segment], memory) or "нет"
        characters = self._relevant_characters([segment], memory) or "нет"
        client = self._ensure_client()
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Переведи один английский фрагмент литературной прозы на русский. "
                        "Верни ТОЛЬКО полный готовый русский перевод без JSON, тегов, комментариев, "
                        "пометок 'перевод:' и альтернатив. Ничего не сокращай и не добавляй."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT_ONLY: {context}\nCHARACTERS: {characters}\nGLOSSARY: {glossary}\n\n"
                        f"SOURCE:\n{segment.text}"
                    ),
                },
            ],
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": max(1200, min(5000, int(self.max_tokens))),
        }
        self.transport_stats["plain_single_calls"] += 1
        try:
            response = client.chat(request)
            self.usage.add(_usage(response), calls=1)
        except Exception as exc:
            print(f"[v10-primary-plain-single] id={segment.id} error={type(exc).__name__}", flush=True)
            return ""
        text = str(response.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^\s*(?:перевод|translation)\s*:\s*", "", text, flags=re.I)
        if _ANY_S_TAG_RE.search(text) or _looks_like_prompt_leak(text):
            return ""
        return text.strip()

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
        for segment in final_missing:
            candidate = self._plain_single_recover(segment, memory, source_segments)
            if candidate:
                result[segment.id] = candidate

        final_missing = [s for s in regular if s.id not in result]
        self.transport_stats["final_missing"] = len(final_missing)
        errors = {s.id: "missing after bounded Giga-only recovery" for s in final_missing}
        print("[v10-primary-transport] " + __import__("json").dumps(self.transport_stats, ensure_ascii=False), flush=True)
        return result, errors
