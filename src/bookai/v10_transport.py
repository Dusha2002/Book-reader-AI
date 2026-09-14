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
    "VOICE:", "RHYTHM:", "DIALOGUE:", "HUMOR:", "ACRONYM_CANON:",
    "Глоссарий:", "Источник:",
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


def _style_payload(memory: BookMemory) -> dict[str, str]:
    style = memory.style
    return {
        "voice": str(style.narrative_voice or ""),
        "rhythm": str(style.rhythm or ""),
        "dialogue": str(style.dialogue or ""),
        "humor": str(style.humor or ""),
    }


def _acronym_canon(memory: BookMemory) -> str:
    rows = [f"{src}→{dst}" for src, dst in sorted(memory.acronyms.items()) if src and dst]
    return "; ".join(rows) if rows else "нет"


class RobustTaggedPrimaryTransport(GigaPrimaryTransport):
    """Complete tagged transport with bounded, domain-aware Giga-only recovery.

    Opening tags themselves are reliable boundaries. If a block forgets its closing
    `</s>` but the next opening `<s id=...>` is present, the body can be salvaged up
    to that boundary without swallowing the neighbor. Only the final unclosed block
    remains missing/recoverable. Prompt/protocol residue is still rejected.
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
            "boundary_salvage": True,
            "domain_aware_prompts": True,
        }

    @staticmethod
    def parse_tagged(text: str, expected: set[str]) -> dict[str, str]:
        """Parse independently bounded blocks without allowing cross-id swallowing.

        Preferred form is `<s id=...>...</s>`. When the close tag is missing but a
        subsequent opening tag exists, that next opening tag safely terminates the
        current body. We deliberately do NOT salvage the final unclosed block because
        response truncation there is indistinguishable from an incomplete translation.
        """
        raw = str(text or "")
        starts = list(_START_TAG_RE.finditer(raw))
        out: dict[str, str] = {}
        for pos, match in enumerate(starts):
            sid = match.group(1).casefold()
            if sid not in expected:
                continue
            body_start = match.end()
            has_next = pos + 1 < len(starts)
            next_start = starts[pos + 1].start() if has_next else len(raw)
            close = _CLOSE_TAG_RE.search(raw, body_start, next_start)
            if close is not None:
                body_end = close.start()
            elif has_next:
                body_end = next_start
            else:
                continue
            value = raw[body_start:body_end].strip()
            value = re.sub(r"\s*</?s\s*>\s*$", "", value, flags=re.I).strip()
            if not value or _ANY_S_TAG_RE.search(value) or _looks_like_prompt_leak(value):
                continue
            out[sid] = value
        return out

    def _tag_prompt(
        self,
        batch: list[Segment],
        memory: BookMemory,
        source_segments: list[Segment] | None,
        *,
        retry: bool = False,
    ) -> str:
        style = memory.style
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        context = self._context_for_batch(batch, source_segments) or "нет"
        acronyms = _acronym_canon(memory)
        targets = "\n".join(f'<src id="{s.id}">{s.text}</src>' for s in batch)
        retry_note = "Это повтор только пропущенных ID; верни КАЖДЫЙ указанный ID." if retry else ""
        return f"""Профессиональный перевод книги EN→RU. Переведи только SRC-блоки.
Не предполагай, что книга художественная, учебная или научная: регистр и жанр определяй по STYLE и CONTEXT_ONLY.
Не сокращай, не пересказывай и не добавляй факты. Сохраняй субъект/объект действия, числа, отрицания, причинность,
хронологию, терминологическую широту, технический смысл, имена, формулы, обозначения и библиографические ссылки.
Для художественного текста сохраняй голос, ритм, иронию и естественный диалог; для академического/технического —
принятую русскую терминологию, точность категорий, нотацию и структуру аргумента.
Если английское предложение продолжается в соседнем сегменте, не закрывай его точкой и не превращай фрагмент
в отдельное предложение: сохрани открытый синтаксис и естественную пунктуационную связь.

VOICE: {style.narrative_voice}
RHYTHM: {style.rhythm}
DIALOGUE: {style.dialogue}
HUMOR: {style.humor}
CHARACTERS: {characters}
GLOSSARY: {glossary}
ACRONYM_CANON: {acronyms}
CONTEXT_ONLY: {context}
{retry_note}

TARGETS:
{targets}

ФОРМАТ: ровно по одному блоку на каждый id, без JSON и комментариев:
<s id="s000001">полный русский перевод</s>"""

    def _call_tagged(
        self,
        batch: list[Segment],
        memory: BookMemory,
        source_segments: list[Segment] | None,
        *,
        retry: bool = False,
    ) -> dict[str, str]:
        client = self._ensure_client()
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты точный профессиональный переводчик книг EN→RU для любых жанров и предметных областей. "
                        "Следуй переданному профилю текста; не навязывай художественный стиль техническому тексту и наоборот. "
                        "Не выводи ничего кроме требуемых <s id=...>...</s> блоков."
                    ),
                },
                {"role": "user", "content": self._tag_prompt(batch, memory, source_segments, retry=retry)},
            ],
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
        }
        response = client.chat(request)
        self.usage.add(_usage(response), calls=1)
        expected = {s.id for s in batch}
        return self.parse_tagged(str(response.choices[0].message.content or ""), expected)

    def _tagged(self, batch: list[Segment], memory: BookMemory, source_segments, *, retry: bool) -> dict[str, str]:
        self.transport_stats["tagged_calls"] += 1
        return self._call_tagged(batch, memory, source_segments, retry=retry)

    def _json_recover(self, batch: list[Segment], memory: BookMemory, source_segments) -> dict[str, str]:
        if not batch:
            return {}
        context = self._context_for_batch(batch, source_segments) or "нет"
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        system = """Recover ONLY the missing EN→RU book-translation rows below.
The source may be literary fiction, narrative nonfiction, academic/technical prose or another book genre.
Infer and preserve its register from the supplied style/context; do not force a literary voice onto technical prose.
Return a complete faithful Russian translation for every id. Preserve every proposition, actor/action/object relation,
number, negation, chronology, category breadth, technical denotation, acronym policy, citation and name.
If a source sentence continues across a segment boundary, preserve that open syntax instead of closing it early.
No commentary. ONLY JSON {"items":[{"id":"s000001","ru":"..."}]} with exactly one row per supplied id."""
        payload = {
            "style": _style_payload(memory),
            "context_only": context,
            "glossary": glossary,
            "acronym_canon": dict(memory.acronyms),
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
        style = _style_payload(memory)
        acronyms = _acronym_canon(memory)
        client = self._ensure_client()
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Переведи один английский фрагмент книги на русский в регистре и предметной области исходника. "
                        "Не считай текст художественным по умолчанию. Для технического/академического текста используй "
                        "принятую русскую терминологию и сохраняй нотацию; для художественного — авторский голос. "
                        "Если предложение продолжается в соседнем сегменте, не закрывай его искусственно. "
                        "Верни ТОЛЬКО полный готовый русский перевод без JSON, тегов, комментариев, "
                        "пометок 'перевод:' и альтернатив. Ничего не сокращай и не добавляй."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"STYLE: {style}\nCONTEXT_ONLY: {context}\nCHARACTERS: {characters}\n"
                        f"GLOSSARY: {glossary}\nACRONYM_CANON: {acronyms}\n\nSOURCE:\n{segment.text}"
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