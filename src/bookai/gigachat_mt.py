from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .models import BookMemory, Segment


@dataclass(slots=True)
class GigaChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    def add(self, usage: dict[str, int], calls: int = 1) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.api_calls += int(calls)

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
        }


class GigaChatLightningBackend:
    """Single-stream GigaChat-3-Lightning bulk EN→RU literary translator.

    Personal GigaChat API access is single-stream, so this backend batches multiple
    source segments into one request and recursively splits only malformed batches.
    It never silently falls back to another bulk model.
    """

    name = "gigachat-3-lightning"

    def __init__(self) -> None:
        self.credentials = (os.getenv("GIGACHAT_AUTH_KEY") or "").strip()
        self.scope = (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
        self.base_url = (os.getenv("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1").strip()
        self.model = (os.getenv("BOOKAI_GIGACHAT_MODEL") or "GigaChat-3-Lightning").strip()
        self.max_batch_segments = max(1, int(os.getenv("BOOKAI_GIGACHAT_BATCH_SEGMENTS") or "8"))
        self.max_batch_chars = max(1000, int(os.getenv("BOOKAI_GIGACHAT_BATCH_CHARS") or "12000"))
        self.max_split_depth = max(1, int(os.getenv("BOOKAI_GIGACHAT_SPLIT_DEPTH") or "5"))
        self.max_tokens = max(1200, int(os.getenv("BOOKAI_GIGACHAT_MAX_TOKENS") or "7000"))
        self._client = None
        self.usage = GigaChatUsage()

    @property
    def backend_name(self) -> str:
        return self.name

    def available(self) -> bool:
        return bool(self.credentials)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.credentials:
            raise RuntimeError("GIGACHAT_AUTH_KEY is not configured")
        from gigachat import GigaChat

        client = GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            base_url=self.base_url,
            verify_ssl_certs=False,
            timeout=180,
            max_retries=7,
            retry_backoff_factor=1.4,
        )
        token = client.get_token()
        if not str(getattr(token, "access_token", "") or ""):
            raise RuntimeError("GigaChat OAuth succeeded but access_token is empty")
        available = []
        try:
            models = client.get_models()
            for row in getattr(models, "data", []) or []:
                name = getattr(row, "id_", None) or getattr(row, "id", None) or getattr(row, "name", None)
                if name:
                    available.append(str(name))
        except Exception:
            available = []
        if available and self.model not in available:
            raise RuntimeError(f"Requested GigaChat model {self.model!r} is unavailable; available={available}")
        self._client = client
        return client

    def _batches(self, segments: list[Segment]) -> list[list[Segment]]:
        batches: list[list[Segment]] = []
        current: list[Segment] = []
        chars = 0
        for segment in segments:
            size = len(segment.text)
            if current and (
                len(current) >= self.max_batch_segments
                or chars + size > self.max_batch_chars
            ):
                batches.append(current)
                current = []
                chars = 0
            current.append(segment)
            chars += size
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, str]:
        raw = text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(raw[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("GigaChat response is not a JSON object")

        # Lightning occasionally decorates a requested id despite the prompt,
        # e.g. "[s000010]" or "s000010:". Normalize those harmless variants so
        # a valid translation is not discarded as a missing segment.
        normalized: dict[str, str] = {}
        for key, val in value.items():
            candidate = str(val).strip()
            if not candidate:
                continue
            raw_key = str(key).strip()
            match = re.search(r"s\d{6}", raw_key, flags=re.I)
            normalized_key = match.group(0).lower() if match else raw_key
            normalized[normalized_key] = candidate
        return normalized

    @staticmethod
    def _relevant_glossary(batch: list[Segment], memory: BookMemory) -> str:
        source = "\n".join(segment.text for segment in batch).casefold()
        rows: list[str] = []
        for key, value in memory.glossary.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text and key_text.casefold() in source:
                rows.append(f"{key_text}={value_text}")
        return "; ".join(rows[:40])

    def _prompt(self, batch: list[Segment], memory: BookMemory) -> str:
        source = "\n\n".join(f"[{segment.id}]\n{segment.text}" for segment in batch)
        ids = ", ".join(segment.id for segment in batch)
        style = memory.style
        glossary = self._relevant_glossary(batch, memory) or "нет терминов в этом батче"
        continuity = (memory.rolling_summary or "").strip()
        if len(continuity) > 5000:
            continuity = continuity[-5000:]
        return f"""Переведи художественные фрагменты с английского на русский.

ОБЯЗАТЕЛЬНО:
- сохрани весь смысл каждого фрагмента без пропусков, сокращений, перестановки фактов и добавлений;
- русский должен звучать как естественная опубликованная проза, а не машинный перевод;
- не меняй субъект действия, родство, национальность/происхождение, количество, отрицание, временные отношения и технические детали;
- сохраняй устройство длинных предложений, если оно естественно по-русски;
- диалоги делай живыми и точными, без канцелярита;
- не объясняй текст читателю и не усиливай эмоции.

СТИЛЬ КНИГИ:
Narrative voice: {style.narrative_voice}
Rhythm: {style.rhythm}
Dialogue: {style.dialogue}
Humor: {style.humor}

КОНТЕКСТ НЕПРЕРЫВНОСТИ:
{continuity or 'нет дополнительного контекста'}

ЗАКРЕПЛЁННЫЕ ТЕРМИНЫ:
{glossary}

Верни ТОЛЬКО валидный JSON-объект без Markdown. Ключами должны быть ровно ID фрагментов ({ids}), значениями — полные русские переводы. Не объединяй фрагменты.

SOURCE:
{source}"""

    def _translate_batch(self, batch: list[Segment], memory: BookMemory) -> tuple[dict[str, str], dict[str, int]]:
        client = self._ensure_client()
        response = client.chat(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты профессиональный литературный переводчик с английского на русский. "
                            "Главные приоритеты: точность смысла, полнота, естественная русская проза "
                            "и сохранение авторского тона. Строго соблюдай JSON-формат."
                        ),
                    },
                    {"role": "user", "content": self._prompt(batch, memory)},
                ],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": self.max_tokens,
            }
        )
        parsed = self._parse_json_object(str(response.choices[0].message.content or ""))
        expected = {segment.id for segment in batch}
        translated = {sid: text for sid, text in parsed.items() if sid in expected and text}
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }
        return translated, usage

    def _translate_resilient(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        depth: int = 0,
    ) -> tuple[dict[str, str], dict[str, str]]:
        try:
            rows, usage = self._translate_batch(batch, memory)
            self.usage.add(usage, calls=1)
        except Exception as exc:
            self.usage.add({}, calls=1)
            if len(batch) > 1 and depth < self.max_split_depth:
                mid = max(1, len(batch) // 2)
                left_rows, left_errors = self._translate_resilient(batch[:mid], memory, depth=depth + 1)
                right_rows, right_errors = self._translate_resilient(batch[mid:], memory, depth=depth + 1)
                return {**left_rows, **right_rows}, {**left_errors, **right_errors}
            return {}, {segment.id: f"{type(exc).__name__}: {exc}" for segment in batch}

        missing = [segment for segment in batch if segment.id not in rows]
        errors: dict[str, str] = {}
        if missing and depth < self.max_split_depth:
            retry_rows, retry_errors = self._translate_resilient(missing, memory, depth=depth + 1)
            rows.update(retry_rows)
            errors.update(retry_errors)
        else:
            for segment in missing:
                errors[segment.id] = "missing from GigaChat batch response"
        return rows, errors

    def translate_many(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        del source_segments
        if not segments:
            return {}, {}
        if not self.available():
            return {}, {segment.id: "GIGACHAT_AUTH_KEY is not configured" for segment in segments}

        out: dict[str, str] = {}
        errors: dict[str, str] = {}
        for batch in self._batches(segments):
            batch_rows, batch_errors = self._translate_resilient(batch, memory)
            out.update(batch_rows)
            errors.update(batch_errors)
        return out, errors


def benchmark_gigachat(
    client: GigaChatLightningBackend,
    segments: list[Segment],
    memory: BookMemory,
    *,
    source_segments: list[Segment] | None = None,
    total_source_chars: int,
) -> dict:
    started = time.perf_counter()
    before_calls = client.usage.api_calls
    translated, errors = client.translate_many(segments, memory, source_segments=source_segments)
    elapsed = max(0.001, time.perf_counter() - started)
    successful_chars = sum(len(segment.text) for segment in segments if segment.id in translated)
    cps = successful_chars / elapsed
    estimate = None if cps <= 0 else total_source_chars / cps * 1.15
    return {
        "probe_segments": len(segments),
        "probe_success": len(translated),
        "probe_errors": errors,
        "source_chars": successful_chars,
        "elapsed_seconds": round(elapsed, 3),
        "source_chars_per_second": round(cps, 3),
        "estimated_full_seconds": None if estimate is None else round(estimate, 1),
        "estimated_full_minutes": None if estimate is None else round(estimate / 60.0, 2),
        "backend": client.backend_name,
        "model": client.model,
        "scope": client.scope,
        "single_stream": True,
        "probe_api_calls": client.usage.api_calls - before_calls,
        "token_usage": client.usage.as_dict(),
    }