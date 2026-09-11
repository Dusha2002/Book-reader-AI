from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .models import BookMemory, Segment
from .quality import is_heading


_CHAPTER_RU = {
    "one": "один", "two": "два", "three": "три", "four": "четыре", "five": "пять",
    "six": "шесть", "seven": "семь", "eight": "восемь", "nine": "девять", "ten": "десять",
    "eleven": "одиннадцать", "twelve": "двенадцать", "thirteen": "тринадцать",
    "fourteen": "четырнадцать", "fifteen": "пятнадцать", "sixteen": "шестнадцать",
    "seventeen": "семнадцать", "eighteen": "восемнадцать", "nineteen": "девятнадцать",
    "twenty": "двадцать", "twenty-one": "двадцать один", "twenty one": "двадцать один",
    "twenty-two": "двадцать два", "twenty two": "двадцать два",
    "twenty-three": "двадцать три", "twenty three": "двадцать три",
    "twenty-four": "двадцать четыре", "twenty four": "двадцать четыре",
    "twenty-five": "двадцать пять", "twenty five": "двадцать пять",
    "twenty-six": "двадцать шесть", "twenty six": "двадцать шесть",
    "twenty-seven": "двадцать семь", "twenty seven": "двадцать семь",
    "twenty-eight": "двадцать восемь", "twenty eight": "двадцать восемь",
    "twenty-nine": "двадцать девять", "twenty nine": "двадцать девять",
    "thirty": "тридцать",
}


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
    """Single-stream GigaChat-3-Lightning EN→RU literary translator.

    PERS access is single-stream, so throughput comes from batching. Requests are
    intentionally fail-fast: one upstream stall must never freeze a whole book job.
    Strict JSON Schema is preferred; schema incompatibility falls back once to a
    plain JSON contract, while rate-limit/transport errors remain retryable failures.
    """

    name = "gigachat-3-lightning"

    def __init__(self) -> None:
        self.credentials = (os.getenv("GIGACHAT_AUTH_KEY") or "").strip()
        self.scope = (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
        self.base_url = (os.getenv("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1").strip()
        self.model = (os.getenv("BOOKAI_GIGACHAT_MODEL") or "GigaChat-3-Lightning").strip()
        self.max_batch_segments = max(1, int(os.getenv("BOOKAI_GIGACHAT_BATCH_SEGMENTS") or "12"))
        self.max_batch_chars = max(1000, int(os.getenv("BOOKAI_GIGACHAT_BATCH_CHARS") or "10000"))
        self.max_split_depth = max(1, int(os.getenv("BOOKAI_GIGACHAT_SPLIT_DEPTH") or "4"))
        self.max_tokens = max(1200, int(os.getenv("BOOKAI_GIGACHAT_MAX_TOKENS") or "6000"))
        self.timeout_seconds = max(15, min(90, int(os.getenv("BOOKAI_GIGACHAT_TIMEOUT") or "55")))
        self.max_retries = max(0, min(3, int(os.getenv("BOOKAI_GIGACHAT_RETRIES") or "1")))
        self.retry_backoff = max(0.2, min(3.0, float(os.getenv("BOOKAI_GIGACHAT_RETRY_BACKOFF") or "0.8")))
        self.validate_model = (os.getenv("BOOKAI_GIGACHAT_VALIDATE_MODEL") or "false").lower() in {"1", "true", "yes"}
        self._client = None
        self._request_seq = 0
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

        print(
            f"[gigachat-client] oauth model={self.model} timeout={self.timeout_seconds}s retries={self.max_retries}",
            flush=True,
        )
        started = time.perf_counter()
        client = GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            base_url=self.base_url,
            verify_ssl_certs=False,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_factor=self.retry_backoff,
        )
        token = client.get_token()
        if not str(getattr(token, "access_token", "") or ""):
            raise RuntimeError("GigaChat OAuth succeeded but access_token is empty")
        print(f"[gigachat-client] oauth_ok elapsed={time.perf_counter()-started:.2f}s", flush=True)

        if self.validate_model:
            available = []
            try:
                models = client.get_models()
                for row in getattr(models, "data", []) or []:
                    name = getattr(row, "id_", None) or getattr(row, "id", None) or getattr(row, "name", None)
                    if name:
                        available.append(str(name))
            except Exception as exc:
                print(f"[gigachat-client] model_list_skipped error={type(exc).__name__}: {exc}", flush=True)
            if available and self.model not in available:
                raise RuntimeError(f"Requested GigaChat model {self.model!r} is unavailable; available={available}")
        self._client = client
        return client

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _deterministic_heading(segment: Segment) -> str | None:
        if not is_heading(segment):
            return None
        match = re.fullmatch(r"\s*Chapter\s+(.+?)\s*", segment.text, flags=re.I)
        if not match:
            return None
        raw = " ".join(match.group(1).strip().casefold().split())
        if raw.isdigit():
            return f"Глава {raw}"
        translated = _CHAPTER_RU.get(raw)
        return f"Глава {translated}" if translated else None

    def _batches(self, segments: list[Segment]) -> list[list[Segment]]:
        batches: list[list[Segment]] = []
        current: list[Segment] = []
        chars = 0
        for segment in segments:
            size = len(segment.text)
            if current and (len(current) >= self.max_batch_segments or chars + size > self.max_batch_chars):
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
    def _usage(response) -> dict[str, int]:
        usage_obj = getattr(response, "usage", None)
        return {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }

    @staticmethod
    def _strict_response_format(batch: list[Segment]) -> dict:
        properties = {
            segment.id: {
                "type": "string",
                "minLength": 1,
                "description": "Полный русский перевод только этого TARGET-фрагмента",
            }
            for segment in batch
        }
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": [segment.id for segment in batch],
                "additionalProperties": False,
            },
            "strict": True,
        }

    @staticmethod
    def _relevant_glossary(batch: list[Segment], memory: BookMemory) -> str:
        source = "\n".join(segment.text for segment in batch).casefold()
        rows: list[str] = []
        for key, value in memory.glossary.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text and key_text.casefold() in source:
                rows.append(f"{key_text}={value_text}")
        return "; ".join(rows[:50])

    @staticmethod
    def _relevant_characters(batch: list[Segment], memory: BookMemory) -> str:
        source = "\n".join(segment.text for segment in batch).casefold()
        rows: list[str] = []
        for name, description in memory.characters.items():
            if name and name.casefold() in source and description:
                rows.append(f"{name}: {description}")
        return "; ".join(rows[:20])

    def _context_for_batch(self, batch: list[Segment], source_segments: list[Segment] | None) -> str:
        if not source_segments or not batch:
            return ""
        by_id = {segment.id: index for index, segment in enumerate(source_segments)}
        indexes = [by_id[s.id] for s in batch if s.id in by_id]
        if not indexes:
            return ""
        lo, hi = min(indexes), max(indexes)
        rows: list[str] = []
        if lo > 0:
            rows.append("BEFORE: " + self._clip(source_segments[lo - 1].text, 900))
        if hi + 1 < len(source_segments):
            rows.append("AFTER: " + self._clip(source_segments[hi + 1].text, 900))
        return "\n".join(rows)

    def _prompt(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        minimal: bool = False,
    ) -> str:
        style = memory.style
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        nearest = self._context_for_batch(batch, source_segments)
        summary = "" if minimal else self._clip(memory.rolling_summary, 1200)
        targets = {segment.id: segment.text for segment in batch}
        style_rows = "" if minimal else (
            f"VOICE: {self._clip(style.narrative_voice, 1400)}\n"
            f"RHYTHM: {self._clip(style.rhythm, 700)}\n"
            f"DIALOGUE: {self._clip(style.dialogue, 500)}\n"
            f"HUMOR: {self._clip(style.humor, 500)}"
        )
        return f"""Задача: профессиональный литературный перевод английского текста на русский.

КРИТИЧЕСКИЙ КОНТРАКТ:
1. Переводи ТОЛЬКО значения внутри блока TARGETS. CONTEXT_ONLY и BOOK_MEMORY нужны только для понимания и НИКОГДА не должны попадать в перевод.
2. Для каждого ID верни перевод ровно соответствующего исходного фрагмента. Не пересказывай соседние события и не дописывай контекст.
3. Короткая реплика должна оставаться короткой. Не расширяй одно слово или вопрос в абзац.
4. Не меняй субъект действия, пол персонажа, родство, национальность, числа, отрицания, временные и причинно-следственные связи.
5. Не вставляй инструкции, комментарии переводчика, содержимое глоссария или служебные формулировки.
6. Русский — естественная опубликованная проза: точная, сдержанная, сухо-ироничная, без кальки и канцелярита.

{style_rows}

BOOK_MEMORY (НЕ ПЕРЕВОДИТЬ):
{summary or 'нет'}

CHARACTERS (ОБЯЗАТЕЛЬНАЯ КОНСИСТЕНТНОСТЬ):
{characters}

GLOSSARY (ОБЯЗАТЕЛЬНАЯ КОНСИСТЕНТНОСТЬ):
{glossary}

CONTEXT_ONLY (НЕ ПЕРЕВОДИТЬ):
{nearest or 'нет'}

TARGETS (ПЕРЕВЕСТИ):
{json.dumps(targets, ensure_ascii=False)}

Ответ обязан соответствовать переданной JSON Schema."""

    @staticmethod
    def _schema_incompatibility(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".casefold()
        return any(token in text for token in ("response_format", "json_schema", "schema", "400", "422", "validation"))

    def _chat(self, payload: dict, batch: list[Segment], *, strict: bool) -> object:
        client = self._ensure_client()
        self._request_seq += 1
        seq = self._request_seq
        chars = sum(len(s.text) for s in batch)
        first = batch[0].id if batch else "-"
        last = batch[-1].id if batch else "-"
        print(
            f"[gigachat-request] seq={seq} strict={str(strict).lower()} segments={len(batch)} chars={chars} ids={first}..{last}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            response = client.chat(payload)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[gigachat-request] seq={seq} error={type(exc).__name__} elapsed={elapsed:.2f}s detail={str(exc)[:240]}",
                flush=True,
            )
            raise
        usage = self._usage(response)
        print(
            f"[gigachat-request] seq={seq} ok elapsed={time.perf_counter()-started:.2f}s tokens={usage.get('total_tokens', 0)}",
            flush=True,
        )
        return response

    def _translate_batch(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        minimal: bool = False,
    ) -> tuple[dict[str, str], dict[str, int]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты литературный переводчик EN→RU. Переводи только TARGETS, "
                    "никогда не переводя CONTEXT_ONLY. Не добавляй ничего от себя."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(batch, memory, source_segments=source_segments, minimal=minimal),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "response_format": self._strict_response_format(batch),
        }
        try:
            response = self._chat(payload, batch, strict=True)
        except Exception as exc:
            if not self._schema_incompatibility(exc):
                raise
            print("[gigachat-schema] strict format rejected; retrying once with plain JSON contract", flush=True)
            fallback = dict(payload)
            fallback.pop("response_format", None)
            fallback["messages"] = [dict(messages[0]), dict(messages[1])]
            fallback["messages"][0]["content"] += " Верни только валидный JSON-объект с ровно требуемыми ID."
            response = self._chat(fallback, batch, strict=False)
        parsed = self._parse_json_object(str(response.choices[0].message.content or ""))
        expected = {segment.id for segment in batch}
        translated = {sid: text for sid, text in parsed.items() if sid in expected and text}
        return translated, self._usage(response)

    def _single_strict_recovery(
        self,
        segment: Segment,
        memory: BookMemory,
        source_segments: list[Segment] | None,
        original_error: Exception | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        try:
            rows, usage = self._translate_batch([segment], memory, source_segments=source_segments, minimal=True)
            self.usage.add(usage, calls=1)
            if segment.id not in rows:
                raise ValueError("strict single response omitted target id")
            return rows, {}
        except Exception as exc:
            self.usage.add({}, calls=1)
            prefix = f"{type(original_error).__name__}: {original_error}; " if original_error else ""
            return {}, {segment.id: prefix + f"single recovery {type(exc).__name__}: {exc}"}

    def _translate_resilient(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None,
        depth: int = 0,
    ) -> tuple[dict[str, str], dict[str, str]]:
        try:
            rows, usage = self._translate_batch(batch, memory, source_segments=source_segments)
            self.usage.add(usage, calls=1)
        except Exception as exc:
            self.usage.add({}, calls=1)
            if len(batch) == 1:
                return self._single_strict_recovery(batch[0], memory, source_segments, exc)
            if depth < self.max_split_depth:
                mid = max(1, len(batch) // 2)
                left_rows, left_errors = self._translate_resilient(batch[:mid], memory, source_segments=source_segments, depth=depth + 1)
                right_rows, right_errors = self._translate_resilient(batch[mid:], memory, source_segments=source_segments, depth=depth + 1)
                return {**left_rows, **right_rows}, {**left_errors, **right_errors}
            return {}, {segment.id: f"{type(exc).__name__}: {exc}" for segment in batch}

        missing = [segment for segment in batch if segment.id not in rows]
        if not missing:
            return rows, {}
        if len(batch) == 1:
            retry_rows, retry_errors = self._single_strict_recovery(batch[0], memory, source_segments)
            rows.update(retry_rows)
            return rows, retry_errors
        if depth < self.max_split_depth:
            retry_rows, retry_errors = self._translate_resilient(missing, memory, source_segments=source_segments, depth=depth + 1)
            rows.update(retry_rows)
            return rows, retry_errors
        return rows, {segment.id: "missing from GigaChat batch response" for segment in missing}

    def translate_many(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if not segments:
            return {}, {}
        if not self.available():
            return {}, {segment.id: "GIGACHAT_AUTH_KEY is not configured" for segment in segments}

        out: dict[str, str] = {}
        errors: dict[str, str] = {}
        llm_segments: list[Segment] = []
        for segment in segments:
            heading = self._deterministic_heading(segment)
            if heading is not None:
                out[segment.id] = heading
            else:
                llm_segments.append(segment)

        batches = self._batches(llm_segments)
        print(
            f"[gigachat-batches] segments={len(llm_segments)} batches={len(batches)} max_segments={self.max_batch_segments} max_chars={self.max_batch_chars}",
            flush=True,
        )
        for index, batch in enumerate(batches, 1):
            print(f"[gigachat-batch] start={index}/{len(batches)}", flush=True)
            batch_rows, batch_errors = self._translate_resilient(batch, memory, source_segments=source_segments)
            out.update(batch_rows)
            errors.update(batch_errors)
            print(
                f"[gigachat-batch] done={index}/{len(batches)} translated={len(batch_rows)} errors={len(batch_errors)}",
                flush=True,
            )
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
    print(f"[gigachat-probe] start segments={len(segments)} chars={sum(len(s.text) for s in segments)}", flush=True)
    translated, errors = client.translate_many(segments, memory, source_segments=source_segments)
    elapsed = max(0.001, time.perf_counter() - started)
    successful_chars = sum(len(segment.text) for segment in segments if segment.id in translated)
    cps = successful_chars / elapsed
    estimate = None if cps <= 0 else total_source_chars / cps * 1.15
    report = {
        "probe_segments": len(segments),
        "probe_success": len(translated),
        "probe_errors": errors,
        "source_chars": successful_chars,
        "elapsed_seconds": round(elapsed, 3),
        "chars_per_second": round(cps, 3),
        "estimated_full_seconds": None if estimate is None else round(estimate, 2),
        "backend": client.backend_name,
        "api_calls": client.usage.api_calls - before_calls,
        "token_usage": client.usage.as_dict(),
    }
    print("[gigachat-probe] done " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report
