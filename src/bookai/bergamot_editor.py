from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .gigachat_v3 import GigaChatLightningV3Backend
from .models import BookMemory, Segment
from .quality_v3 import speaker_metadata


class BergamotGigaLiteraryEditorBackend(GigaChatLightningV3Backend):
    """Experimental two-stage primary translator.

    Stage 1 is a local Firefox/Bergamot EN->RU model. It produces a cheap literal
    Russian draft for every segment. Stage 2 is GigaChat-3-Lightning, but instead
    of translating from a blank page it edits the Russian draft while treating the
    English source as authoritative. All downstream v9ah routing/DeepSeek/sanitizer
    stages remain unchanged.

    The implementation deliberately caches local drafts by segment id so recursive
    GigaChat recovery does not re-run Bergamot for the same source segment.
    """

    name = "bergamot-base-enru+gigachat-literary-editor"

    def __init__(self) -> None:
        super().__init__()
        self.bergamot_config = Path(
            os.getenv("BOOKAI_BERGAMOT_CONFIG")
            or "/tmp/bookai-bergamot-enru/config.bergamot.yml"
        )
        self.bergamot_workers = max(1, min(8, int(os.getenv("BOOKAI_BERGAMOT_WORKERS") or "4")))
        self._bergamot_service = None
        self._bergamot_model = None
        self._bergamot_options = None
        self._bergamot_vector = None
        self._draft_cache: dict[str, str] = {}
        self._draft_source: dict[str, str] = {}
        self._editor_changed_ids: set[str] = set()
        self._editor_seen_ids: set[str] = set()
        self.local_usage: dict[str, Any] = {
            "segments": 0,
            "source_chars": 0,
            "elapsed_seconds": 0.0,
            "batches": 0,
        }

    @property
    def backend_name(self) -> str:
        return self.name

    def available(self) -> bool:
        return super().available() and self.bergamot_config.exists()

    def _ensure_bergamot(self):
        if self._bergamot_service is not None:
            return self._bergamot_service, self._bergamot_model, self._bergamot_options, self._bergamot_vector
        if not self.bergamot_config.exists():
            raise RuntimeError(f"Bergamot config is missing: {self.bergamot_config}")
        try:
            from bergamot import ResponseOptions, Service, ServiceConfig, VectorString
        except Exception as exc:  # pragma: no cover - only exercised in experiment CI
            raise RuntimeError("bergamot==0.4.5 is required for the local draft experiment") from exc

        started = time.perf_counter()
        service = Service(ServiceConfig(numWorkers=self.bergamot_workers, logLevel="off"))
        model = service.modelFromConfigPath(str(self.bergamot_config))
        options = ResponseOptions(
            qualityScores=False,
            alignment=False,
            HTML=False,
            sentenceMappings=False,
        )
        self._bergamot_service = service
        self._bergamot_model = model
        self._bergamot_options = options
        self._bergamot_vector = VectorString
        print(
            f"[bergamot-client] ready config={self.bergamot_config} workers={self.bergamot_workers} "
            f"load_seconds={time.perf_counter()-started:.3f}",
            flush=True,
        )
        return service, model, options, VectorString

    def _local_drafts(self, batch: list[Segment]) -> dict[str, str]:
        missing = [
            segment
            for segment in batch
            if segment.id not in self._draft_cache or self._draft_source.get(segment.id) != segment.text
        ]
        if missing:
            service, model, options, vector_cls = self._ensure_bergamot()
            inputs = vector_cls([segment.text for segment in missing])
            started = time.perf_counter()
            responses = service.translate(model, inputs, options)
            elapsed = time.perf_counter() - started
            if len(responses) != len(missing):
                raise RuntimeError(
                    f"Bergamot returned {len(responses)} responses for {len(missing)} segments"
                )
            for segment, response in zip(missing, responses):
                value = str(response.target.text or "").strip()
                if not value:
                    raise RuntimeError(f"Bergamot returned an empty draft for {segment.id}")
                self._draft_cache[segment.id] = value
                self._draft_source[segment.id] = segment.text
            chars = sum(len(segment.text) for segment in missing)
            self.local_usage["segments"] += len(missing)
            self.local_usage["source_chars"] += chars
            self.local_usage["elapsed_seconds"] = round(
                float(self.local_usage["elapsed_seconds"]) + elapsed, 3
            )
            self.local_usage["batches"] += 1
            cps = chars / max(0.001, elapsed)
            print(
                f"[bergamot-draft] segments={len(missing)} chars={chars} elapsed={elapsed:.3f}s cps={cps:.1f}",
                flush=True,
            )
        return {segment.id: self._draft_cache[segment.id] for segment in batch}

    def _editor_prompt(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        minimal: bool = False,
    ) -> str:
        drafts = self._local_drafts(batch)
        style = memory.style
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        nearest = self._context_for_batch(batch, source_segments)
        summary = "" if minimal else self._clip(memory.rolling_summary, 1000)
        style_rows = "" if minimal else (
            f"VOICE: {self._clip(style.narrative_voice, 1000)}\n"
            f"RHYTHM: {self._clip(style.rhythm, 500)}\n"
            f"DIALOGUE: {self._clip(style.dialogue, 400)}\n"
            f"HUMOR: {self._clip(style.humor, 400)}"
        )
        targets = {
            segment.id: {
                "source_en": segment.text,
                "draft_ru": drafts[segment.id],
            }
            for segment in batch
        }
        metadata = speaker_metadata(batch, source_segments, memory)
        metadata_block = json.dumps(metadata, ensure_ascii=False) if metadata else "нет"
        return f"""Задача: литературная РЕДАКТУРА русского машинного черновика по английскому оригиналу.

SOURCE_EN — единственный источник истины по смыслу. DRAFT_RU создан маленькой локальной MT-моделью и может быть
буквальным, неестественным или просто ошибочным. Не доверяй ему в споре с SOURCE_EN.

КРИТИЧЕСКИЙ КОНТРАКТ:
1. Для каждого ID верни ПОЛНЫЙ финальный русский текст ровно соответствующего SOURCE_EN.
2. Это редактура, а не слепое сохранение черновика: активно исправляй машинные кальки, синтаксис, лексику,
   ритм, реплики и литературную естественность. Если черновик уже публикационного уровня, его можно сохранить.
3. Не сокращай и не расширяй смысл. Сохраняй субъект/объект, числа, отрицания, время, причинность, родство,
   юридические функции, сравнения, местоименные связи и физический смысл специальных терминов.
4. Стиль — естественная опубликованная русская проза с тем же голосом автора, ритмом, сухой иронией и
   характером диалогов. Никакого канцелярита и буквальной англоязычной конструкции фраз.
5. CONTEXT_ONLY, BOOK_MEMORY и служебные блоки нужны только для понимания и не должны попадать в результат.
6. Не оставляй обычные английские слова в русском тексте; имена передавай согласованно с каноном.

{style_rows}

BOOK_MEMORY (НЕ КОПИРОВАТЬ):
{summary or 'нет'}

CHARACTERS:
{characters}

GLOSSARY:
{glossary}

SPEAKER_METADATA (только для рода/голоса):
{metadata_block}

CONTEXT_ONLY:
{nearest or 'нет'}

TARGETS:
{json.dumps(targets, ensure_ascii=False)}

Верни только JSON по переданной schema: ID -> полный отредактированный русский текст."""

    def _translate_batch(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        minimal: bool = False,
    ) -> tuple[dict[str, str], dict[str, int]]:
        # Ensure local drafts are generated before constructing the editor prompt.
        drafts = self._local_drafts(batch)
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты литературный редактор русского перевода. Английский оригинал — источник истины; "
                    "машинный русский черновик — только заготовка. Верни публикационный русский текст."
                ),
            },
            {
                "role": "user",
                "content": self._editor_prompt(
                    batch,
                    memory,
                    source_segments=source_segments,
                    minimal=minimal,
                ),
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
            fallback = dict(payload)
            fallback.pop("response_format", None)
            fallback["messages"] = [dict(messages[0]), dict(messages[1])]
            fallback["messages"][0]["content"] += " Верни только JSON-объект с ровно требуемыми ID."
            response = self._chat(fallback, batch, strict=False)

        parsed = self._parse_json_object(str(response.choices[0].message.content or ""))
        expected = {segment.id for segment in batch}
        translated = {sid: text for sid, text in parsed.items() if sid in expected and text}
        for segment in batch:
            if segment.id not in translated:
                continue
            self._editor_seen_ids.add(segment.id)
            if " ".join(translated[segment.id].split()) != " ".join(drafts[segment.id].split()):
                self._editor_changed_ids.add(segment.id)
        if self._editor_seen_ids:
            print(
                "[bergamot-editor] "
                + json.dumps(
                    {
                        "seen_unique": len(self._editor_seen_ids),
                        "changed_unique": len(self._editor_changed_ids),
                        "changed_percent": round(
                            len(self._editor_changed_ids) / max(1, len(self._editor_seen_ids)) * 100, 1
                        ),
                        "local_usage": self.local_usage,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return translated, self._usage(response)

    def experiment_stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "bergamot": dict(self.local_usage),
            "giga_editor_seen_unique": len(self._editor_seen_ids),
            "giga_editor_changed_unique": len(self._editor_changed_ids),
            "giga_editor_changed_percent": round(
                len(self._editor_changed_ids) / max(1, len(self._editor_seen_ids)) * 100, 2
            ),
            "gigachat_usage": self.usage.as_dict(),
        }
