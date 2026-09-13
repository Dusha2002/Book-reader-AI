from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .gigachat_v3 import GigaChatLightningV3Backend
from .models import BookMemory, Segment
from .quality_v3 import speaker_metadata


class BergamotGigaLiteraryEditorBackend(GigaChatLightningV3Backend):
    """Experimental local MT draft -> source-grounded GigaChat literary editor.

    Bergamot 0.4.5 only ships a CPython 3.10 Linux wheel while Book Reader requires
    Python >=3.11. To keep both fast and isolated, Bergamot runs as one persistent
    Python 3.10 worker process per chapter. The main pipeline stays on Python 3.12.
    """

    name = "bergamot-base-enru+gigachat-literary-editor"

    def __init__(self) -> None:
        super().__init__()
        self.bergamot_config = Path(
            os.getenv("BOOKAI_BERGAMOT_CONFIG")
            or "/tmp/bookai-bergamot-enru/config.bergamot.yml"
        )
        self.bergamot_python = (
            os.getenv("BOOKAI_BERGAMOT_PYTHON") or sys.executable
        ).strip()
        self.bergamot_worker_script = Path(
            os.getenv("BOOKAI_BERGAMOT_WORKER") or "scripts/bergamot_worker.py"
        )
        self.bergamot_workers = max(1, min(8, int(os.getenv("BOOKAI_BERGAMOT_WORKERS") or "4")))
        self._bergamot_proc: subprocess.Popen[str] | None = None
        self._draft_cache: dict[str, str] = {}
        self._draft_source: dict[str, str] = {}
        self._editor_changed_ids: set[str] = set()
        self._editor_seen_ids: set[str] = set()
        self.local_usage: dict[str, Any] = {
            "segments": 0,
            "source_chars": 0,
            "elapsed_seconds": 0.0,
            "worker_inference_seconds": 0.0,
            "batches": 0,
        }
        atexit.register(self._close_bergamot)

    @property
    def backend_name(self) -> str:
        return self.name

    def available(self) -> bool:
        return (
            super().available()
            and self.bergamot_config.exists()
            and Path(self.bergamot_python).exists()
            and self.bergamot_worker_script.exists()
        )

    def _close_bergamot(self) -> None:
        proc = self._bergamot_proc
        self._bergamot_proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_bergamot(self) -> subprocess.Popen[str]:
        proc = self._bergamot_proc
        if proc is not None and proc.poll() is None:
            return proc
        if not self.bergamot_config.exists():
            raise RuntimeError(f"Bergamot config is missing: {self.bergamot_config}")
        if not Path(self.bergamot_python).exists():
            raise RuntimeError(f"Bergamot Python is missing: {self.bergamot_python}")
        if not self.bergamot_worker_script.exists():
            raise RuntimeError(f"Bergamot worker script is missing: {self.bergamot_worker_script}")
        cmd = [
            self.bergamot_python,
            str(self.bergamot_worker_script),
            "--config",
            str(self.bergamot_config),
            "--workers",
            str(self.bergamot_workers),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._bergamot_proc = proc
        print(
            f"[bergamot-client] worker_started python={self.bergamot_python} config={self.bergamot_config}",
            flush=True,
        )
        return proc

    def _worker_translate(self, missing: list[Segment]) -> tuple[dict[str, str], float]:
        proc = self._ensure_bergamot()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Bergamot worker pipes are unavailable")
        request = {
            "items": [{"id": segment.id, "text": segment.text} for segment in missing]
        }
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        raw = proc.stdout.readline()
        if not raw:
            code = proc.poll()
            raise RuntimeError(f"Bergamot worker exited without response code={code}")
        obj = json.loads(raw)
        if obj.get("error"):
            raise RuntimeError(f"Bergamot worker error: {obj['error']}")
        rows = obj.get("items") or []
        parsed = {
            str(row.get("id") or ""): str(row.get("text") or "").strip()
            for row in rows
            if isinstance(row, dict)
        }
        return parsed, float(obj.get("elapsed_seconds") or 0.0)

    def _local_drafts(self, batch: list[Segment]) -> dict[str, str]:
        missing = [
            segment
            for segment in batch
            if segment.id not in self._draft_cache or self._draft_source.get(segment.id) != segment.text
        ]
        if missing:
            started = time.perf_counter()
            parsed, worker_elapsed = self._worker_translate(missing)
            elapsed = time.perf_counter() - started
            for segment in missing:
                value = parsed.get(segment.id, "")
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
            self.local_usage["worker_inference_seconds"] = round(
                float(self.local_usage["worker_inference_seconds"]) + worker_elapsed, 3
            )
            self.local_usage["batches"] += 1
            cps = chars / max(0.001, worker_elapsed or elapsed)
            print(
                f"[bergamot-draft] segments={len(missing)} chars={chars} elapsed={elapsed:.3f}s "
                f"worker={worker_elapsed:.3f}s cps={cps:.1f}",
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
            segment.id: {"source_en": segment.text, "draft_ru": drafts[segment.id]}
            for segment in batch
        }
        metadata = speaker_metadata(batch, source_segments, memory)
        metadata_block = json.dumps(metadata, ensure_ascii=False) if metadata else "нет"
        return f"""Задача: литературная РЕДАКТУРА русского машинного черновика по английскому оригиналу.

SOURCE_EN — единственный источник истины. DRAFT_RU создан маленькой локальной MT-моделью и может быть буквальным,
неестественным или ошибочным. Не доверяй ему в споре с SOURCE_EN.

КРИТИЧЕСКИЙ КОНТРАКТ:
1. Для каждого ID верни ПОЛНЫЙ финальный русский текст ровно соответствующего SOURCE_EN.
2. Активно исправляй машинные кальки, синтаксис, лексику, ритм, реплики и литературную естественность. Если
   черновик уже публикационного уровня, его можно сохранить без косметической перефразировки.
3. Не сокращай и не расширяй смысл. Сохраняй субъект/объект, числа, отрицания, время, причинность, родство,
   юридические функции, сравнения, местоименные связи и точный физический смысл специальных терминов.
4. Стиль — естественная опубликованная русская проза с тем же голосом автора, ритмом, сухой иронией и характером
   диалогов. Никакого канцелярита и буквальной англоязычной конструкции фраз.
5. CONTEXT_ONLY и служебные блоки нужны только для понимания и не должны попадать в результат.
6. Не оставляй обычные английские слова; имена передавай согласованно с каноном.

{style_rows}

BOOK_MEMORY:
{summary or 'нет'}

CHARACTERS:
{characters}

GLOSSARY:
{glossary}

SPEAKER_METADATA:
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
                "content": self._editor_prompt(batch, memory, source_segments=source_segments, minimal=minimal),
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
