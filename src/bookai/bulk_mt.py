from __future__ import annotations

import html
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .models import BookMemory, Segment


def _source_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z'-])", re.I)


def _glossary_terms(text: str, glossary: dict[str, str]) -> list[tuple[str, str]]:
    """Return high-confidence proper-name terms safe for MT terminology locking."""
    rows: list[tuple[str, str]] = []
    for source, target in glossary.items():
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target or not source[0].isupper():
            continue
        if _source_pattern(source).search(text):
            rows.append((source, target))
    rows.sort(key=lambda pair: len(pair[0]), reverse=True)
    return rows[:24]


def azure_dictionary_markup(text: str, glossary: dict[str, str]) -> str:
    """Escape prose and add Azure dynamic-dictionary markup around safe locked names."""
    terms = _glossary_terms(text, glossary)
    if not terms:
        return html.escape(text, quote=False)

    combined = re.compile(
        "|".join(
            r"(?<![A-Za-z])(" + re.escape(source) + r")(?![A-Za-z'-])"
            for source, _ in terms
        ),
        re.I,
    )
    targets = {source.casefold(): target for source, target in terms}
    out: list[str] = []
    cursor = 0
    for match in combined.finditer(text):
        out.append(html.escape(text[cursor:match.start()], quote=False))
        source_text = match.group(0)
        target = targets.get(source_text.casefold())
        if not target:
            out.append(html.escape(source_text, quote=False))
        else:
            out.append(
                '<mstrans:dictionary translation="'
                + html.escape(target, quote=True)
                + '">'
                + html.escape(source_text, quote=False)
                + '</mstrans:dictionary>'
            )
        cursor = match.end()
    out.append(html.escape(text[cursor:], quote=False))
    return "".join(out)


@dataclass
class HybridStats:
    bulk_accepted: int = 0
    bulk_qa_escalated: int = 0
    bulk_errors: int = 0
    deep_direct: int = 0
    deep_escalated: int = 0
    source_chars_bulk: int = 0
    source_chars_deep: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, int(getattr(self, key)) + int(value))

    def as_dict(self) -> dict[str, int]:
        return {
            "bulk_accepted": self.bulk_accepted,
            "bulk_qa_escalated": self.bulk_qa_escalated,
            "bulk_errors": self.bulk_errors,
            "deep_direct": self.deep_direct,
            "deep_escalated": self.deep_escalated,
            "source_chars_bulk": self.source_chars_bulk,
            "source_chars_deep": self.source_chars_deep,
        }


class AzureTranslatorBackend:
    """Azure Translator v3 bulk backend. It never falls back to a paid LLM."""

    name = "azure-translator-v3"

    def __init__(self) -> None:
        self.key = (os.getenv("AZURE_TRANSLATOR_KEY") or "").strip()
        self.region = (os.getenv("AZURE_TRANSLATOR_REGION") or "").strip()
        self.endpoint = (os.getenv("AZURE_TRANSLATOR_ENDPOINT") or "https://api.cognitive.microsofttranslator.com").rstrip("/")
        self.timeout = float(os.getenv("BOOKAI_AZURE_TIMEOUT") or "30")
        self.max_chars = max(1000, min(49000, int(os.getenv("BOOKAI_AZURE_REQUEST_CHARS") or "45000")))
        self.max_items = max(1, min(1000, int(os.getenv("BOOKAI_AZURE_REQUEST_ITEMS") or "500")))

    def available(self) -> bool:
        return bool(self.key)

    def _batches(self, segments: list[Segment]):
        batch: list[Segment] = []
        chars = 0
        for segment in segments:
            n = len(segment.text)
            if batch and (len(batch) >= self.max_items or chars + n > self.max_chars):
                yield batch
                batch = []
                chars = 0
            batch.append(segment)
            chars += n
        if batch:
            yield batch

    def _translate_batch(self, batch: list[Segment], memory: BookMemory) -> dict[str, str]:
        url = f"{self.endpoint}/translate"
        params = {"api-version": "3.0", "from": "en", "to": "ru", "textType": "html"}
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        payload = [
            {"Text": azure_dictionary_markup(segment.text, memory.glossary)}
            for segment in batch
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = httpx.post(url, params=params, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429 and attempt == 0:
                    try:
                        delay = min(3.0, max(0.25, float(response.headers.get("Retry-After") or 1)))
                    except ValueError:
                        delay = 1.0
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list) or len(data) != len(batch):
                    raise ValueError("Azure Translator returned an unexpected result count")
                out: dict[str, str] = {}
                for segment, row in zip(batch, data):
                    translations = row.get("translations") if isinstance(row, dict) else None
                    text = str((translations or [{}])[0].get("text") or "").strip()
                    if not text:
                        raise ValueError(f"Azure Translator returned empty translation for {segment.id}")
                    out[segment.id] = text
                return out
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

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
            return {}, {segment.id: "Azure Translator is not configured" for segment in segments}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        batches = list(self._batches(segments))
        workers = max(1, min(4, int(os.getenv("BOOKAI_AZURE_WORKERS") or "2")))
        out: dict[str, str] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="azure-mt") as pool:
            future_to_batch = {pool.submit(self._translate_batch, batch, memory): batch for batch in batches}
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    out.update(future.result())
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    for segment in batch:
                        errors[segment.id] = message
        return out, errors


class OPUSMTBackend:
    """Tiny EN→RU Marian/OPUS-MT backend through CTranslate2 INT8 on CPU."""

    name = "opus-mt-en-ru-ctranslate2-int8"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir or os.getenv("BOOKAI_OPUS_MODEL_DIR") or ".bookai-opus/opus-mt-en-ru-ctranslate2-int8")
        self._translator = None
        self._source_sp = None
        self._target_sp = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return all((self.model_dir / name).exists() for name in ("model.bin", "source.spm", "target.spm"))

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:
                return
            if not self.available():
                raise FileNotFoundError(f"OPUS-MT model is incomplete: {self.model_dir}")
            import ctranslate2
            import sentencepiece as spm

            threads = max(1, int(os.getenv("BOOKAI_OPUS_THREADS") or str(os.cpu_count() or 2)))
            self._translator = ctranslate2.Translator(
                str(self.model_dir),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=threads,
            )
            self._source_sp = spm.SentencePieceProcessor(model_file=str(self.model_dir / "source.spm"))
            self._target_sp = spm.SentencePieceProcessor(model_file=str(self.model_dir / "target.spm"))

    def translate_many(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        del memory, source_segments
        if not segments:
            return {}, {}
        try:
            self._ensure_loaded()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return {}, {segment.id: message for segment in segments}

        assert self._translator is not None and self._source_sp is not None and self._target_sp is not None
        try:
            source_tokens = [self._source_sp.encode(segment.text, out_type=str) for segment in segments]
            results = self._translator.translate_batch(
                source_tokens,
                beam_size=max(1, int(os.getenv("BOOKAI_OPUS_BEAM_SIZE") or "1")),
                max_batch_size=max(1, int(os.getenv("BOOKAI_OPUS_MAX_BATCH") or "256")),
                batch_type="examples",
            )
            out: dict[str, str] = {}
            errors: dict[str, str] = {}
            for segment, result in zip(segments, results):
                hypotheses = getattr(result, "hypotheses", None) or []
                if not hypotheses:
                    errors[segment.id] = "OPUS-MT returned no hypothesis"
                    continue
                text = self._target_sp.decode(hypotheses[0]).strip()
                if text:
                    out[segment.id] = text
                else:
                    errors[segment.id] = "OPUS-MT returned empty translation"
            return out, errors
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return {}, {segment.id: message for segment in segments}


class BulkMTClient:
    """Azure-first bulk MT with local OPUS fallback for only failed/unavailable calls."""

    def __init__(self, azure: AzureTranslatorBackend | None = None, opus: OPUSMTBackend | None = None) -> None:
        self.azure = azure or AzureTranslatorBackend()
        self.opus = opus or OPUSMTBackend()

    @property
    def backend_name(self) -> str:
        if self.azure.available():
            return "azure-translator-v3+opus-fallback"
        return self.opus.name

    def available(self) -> bool:
        return self.azure.available() or self.opus.available()

    def translate_many(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if not segments:
            return {}, {}
        if self.azure.available():
            out, errors = self.azure.translate_many(segments, memory, source_segments=source_segments)
            if errors and self.opus.available():
                failed = [segment for segment in segments if segment.id in errors]
                fallback, fallback_errors = self.opus.translate_many(failed, memory, source_segments=source_segments)
                out.update(fallback)
                errors = {sid: msg for sid, msg in errors.items() if sid not in fallback}
                errors.update(fallback_errors)
            return out, errors
        return self.opus.translate_many(segments, memory, source_segments=source_segments)


def benchmark_bulk_mt(
    client: BulkMTClient,
    segments: list[Segment],
    memory: BookMemory,
    *,
    source_segments: list[Segment] | None = None,
    total_source_chars: int,
) -> dict:
    started = time.perf_counter()
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
        "azure_configured": client.azure.available(),
        "opus_available": client.opus.available(),
    }
