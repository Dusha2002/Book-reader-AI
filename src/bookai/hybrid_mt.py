from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import httpx

from .models import BookMemory, Segment


@dataclass(slots=True)
class RouteDecision:
    route: str  # hymt | deepseek
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class HybridStats:
    hymt_accepted: int = 0
    hymt_qa_escalated: int = 0
    hymt_errors: int = 0
    deep_direct: int = 0
    deep_escalated: int = 0
    source_chars_hymt: int = 0
    source_chars_deep: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, int(getattr(self, key)) + int(value))

    def as_dict(self) -> dict[str, int]:
        return {
            "hymt_accepted": self.hymt_accepted,
            "hymt_qa_escalated": self.hymt_qa_escalated,
            "hymt_errors": self.hymt_errors,
            "deep_direct": self.deep_direct,
            "deep_escalated": self.deep_escalated,
            "source_chars_hymt": self.source_chars_hymt,
            "source_chars_deep": self.source_chars_deep,
        }


def complexity_score(segment: Segment) -> tuple[float, list[str]]:
    """Cheap routing proxy: easy prose to MT, structurally risky prose to DeepSeek."""
    text = segment.text.strip()
    if not text:
        return 0.0, ["empty"]
    if len(text) <= 90 and "\n" not in text:
        return 0.25, ["short"]

    reasons: list[str] = []
    score = 0.0
    n = len(text)
    if n >= 700:
        score += min(3.0, (n - 500) / 550.0)
        reasons.append("long")

    semicolons = text.count(";")
    colons = text.count(":")
    dashes = text.count("—") + text.count(" - ")
    clause_pressure = text.count(",") + 2 * semicolons + colons + dashes
    if clause_pressure >= 8:
        score += min(2.4, (clause_pressure - 6) * 0.24)
        reasons.append("clause_pressure")

    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if len(sentences) >= 6:
        score += min(1.5, (len(sentences) - 4) * 0.25)
        reasons.append("many_sentences")
    if len(sentences) == 1 and n >= 900:
        score += 2.0
        reasons.append("very_long_single_sentence")

    subordinate = len(re.findall(r"\b(?:which|whose|whom|whereby|although|whereas|unless|whether|while|because|since)\b", text, re.I))
    if subordinate >= 3:
        score += min(1.5, subordinate * 0.25)
        reasons.append("subordination")

    if re.search(r"\b(?:gear|wheel|shaft|spring|lever|blade|axle|bearing|pulley|ratchet|mechanism|diameter|angle|pressure)\b", text, re.I):
        score += 0.65
        reasons.append("technical")
    if re.search(r"[!?].*[!?]|\b(?:joke|funny|laugh|irony|ironic)\b", text, re.I):
        score += 0.35
        reasons.append("tone_risk")
    if text.count('"') + text.count("“") + text.count("”") >= 4:
        score += 0.35
        reasons.append("dialogue_switches")

    return round(score, 3), reasons


def decide_route(segment: Segment, *, hard_threshold: float | None = None) -> RouteDecision:
    threshold = float(hard_threshold if hard_threshold is not None else os.getenv("BOOKAI_HYMT_HARD_THRESHOLD") or "4.8")
    score, reasons = complexity_score(segment)
    return RouteDecision("deepseek" if score >= threshold else "hymt", score, reasons)


def routing_summary(segments: Iterable[Segment], *, hard_threshold: float | None = None) -> dict:
    rows = [decide_route(segment, hard_threshold=hard_threshold) for segment in segments]
    return {
        "total": len(rows),
        "hymt": sum(row.route == "hymt" for row in rows),
        "deepseek": sum(row.route == "deepseek" for row in rows),
        "mean_score": round(sum(row.score for row in rows) / max(1, len(rows)), 3),
        "hard_threshold": float(hard_threshold if hard_threshold is not None else os.getenv("BOOKAI_HYMT_HARD_THRESHOLD") or "4.8"),
    }


def _relevant_glossary(text: str, glossary: dict[str, str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for source, target in glossary.items():
        if re.search(r"(?<![A-Za-z])" + re.escape(source) + r"(?![A-Za-z])", text, re.I):
            out.append((source, target))
    return out[:20]


def build_hymt_prompt(
    segment: Segment,
    memory: BookMemory,
    *,
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> str:
    """Tencent HY-MT contextual prompt, adapted to EN→RU with compact literary hints."""
    context_parts: list[str] = []
    rolling = " ".join(str(memory.rolling_summary or "").split())
    if rolling:
        context_parts.append("Book/chapter context: " + rolling[-2200:])

    style_parts = [
        str(memory.style.narrative_voice or ""),
        str(memory.style.rhythm or ""),
        str(memory.style.dialogue or ""),
        str(memory.style.humor or ""),
    ]
    style = " ".join(" ".join(style_parts).split())
    if style:
        context_parts.append("Literary style tendencies: " + style[-1400:])

    relevant = _relevant_glossary(segment.text, memory.glossary)
    if relevant:
        context_parts.append(
            "Locked terminology: " + "; ".join(f"{source} -> {target}" for source, target in relevant)
        )

    before = [s.text for s in (context_before or [])[-2:] if s.text.strip()]
    after = [s.text for s in (context_after or [])[:2] if s.text.strip()]
    if before:
        context_parts.append("Immediate source context before: " + " ".join(before)[-1200:])
    if after:
        context_parts.append("Immediate source context after: " + " ".join(after)[:1200])

    if not context_parts:
        return (
            "Translate the following segment into Russian, without additional explanation.\n\n"
            + segment.text
        )

    context = "\n".join(context_parts)
    return (
        f"{context}\n\n"
        "Using the information above, translate the text below into Russian. "
        "Do not translate the context and do not add explanations. Preserve every fact, relation, number, negation, "
        "deliberate repetition and the source's tone. Use natural literary Russian.\n\n"
        f"{segment.text}"
    )


class HYMTClient:
    """OpenAI-compatible client for a persistent local llama.cpp HY-MT server."""

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.getenv("BOOKAI_HYMT_BASE_URL") or "http://127.0.0.1:8080/v1").rstrip("/")
        self.timeout = float(timeout if timeout is not None else os.getenv("BOOKAI_HYMT_TIMEOUT") or "120")
        self.model = model or os.getenv("BOOKAI_HYMT_MODEL") or self._discover_model()

    def _discover_model(self) -> str:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=min(self.timeout, 15.0))
            response.raise_for_status()
            rows = response.json().get("data") or []
            if rows and rows[0].get("id"):
                return str(rows[0]["id"])
        except Exception:
            pass
        return "hy-mt-local"

    def healthy(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=min(self.timeout, 10.0))
            return response.status_code < 400
        except Exception:
            return False

    def translate_one(
        self,
        segment: Segment,
        memory: BookMemory,
        *,
        context_before: list[Segment] | None = None,
        context_after: list[Segment] | None = None,
    ) -> str:
        prompt = build_hymt_prompt(
            segment,
            memory,
            context_before=context_before,
            context_after=context_after,
        )
        max_tokens = max(128, min(4096, int(len(segment.text) * 1.5) + 256))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("BOOKAI_HYMT_TEMPERATURE") or "0.7"),
            "top_p": float(os.getenv("BOOKAI_HYMT_TOP_P") or "0.6"),
            "max_tokens": max_tokens,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        obj = response.json()
        text = str(obj["choices"][0]["message"]["content"] or "").strip()
        text = re.sub(r"^<target>|</target>$", "", text, flags=re.I).strip()
        if not text:
            raise ValueError(f"HY-MT returned empty translation for {segment.id}")
        return text

    def translate_many(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        workers: int | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if not segments:
            return {}, {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        worker_count = max(1, min(8, int(workers if workers is not None else os.getenv("BOOKAI_HYMT_WORKERS") or "2")))
        positions = {s.id: i for i, s in enumerate(source_segments or [])}

        def translate(segment: Segment) -> str:
            before: list[Segment] = []
            after: list[Segment] = []
            if source_segments and segment.id in positions:
                index = positions[segment.id]
                before = source_segments[max(0, index - 2):index]
                after = source_segments[index + 1:index + 3]
            return self.translate_one(segment, memory, context_before=before, context_after=after)

        out: dict[str, str] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hymt") as pool:
            futures = {pool.submit(translate, segment): segment for segment in segments}
            for future in as_completed(futures):
                segment = futures[future]
                try:
                    out[segment.id] = future.result()
                except Exception as exc:
                    errors[segment.id] = f"{type(exc).__name__}: {exc}"
        return out, errors


def choose_probe_segments(segments: list[Segment], count: int = 10) -> list[Segment]:
    candidates = [
        segment for segment in segments
        if decide_route(segment).route == "hymt" and 180 <= len(segment.text) <= 1000
    ]
    if not candidates:
        candidates = [segment for segment in segments if segment.text.strip()]
    if len(candidates) <= count:
        return candidates
    step = (len(candidates) - 1) / max(1, count - 1)
    return [candidates[round(i * step)] for i in range(count)]


def benchmark_hymt(
    client: HYMTClient,
    segments: list[Segment],
    memory: BookMemory,
    *,
    source_segments: list[Segment] | None = None,
    total_source_chars: int,
) -> dict:
    started = time.perf_counter()
    translated, errors = client.translate_many(
        segments,
        memory,
        source_segments=source_segments,
    )
    elapsed = max(0.001, time.perf_counter() - started)
    successful_chars = sum(len(segment.text) for segment in segments if segment.id in translated)
    chars_per_second = successful_chars / elapsed
    estimated_seconds = math.inf if chars_per_second <= 0 else total_source_chars / chars_per_second * 1.15
    return {
        "probe_segments": len(segments),
        "probe_success": len(translated),
        "probe_errors": errors,
        "source_chars": successful_chars,
        "elapsed_seconds": round(elapsed, 3),
        "source_chars_per_second": round(chars_per_second, 3),
        "estimated_full_seconds": None if not math.isfinite(estimated_seconds) else round(estimated_seconds, 1),
        "estimated_full_minutes": None if not math.isfinite(estimated_seconds) else round(estimated_seconds / 60.0, 2),
        "model": client.model,
        "base_url": client.base_url,
    }


def dumps_report(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
