from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

import httpx

from .models import BookMemory, LLMProvider, Segment, StyleGuide


class OpenAICompatibleProvider:
    """Small OpenAI-compatible client with DeepSeek V4 Flash defaults and usage tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
    ):
        self.api_key = api_key or os.getenv("BOOKAI_API_KEY")
        self.base_url = (base_url or os.getenv("BOOKAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("BOOKAI_MODEL") or "deepseek-v4-flash"
        self.thinking = (thinking or os.getenv("BOOKAI_THINKING") or "disabled").lower()
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("BOOKAI_THINKING must be enabled or disabled")
        if not self.api_key:
            raise ValueError("BOOKAI_API_KEY is required")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "thinking": {"type": self.thinking},
            "response_format": {"type": "json_object"},
        }
        if self.thinking == "disabled":
            payload["temperature"] = temperature

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240.0,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.usage["total_tokens"] += int(usage.get("total_tokens") or 0)
        self.usage["requests"] += 1
        return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        end = text.rfind(closer)
        if end < start:
            raise
        return json.loads(text[start : end + 1])


def analyze_memory(provider: LLMProvider, sample: str, title: str = "", author: str = "") -> BookMemory:
    system = """You are a senior literary editor preparing a translation bible for a novel.
Return ONLY valid JSON. Never invent plot details. Analyze only evidence in the supplied sample."""
    user = f"""Analyze this English book sample before translation into Russian.
Book title: {title!r}
Author: {author!r}
Return an object with keys:
style: {{narrative_voice, rhythm, dialogue, humor, taboos:[...]}},
glossary: {{original_term: preferred_russian}},
characters: {{name: speech/personality note}},
rolling_summary: a short factual summary in Russian.

SAMPLE:
{sample[:50000]}"""
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.1))
        assert isinstance(obj, dict)
        style_obj = obj.get("style") or {}
        defaults = StyleGuide()
        style = StyleGuide(
            narrative_voice=str(style_obj.get("narrative_voice") or defaults.narrative_voice),
            rhythm=str(style_obj.get("rhythm") or defaults.rhythm),
            dialogue=str(style_obj.get("dialogue") or defaults.dialogue),
            humor=str(style_obj.get("humor") or defaults.humor),
            taboos=[str(x) for x in (style_obj.get("taboos") or defaults.taboos)],
        )
        return BookMemory(
            title=title,
            author=author,
            style=style,
            glossary={str(k): str(v) for k, v in (obj.get("glossary") or {}).items()},
            characters={str(k): str(v) for k, v in (obj.get("characters") or {}).items()},
            rolling_summary=str(obj.get("rolling_summary") or ""),
        )
    except Exception:
        return BookMemory(title=title, author=author)


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, indent=2)


def translate_batch(provider: LLMProvider, segments: list[Segment], memory: BookMemory) -> dict[str, str]:
    system = """You are a professional English-to-Russian literary translator.
Faithfulness outranks prettiness. Preserve meaning, ambiguity, tone, characterization, paragraph function, profanity and deliberate stylistic roughness.
Adapt idioms, puns and wordplay to reproduce their function in Russian; never silently invent facts.
Return ONLY a JSON object mapping every supplied segment id to its Russian translation. Do not omit ids."""
    payload = {s.id: s.text for s in segments}
    user = f"""TRANSLATION BIBLE:
{_memory_prompt(memory)}

TEXT SEGMENTS:
{json.dumps(payload, ensure_ascii=False)}"""
    obj = extract_json(provider.complete(system, user, temperature=0.25))
    if not isinstance(obj, dict):
        raise ValueError("Translator returned non-object JSON")
    return {s.id: str(obj.get(s.id, s.text)) for s in segments}


def edit_batch(provider: LLMProvider, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> dict[str, str]:
    system = """You are a Russian literary editor working from an English original and a draft translation.
Fix calques, unnatural syntax, broken idioms, puns and rhythm while preserving the author's exact intent and idiosyncrasies.
Do not beautify plain prose, flatten strange prose, censor, summarize, explain, or change facts.
Return ONLY a JSON object mapping every segment id to edited Russian text."""
    pairs = {s.id: {"original": s.text, "draft": draft[s.id]} for s in originals}
    user = f"""TRANSLATION BIBLE:
{_memory_prompt(memory)}

PAIRS:
{json.dumps(pairs, ensure_ascii=False)}"""
    obj = extract_json(provider.complete(system, user, temperature=0.2))
    if not isinstance(obj, dict):
        raise ValueError("Editor returned non-object JSON")
    return {s.id: str(obj.get(s.id, draft[s.id])) for s in originals}


def qa_batch(provider: LLMProvider, originals: list[Segment], edited: dict[str, str], memory: BookMemory) -> dict[str, str]:
    system = """You are a bilingual translation QA editor. Compare English source to Russian candidate.
Correct only real issues: mistranslation, omitted meaning, added facts, inconsistent names/terms, broken references, lost jokes, or wrong register.
If a candidate is already faithful and literary, keep it unchanged. Return ONLY a JSON object mapping every id to the final Russian text."""
    pairs = {s.id: {"original": s.text, "candidate": edited[s.id]} for s in originals}
    user = f"""TRANSLATION BIBLE:
{_memory_prompt(memory)}

PAIRS:
{json.dumps(pairs, ensure_ascii=False)}"""
    obj = extract_json(provider.complete(system, user, temperature=0.05))
    if not isinstance(obj, dict):
        raise ValueError("QA returned non-object JSON")
    return {s.id: str(obj.get(s.id, edited[s.id])) for s in originals}


def update_memory(provider: LLMProvider, originals: list[Segment], translated: dict[str, str], memory: BookMemory) -> BookMemory:
    """Update long-term continuity without rewriting established style rules."""
    system = """You maintain continuity notes for a literary translation.
Update only facts supported by the new passage. Keep established translations unless the new passage proves them wrong.
Do not invent backstory. Return ONLY valid JSON."""
    pairs = [
        {"id": s.id, "chapter": s.chapter, "original": s.text, "russian": translated.get(s.id, "")}
        for s in originals
    ]
    user = f"""CURRENT MEMORY:
{_memory_prompt(memory)}

NEW PASSAGE:
{json.dumps(pairs, ensure_ascii=False)}

Return an object with:
glossary: complete merged glossary,
characters: complete merged character/speech notes,
rolling_summary: updated concise factual story summary (max ~1200 Russian words),
last_chapter: the latest chapter label seen.
Do not return style; style is intentionally stable."""
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.05))
        if not isinstance(obj, dict):
            return memory
        return BookMemory(
            title=memory.title,
            author=memory.author,
            style=memory.style,
            glossary={str(k): str(v) for k, v in (obj.get("glossary") or memory.glossary).items()},
            characters={str(k): str(v) for k, v in (obj.get("characters") or memory.characters).items()},
            rolling_summary=str(obj.get("rolling_summary") or memory.rolling_summary),
            last_chapter=str(obj.get("last_chapter") or (originals[-1].chapter if originals else memory.last_chapter)),
        )
    except Exception:
        return memory
