from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

import httpx

from .models import BookMemory, LLMProvider, Segment, StyleGuide


class OpenAICompatibleProvider:
    """Minimal client for OpenAI-compatible /chat/completions endpoints."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("BOOKAI_API_KEY")
        self.base_url = (base_url or os.getenv("BOOKAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("BOOKAI_MODEL") or "deepseek-v4-pro"
        if not self.api_key:
            raise ValueError("BOOKAI_API_KEY is required")

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
            },
            timeout=180.0,
        )
        response.raise_for_status()
        data = response.json()
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
Return ONLY valid JSON. Never imitate prose by adding new plot details. Analyze only evidence in the sample."""
    user = f"""Analyze this English book sample before translation into Russian.
Book title: {title!r}
Author: {author!r}
Return an object with keys:
style: {{narrative_voice, rhythm, dialogue, humor, taboos:[...]}},
glossary: {{original_term: preferred_russian}},
characters: {{name: speech/personality note}},
rolling_summary: a short factual summary in Russian.

SAMPLE:
{sample[:30000]}"""
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.1))
        assert isinstance(obj, dict)
        style_obj = obj.get("style") or {}
        style = StyleGuide(
            narrative_voice=str(style_obj.get("narrative_voice") or StyleGuide().narrative_voice),
            rhythm=str(style_obj.get("rhythm") or StyleGuide().rhythm),
            dialogue=str(style_obj.get("dialogue") or StyleGuide().dialogue),
            humor=str(style_obj.get("humor") or StyleGuide().humor),
            taboos=[str(x) for x in (style_obj.get("taboos") or StyleGuide().taboos)],
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
Return ONLY a JSON object mapping each supplied segment id to its Russian translation. Do not omit ids."""
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
Return ONLY a JSON object mapping segment ids to edited Russian text."""
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
