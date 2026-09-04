from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

import httpx

from .models import BookMemory, GateFinding, LLMProvider, Segment, StyleGuide


class OpenAICompatibleProvider:
    """Small OpenAI-compatible JSON client.

    Defaults to OpenRouter but still works with direct DeepSeek-compatible endpoints.
    Each instance represents one role/model and tracks its own usage.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        role: str = "llm",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("BOOKAI_API_KEY")
        self.base_url = (base_url or os.getenv("BOOKAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model or os.getenv("BOOKAI_MODEL") or "deepseek/deepseek-v4-flash-0731"
        self.reasoning_effort = (reasoning_effort or os.getenv("BOOKAI_REASONING") or "none").lower()
        self.role = role
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or BOOKAI_API_KEY is required")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if "openrouter.ai" in self.base_url:
            payload["reasoning"] = {"effort": self.reasoning_effort}
            headers["X-Title"] = "Book Reader AI"
        elif "deepseek.com" in self.base_url:
            payload["thinking"] = {"type": "enabled" if self.reasoning_effort != "none" else "disabled"}

        if self.reasoning_effort == "none":
            payload["temperature"] = temperature

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
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
    system = """You are a senior literary editor preparing a compact translation bible for an English novel translated into Russian.
Return ONLY valid JSON. Never invent plot details. Analyze only evidence in the supplied sample."""
    user = f"""Book title: {title!r}\nAuthor: {author!r}
Return an object with keys:
style: {{narrative_voice, rhythm, dialogue, humor, taboos:[...]}},
glossary: {{original_term: preferred_russian}},
characters: {{name: speech/personality note}},
rolling_summary: a short factual summary in Russian.

SAMPLE:\n{sample[:50000]}"""
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
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


def translate_batch(
    provider: LLMProvider,
    segments: list[Segment],
    memory: BookMemory,
    *,
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> dict[str, str]:
    system = """You are a professional English-to-Russian literary translator.
Faithfulness outranks prettiness. Preserve meaning, ambiguity, tone, characterization, paragraph function, profanity and deliberate stylistic roughness.
Adapt idioms and wordplay by function when literal Russian would fail. Never invent facts.
Context-only passages exist only for resolving references; DO NOT translate or return them.
Return ONLY a JSON object mapping EVERY target segment id to its Russian translation."""
    payload = {s.id: s.text for s in segments}
    before = [{"id": s.id, "text": s.text} for s in (context_before or [])]
    after = [{"id": s.id, "text": s.text} for s in (context_after or [])]
    user = f"""TRANSLATION_BIBLE:{_memory_prompt(memory)}
CONTEXT_BEFORE:{json.dumps(before, ensure_ascii=False)}
TARGET_SEGMENTS:{json.dumps(payload, ensure_ascii=False)}
CONTEXT_AFTER:{json.dumps(after, ensure_ascii=False)}"""
    obj = extract_json(provider.complete(system, user, temperature=0.2))
    if not isinstance(obj, dict):
        raise ValueError("Translator returned non-object JSON")
    return {s.id: str(obj.get(s.id, s.text)) for s in segments}


def quality_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    system = """You are a fast bilingual EN-RU translation quality gate. DO NOT rewrite text.
Flag only passages that truly need human-grade literary editing. Be conservative: routine faithful translations should pass.
Use severity 'hard' only for puns, ambiguity, poetry, culturally dependent humor, voice-critical dialogue, or a likely serious semantic error.
Use 'medium' for calques, register/style mismatch, terminology inconsistency, awkward Russian, or smaller mistranslation.
Return ONLY JSON: {"issues":[{"id":"...","severity":"medium|hard","reason":"short reason"}]}.
Never flag merely because the wording differs literally from English."""
    pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.0))
        if not isinstance(obj, dict):
            return []
        findings: list[GateFinding] = []
        valid_ids = {s.id for s in originals}
        for raw in obj.get("issues") or []:
            if not isinstance(raw, dict) or str(raw.get("id")) not in valid_ids:
                continue
            severity = str(raw.get("severity") or "medium").lower()
            if severity not in {"medium", "hard"}:
                severity = "medium"
            findings.append(GateFinding(str(raw["id"]), severity, str(raw.get("reason") or "")))
        return findings
    except Exception:
        return []


def edit_batch(provider: LLMProvider, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> dict[str, str]:
    if not originals:
        return {}
    system = """You are a Russian literary editor working from an English original and a draft translation.
Fix only genuine translation/literary problems: calques, unnatural syntax, broken idioms or puns, wrong register, lost voice, terminology inconsistency and mistranslation.
Preserve the author's exact intent and idiosyncrasies. Do not beautify plain prose, flatten strange prose, censor, summarize, explain, or change facts.
Return ONLY a JSON object mapping every supplied segment id to edited Russian text."""
    pairs = {s.id: {"original": s.text, "draft": draft[s.id]} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.15))
    if not isinstance(obj, dict):
        raise ValueError("Editor returned non-object JSON")
    return {s.id: str(obj.get(s.id, draft[s.id])) for s in originals}


def qa_batch(provider: LLMProvider, originals: list[Segment], edited: dict[str, str], memory: BookMemory) -> dict[str, str]:
    if not originals:
        return {}
    system = """You are the senior bilingual editor for difficult EN-RU literary passages.
Resolve semantic ambiguity, wordplay, culturally dependent humor, poetry, voice-critical dialogue and serious mistranslations.
Prefer the existing candidate when already correct. Never invent or explain. Return ONLY a JSON object mapping every supplied id to final Russian text."""
    pairs = {s.id: {"original": s.text, "candidate": edited[s.id]} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nHARD_PAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.1))
    if not isinstance(obj, dict):
        raise ValueError("Senior QA returned non-object JSON")
    return {s.id: str(obj.get(s.id, edited[s.id])) for s in originals}


def update_memory(provider: LLMProvider, originals: list[Segment], translated: dict[str, str], memory: BookMemory) -> BookMemory:
    system = """You maintain compact continuity notes for a literary translation.
Update only facts supported by the new passage. Keep established translations unless new evidence proves them wrong.
Do not invent backstory. Return ONLY valid JSON."""
    pairs = [{"id": s.id, "chapter": s.chapter, "en": s.text, "ru": translated.get(s.id, "")} for s in originals]
    user = f"""CURRENT_MEMORY:{_memory_prompt(memory)}
NEW_PASSAGE:{json.dumps(pairs, ensure_ascii=False)}
Return {{"glossary":complete_merged_glossary,"characters":complete_merged_notes,"rolling_summary":"concise factual summary max ~1000 Russian words","last_chapter":"latest label"}}."""
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.0))
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
