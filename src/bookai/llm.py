from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import asdict

import httpx
from dotenv import load_dotenv

from .models import BookMemory, GateFinding, LLMProvider, Segment, StyleGuide
from .quality import assert_exact_ids, is_heading


class OpenAICompatibleProvider:
    """Small OpenAI-compatible JSON client with retries and usage accounting."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        role: str = "llm",
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("BOOKAI_API_KEY")
        self.base_url = (base_url or os.getenv("BOOKAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model or os.getenv("BOOKAI_MODEL") or "deepseek/deepseek-v4-flash-0731"
        self.reasoning_effort = (reasoning_effort or os.getenv("BOOKAI_REASONING") or "none").lower()
        self.role = role
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or BOOKAI_API_KEY is required")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "cost": 0.0,
        }
        self._usage_lock = threading.Lock()

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if "openrouter.ai" in self.base_url:
            payload["reasoning"] = {"effort": self.reasoning_effort}
            payload["usage"] = {"include": True}
            headers["X-Title"] = "Book Reader AI"
        elif "deepseek.com" in self.base_url:
            payload["thinking"] = {"type": "enabled" if self.reasoning_effort != "none" else "disabled"}

        if self.reasoning_effort == "none":
            payload["temperature"] = temperature

        max_attempts = max(1, int(os.getenv("BOOKAI_RETRY_ATTEMPTS") or "3"))
        request_timeout = max(30.0, float(os.getenv("BOOKAI_REQUEST_TIMEOUT") or "180"))
        response: httpx.Response | None = None

        for attempt in range(max_attempts):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
                retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
                if not retryable:
                    response.raise_for_status()
                    break
                if attempt + 1 >= max_attempts:
                    response.raise_for_status()
                raw_retry_after = response.headers.get("Retry-After")
                try:
                    server_delay = float(raw_retry_after) if raw_retry_after else 0.0
                except ValueError:
                    server_delay = 0.0
                delay = min(45.0, max(server_delay, min(20.0, 2.0**attempt)))
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{max_attempts} status={response.status_code} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.8))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 >= max_attempts:
                    raise
                delay = min(20.0, 2.0**attempt)
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{max_attempts} error={type(exc).__name__} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.8))

        if response is None:
            raise RuntimeError("LLM request did not produce a response")
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        with self._usage_lock:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.usage["total_tokens"] += int(usage.get("total_tokens") or 0)
            self.usage["requests"] += 1
            self.usage["cost"] += float(usage.get("cost") or 0.0)
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
    system = """You build a compact translation bible for a Russian literary translation.
You are not translating the sample. Infer only recurring evidence; never invent plot facts.
Focus on what a cheap translator must know to imitate the author's function in natural Russian:
voice, sentence rhythm, irony mechanism, internal monologue, dialogue register, technical vocabulary,
proper names, titles and recurring terms. Return ONLY valid JSON."""
    analysis_limit = max(30000, int(os.getenv("BOOKAI_ANALYSIS_CHARS") or "104000"))
    user = f"""Book title: {title!r}\nAuthor: {author!r}
Return exactly this schema:
{{
  "style": {{
    "narrative_voice":"...",
    "rhythm":"...",
    "dialogue":"...",
    "humor":"...",
    "taboos":["specific failure modes to avoid"]
  }},
  "glossary": {{"English term/name":"preferred Russian rendering"}},
  "characters": {{"name":"voice/personality cues only when evidenced"}},
  "rolling_summary":"very short factual continuity summary in Russian"
}}
Rules for glossary: include only high-confidence recurring names/terms. Do not transliterate common English words.

REPRESENTATIVE_SAMPLE:\n{sample[:analysis_limit]}"""
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Analyzer returned non-object JSON")
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


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


def chapter_brief(provider: LLMProvider, segments: list[Segment], memory: BookMemory) -> str:
    """Cheap original-only scene map. It gives the translator local intent without translating."""
    limit = max(12000, int(os.getenv("BOOKAI_CHAPTER_BRIEF_CHARS") or "30000"))
    text = "\n\n".join(s.text for s in segments)
    if len(text) > limit:
        half = limit // 2
        text = text[:half] + "\n...[middle omitted]...\n" + text[-half:]
    system = """Create a compact scene/voice brief for an EN→RU literary translator.
Do NOT translate prose and do NOT invent. Identify who is speaking/thinking, scene purpose, emotional subtext,
important irony, ambiguous references and local terminology. Return ONLY JSON {"brief":"..."}.
Maximum 350 Russian words."""
    user = f"BOOK_MEMORY:{_memory_prompt(memory)}\nCHAPTER_SOURCE:{text}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Chapter briefer returned non-object JSON")
    return str(obj.get("brief") or "").strip()[:5000]


def _target_payload(segments: list[Segment]) -> dict[str, dict[str, str]]:
    return {
        s.id: {"text": s.text, "kind": "heading" if is_heading(s) else "prose"}
        for s in segments
    }


def translate_batch(
    provider: LLMProvider,
    segments: list[Segment],
    memory: BookMemory,
    *,
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> dict[str, str]:
    system = """You are the first pass of a production EN→RU literary translation harness.
Translate meaning first, then WRITE THE SENTENCE AS NATIVE RUSSIAN PROSE rather than mirroring English syntax.
Preserve every factual relation: who did what to whom, negation, modality, chronology, numbers, names, ambiguity,
paragraph function, character voice, dry irony, profanity and deliberate awkwardness when it is authorial.
For idioms/metaphors/jokes preserve their FUNCTION and effect; literal wording is secondary.
Do not summarize, explain, embellish, censor, add motives, or repair the author's logic.
Russian dialogue should use normal literary punctuation when the target segment contains dialogue.
Context passages are reference only: never return them.
Every target id is mandatory. Never switch languages or scripts.
Return ONLY one JSON object mapping EXACTLY every target id to its Russian translation, no missing/extra ids."""
    before = [{"id": s.id, "text": s.text} for s in (context_before or [])]
    after = [{"id": s.id, "text": s.text} for s in (context_after or [])]
    user = f"""TRANSLATION_BIBLE:{_memory_prompt(memory)}
CONTEXT_BEFORE:{json.dumps(before, ensure_ascii=False)}
TARGET_SEGMENTS:{json.dumps(_target_payload(segments), ensure_ascii=False)}
CONTEXT_AFTER:{json.dumps(after, ensure_ascii=False)}
Before returning JSON, silently verify that no thought/fact/negation/number from each source segment was dropped and that the result sounds written in Russian."""
    obj = extract_json(provider.complete(system, user, temperature=0.15))
    return assert_exact_ids(segments, obj, "translator")


def quality_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    system = """You are a strict bilingual EN-RU literary QA auditor. Do NOT rewrite.
Compare source and candidate proposition by proposition, then judge Russian prose quality.
Flag: omission/addition; wrong actor/action/object; changed negation/modality/chronology; mistranslated metaphor/idiom;
flattened irony/voice; calqued or non-native Russian; terminology/name drift; broken dialogue/register.
Ignore harmless nonliteral wording. Severity hard = semantic corruption/omission or severe voice failure;
medium = genuine but repairable literary/terminology problem.
Return ONLY {"issues":[{"id":"...","severity":"medium|hard","code":"...","reason":"specific concise reason"}]}.
If a segment is faithful and naturally Russian, do not flag it."""
    pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Quality gate returned non-object JSON")
    findings: list[GateFinding] = []
    valid_ids = {s.id for s in originals}
    for raw in obj.get("issues") or []:
        if not isinstance(raw, dict) or str(raw.get("id")) not in valid_ids:
            continue
        severity = str(raw.get("severity") or "medium").lower()
        if severity not in {"medium", "hard"}:
            severity = "medium"
        code = str(raw.get("code") or "qa")
        reason = str(raw.get("reason") or "").strip()
        findings.append(GateFinding(str(raw["id"]), severity, f"{code}: {reason}"))
    return findings


def edit_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
    *,
    reasons: dict[str, str] | None = None,
    context: list[dict] | None = None,
) -> dict[str, str]:
    if not originals:
        return {}
    system = """You are the repair pass in an EN→RU literary translation harness.
The draft already exists. Fix the stated defects while preserving all correct content.
Priority order: (1) exact source meaning and relationships, (2) character/author voice and irony,
(3) idiomatic Russian syntax/rhythm, (4) terminology consistency.
Do not summarize, expand, explain or beautify beyond the source. Avoid English syntactic calques.
Use surrounding context for reference but return only target ids.
Return ONLY a JSON object mapping EXACTLY every supplied id to final Russian text."""
    pairs = {
        s.id: {
            "original": s.text,
            "draft": draft[s.id],
            "issue": (reasons or {}).get(s.id, "literary/semantic QA repair"),
        }
        for s in originals
    }
    user = (
        f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\n"
        f"SURROUNDING_CONTEXT:{json.dumps(context or [], ensure_ascii=False)}\n"
        f"PAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    )
    obj = extract_json(provider.complete(system, user, temperature=0.1))
    return assert_exact_ids(originals, obj, "editor")


def alternative_batch(provider: LLMProvider, originals: list[Segment], memory: BookMemory) -> dict[str, str]:
    """Independent second candidate for genuinely hard passages, still using the same cheap model."""
    if not originals:
        return {}
    system = """Produce an independent EN→RU literary translation for difficult passages.
Do not look for word-for-word correspondence: reconstruct the exact meaning, subtext, irony and voice in natural Russian.
No additions, omissions, explanations or foreign-script output. Return ONLY an exact id→translation JSON object."""
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nTARGETS:{json.dumps(_target_payload(originals), ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.35))
    return assert_exact_ids(originals, obj, "alternative translator")


def choose_candidate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    first: dict[str, str],
    second: dict[str, str],
    memory: BookMemory,
) -> dict[str, str]:
    """Blind A/B selection. Judge cannot rewrite, which sharply limits judge hallucination."""
    if not originals:
        return {}
    system = """Choose the better of two Russian translations for each English literary segment.
Judge exact semantic fidelity first, then preservation of irony/voice, then natural Russian literary phrasing.
Do NOT rewrite either candidate. Return ONLY JSON mapping every id to the single letter "A" or "B"."""
    pairs = {
        s.id: {"source": s.text, "A": first[s.id], "B": second[s.id]}
        for s in originals
    }
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nCANDIDATES:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Candidate judge returned non-object JSON")
    expected = {s.id for s in originals}
    if set(map(str, obj)) != expected:
        raise ValueError("Candidate judge id contract violation")
    out: dict[str, str] = {}
    for s in originals:
        choice = str(obj[s.id]).strip().upper()
        if choice not in {"A", "B"}:
            raise ValueError(f"Candidate judge returned invalid choice for {s.id}: {choice!r}")
        out[s.id] = first[s.id] if choice == "A" else second[s.id]
    return out


def qa_batch(provider: LLMProvider, originals: list[Segment], edited: dict[str, str], memory: BookMemory) -> dict[str, str]:
    """Compatibility name for a final Flash repair pass; no stronger model is required."""
    return edit_batch(provider, originals, edited, memory)


def update_memory(provider: LLMProvider, originals: list[Segment], translated: dict[str, str], memory: BookMemory) -> BookMemory:
    system = """Maintain a compact continuity bible for an EN→RU novel translation.
Update only facts and naming/terminology supported by this completed chapter. Keep established preferred translations
unless evidence proves them wrong. Track character speech cues, not plot speculation. Return ONLY valid JSON."""
    pairs = [{"id": s.id, "chapter": s.chapter, "en": s.text, "ru": translated.get(s.id, "")} for s in originals]
    user = f"""CURRENT_MEMORY:{_memory_prompt(memory)}
COMPLETED_CHAPTER:{json.dumps(pairs, ensure_ascii=False)}
Return {{"glossary":complete_merged_glossary,"characters":complete_merged_notes,
"rolling_summary":"factual continuity summary max 700 Russian words","last_chapter":"latest label"}}."""
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Memory updater returned non-object JSON")
    return BookMemory(
        title=memory.title,
        author=memory.author,
        style=memory.style,
        glossary={str(k): str(v) for k, v in (obj.get("glossary") or memory.glossary).items()},
        characters={str(k): str(v) for k, v in (obj.get("characters") or memory.characters).items()},
        rolling_summary=str(obj.get("rolling_summary") or memory.rolling_summary),
        last_chapter=str(obj.get("last_chapter") or (originals[-1].chapter if originals else memory.last_chapter)),
    )
