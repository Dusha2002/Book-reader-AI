from __future__ import annotations

import json
import re
from dataclasses import asdict

from .llm import chapter_brief as _raw_chapter_brief, extract_json, update_memory as _raw_update_memory
from .models import BookMemory, GateFinding, LLMProvider, Segment, StyleGuide

# Scripts that have no legitimate reason to appear in a Russian scene brief.
# Latin is deliberately allowed for source terms/names.
_FOREIGN_SCRIPT = re.compile(
    r"[\u0370-\u03ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
    r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)
_SECTION = re.compile(r"(?m)(?=^\s*###\s+)")
_PROPERISH = re.compile(r"\b[A-Z][A-Za-z'’-]{2,}\b")
_COMMON_CAPS = {
    "The", "This", "That", "Then", "There", "When", "While", "With", "Without", "What", "Why",
    "How", "He", "She", "His", "Her", "They", "Their", "It", "Its", "I", "We", "You", "But",
    "And", "Or", "If", "As", "At", "By", "For", "From", "In", "Into", "No", "Not", "Of", "On",
    "So", "To", "Was", "Were", "A", "An",
}


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


def _slice_text(text: str, budget: int) -> str:
    text = text.strip()
    if len(text) <= budget:
        return text
    if budget < 900:
        return text[:budget]
    each = max(250, budget // 3)
    middle_start = max(0, len(text) // 2 - each // 2)
    return (
        text[:each]
        + "\n...[middle]...\n"
        + text[middle_start : middle_start + each]
        + "\n...[end]...\n"
        + text[-each:]
    )[:budget]


def _representative_slice(sample: str, budget: int) -> str:
    """Shrink a book sample without silently dropping whole chapter sections.

    `_analysis_sample` already labels chapter excerpts with `###`. If we simply
    take beginning/middle/end again, most chapters disappear from the global
    translation bible. Allocate the budget across every labelled section first;
    fall back to ordinary beginning/middle/end only for unstructured text.
    """
    sample = sample.strip()
    if len(sample) <= budget:
        return sample
    sections = [part.strip() for part in _SECTION.split(sample) if part.strip()]
    if len(sections) >= 4:
        per = max(500, budget // len(sections))
        packed = [_slice_text(section, per) for section in sections]
        return "\n".join(packed)[:budget]
    return _slice_text(sample, budget)


def safe_analyze_memory(
    provider: LLMProvider,
    sample: str,
    *,
    title: str = "",
    author: str = "",
    attempts: int = 3,
) -> BookMemory:
    """Build a compact translation bible and retry malformed/runaway JSON."""
    budgets = (30000, 18000, 10000)
    last_error: BaseException | None = None
    system = """You build a COMPACT translation bible for a Russian literary translation.
Do not translate the sample. Infer only recurring evidence and never invent plot facts.
Focus on narrative voice, sentence rhythm, irony mechanism, dialogue register, technical vocabulary,
proper names/titles, recurring terms and character speech cues. Return ONLY valid compact JSON.
The entire response must stay under 1200 words. No prose outside JSON."""

    for attempt in range(max(1, attempts)):
        budget = budgets[min(attempt, len(budgets) - 1)]
        excerpt = _representative_slice(sample, budget)
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
Rules: keep glossary high-confidence and compact; do not transliterate common English words.
REPRESENTATIVE_SAMPLE:\n{excerpt}"""
        try:
            raw = provider.complete(system, user, temperature=0.0)
            if len(raw) > 24000:
                raise ValueError(f"Analyzer runaway output: {len(raw)} chars")
            obj = extract_json(raw)
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
        except BaseException as exc:
            last_error = exc
            print(
                f"[bookai-analysis-retry] attempt={attempt + 1}/{max(1, attempts)} "
                f"sample_chars={len(excerpt)} error={type(exc).__name__}",
                flush=True,
            )

    raise ValueError("Book analysis repeatedly returned malformed/runaway output") from last_error


def brief_is_corrupt(text: str) -> bool:
    """Reject mixed-script/model-garbage briefs before they contaminate translation context."""
    if not text or len(text.strip()) < 15:
        return True
    if _FOREIGN_SCRIPT.search(text):
        return True
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    alpha = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
    if alpha >= 80 and cyr / max(alpha, 1) < 0.55:
        return True
    garbage_markers = ("assistant", "system prompt", "дaвайте честно", "давайте честно", "перегрузил контекст")
    low = text.lower()
    return any(marker in low for marker in garbage_markers)


def _fallback_chapter_brief() -> str:
    """Neutral fail-safe: never inject guessed plot facts when the briefer fails."""
    return (
        "Надёжный локальный бриф не получен. Опирайся только на исходный текст, соседний контекст "
        "и translation bible; сохраняй точный смысл, голос и иронию и не добавляй фактов или мотивов, "
        "которых нет в оригинале."
    )


def safe_chapter_brief(
    provider: LLMProvider,
    segments: list[Segment],
    memory: BookMemory,
    *,
    attempts: int = 2,
) -> str:
    """Build a chapter brief without allowing a control-plane call to block the book."""
    for attempt in range(max(1, attempts)):
        try:
            candidate = _raw_chapter_brief(provider, segments, memory)
            if not brief_is_corrupt(candidate):
                return candidate
            reason = "corrupt_output"
        except BaseException as exc:
            reason = type(exc).__name__
        print(
            f"[bookai-brief-retry] attempt={attempt + 1}/{max(1, attempts)} reason={reason}",
            flush=True,
        )
    print("[bookai-brief-fallback] using deterministic source-only guidance", flush=True)
    return _fallback_chapter_brief()


def _chapter_memory_sample(
    originals: list[Segment],
    translated: dict[str, str],
    memory: BookMemory,
    budget: int,
) -> list[Segment]:
    """Pick bounded, ordered evidence for continuity updates.

    Prioritise lines likely to contain new proper names/dialogue, then fill the
    remainder with uniformly spaced evidence so beginning/middle/end all remain
    represented. The returned Segment objects are unchanged; `_raw_update_memory`
    reads their translations from the full translation dict.
    """
    if not originals:
        return []
    known = {key.casefold() for key in memory.glossary}
    priority: list[int] = []
    for idx, segment in enumerate(originals):
        names = {
            token for token in _PROPERISH.findall(segment.text)
            if token not in _COMMON_CAPS and token.casefold() not in known
        }
        dialogue = '"' in segment.text or "'" in segment.text or "“" in segment.text or "”" in segment.text
        if names or dialogue:
            priority.append(idx)

    uniform_count = min(len(originals), 36)
    if uniform_count <= 1:
        uniform = [0]
    else:
        uniform = [round(i * (len(originals) - 1) / (uniform_count - 1)) for i in range(uniform_count)]

    ordered_candidates: list[int] = []
    # Interleave priority with broad coverage instead of letting dialogue-heavy
    # scenes consume the whole budget.
    for pos in range(max(len(priority), len(uniform))):
        if pos < len(uniform):
            ordered_candidates.append(uniform[pos])
        if pos < len(priority):
            ordered_candidates.append(priority[pos])

    selected: set[int] = set()
    used = 0
    for idx in ordered_candidates:
        if idx in selected:
            continue
        segment = originals[idx]
        cost = len(segment.text) + len(translated.get(segment.id, "")) + 80
        if selected and used + cost > budget:
            continue
        selected.add(idx)
        used += cost
        if used >= budget:
            break

    # Always retain chapter edges when possible.
    selected.add(0)
    selected.add(len(originals) - 1)
    return [originals[idx] for idx in sorted(selected)]


def safe_update_memory(
    provider: LLMProvider,
    originals: list[Segment],
    translated: dict[str, str],
    memory: BookMemory,
    *,
    attempts: int = 2,
) -> BookMemory:
    """Update continuity from bounded evidence; retry smaller rather than sending a whole huge chapter."""
    budgets = (24000, 12000)
    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        budget = budgets[min(attempt, len(budgets) - 1)]
        sample = _chapter_memory_sample(originals, translated, memory, budget)
        try:
            updated = _raw_update_memory(provider, sample, translated, memory)
            print(
                f"[bookai-memory-update] evidence_segments={len(sample)}/{len(originals)} budget={budget}",
                flush=True,
            )
            return updated
        except BaseException as exc:
            last_error = exc
            print(
                f"[bookai-memory-retry] attempt={attempt + 1}/{max(1, attempts)} "
                f"evidence_segments={len(sample)} error={type(exc).__name__}",
                flush=True,
            )
    raise ValueError("Chapter memory update repeatedly failed") from last_error


def semantic_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    """Independent fidelity audit intentionally separated from literary criticism."""
    if not originals:
        return []
    system = """You are an adversarial EN→RU SEMANTIC COVERAGE auditor. Do NOT rewrite and do NOT judge style.
For EACH source segment, silently decompose the English into atomic obligations and compare them with the Russian.
Check especially:
1. EVERY item in enumerations/lists (A, B, C, D must not become only three items, even for obscure technical terms);
2. actor → action → object relations and pronoun referents;
3. all numbers, quantities, comparisons, direction, chronology and cause/effect;
4. negation, modality, uncertainty, conditionals and degree/intensity;
5. proper names and technical terms when a changed referent changes meaning;
6. concrete physical images/actions (through/into, hit/miss, enter/pass, etc.);
7. any source proposition, joke premise, contrast or qualification that disappeared or was invented.
Do not flag harmless paraphrase, Russian word order, or stylistic choices if meaning is intact.
A missing/added source fact, missing enumeration member, wrong relation/referent, or changed technical referent is HARD.
Return ONLY JSON: {"issues":[{"id":"...","code":"semantic_omission|semantic_addition|relation|enumeration|term|modality|other","reason":"specific source obligation that failed"}]}.
If fully faithful, return {"issues":[]}."""
    pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    if not isinstance(obj, dict):
        raise ValueError("Semantic gate returned non-object JSON")
    valid_ids = {s.id for s in originals}
    findings: list[GateFinding] = []
    for raw in obj.get("issues") or []:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("id") or "")
        if sid not in valid_ids:
            continue
        code = str(raw.get("code") or "semantic")
        reason = str(raw.get("reason") or "").strip()
        findings.append(GateFinding(sid, "hard", f"{code}: {reason}"))
    return findings
