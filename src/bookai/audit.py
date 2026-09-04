from __future__ import annotations

import json
import re
from dataclasses import asdict

from .llm import chapter_brief as _raw_chapter_brief, extract_json
from .models import BookMemory, GateFinding, LLMProvider, Segment

# Scripts that have no legitimate reason to appear in a Russian scene brief.
# Latin is deliberately allowed for source terms/names.
_FOREIGN_SCRIPT = re.compile(
    r"[\u0370-\u03ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
    r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


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


def safe_chapter_brief(
    provider: LLMProvider,
    segments: list[Segment],
    memory: BookMemory,
    *,
    attempts: int = 3,
) -> str:
    """Build a chapter brief but never accept obvious model corruption."""
    last: str = ""
    for attempt in range(max(1, attempts)):
        last = _raw_chapter_brief(provider, segments, memory)
        if not brief_is_corrupt(last):
            return last
        print(
            f"[bookai-brief-retry] attempt={attempt + 1}/{max(1, attempts)} reason=corrupt_output",
            flush=True,
        )
    raise ValueError("Chapter brief repeatedly contained mixed-script/corrupted output")


def semantic_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    """Independent fidelity audit intentionally separated from literary criticism.

    Small models are much more reliable when they do one thing at a time. This
    pass ignores elegance and checks whether every semantic obligation survived.
    """
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
