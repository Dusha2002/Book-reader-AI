from __future__ import annotations

import json
from dataclasses import asdict

from .contracts import issue_rows
from .llm import extract_json
from .models import BookMemory, GateFinding, LLMProvider, Segment


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


def semantic_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    """Adversarial fidelity critic with a strict, normalized issues contract."""
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
Return ONLY JSON with key "issues". "issues" MUST ALWAYS BE A JSON ARRAY; for no problems use exactly [].
Schema: {"issues":[{"id":"...","code":"semantic_omission|semantic_addition|relation|enumeration|term|modality|other","reason":"specific source obligation that failed"}]}.
Never return an integer issue count instead of the array."""
    pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    valid_ids = {s.id for s in originals}
    findings: list[GateFinding] = []
    for raw in issue_rows(obj, "semantic gate"):
        sid = str(raw.get("id") or "")
        if sid not in valid_ids:
            raise ValueError(f"semantic gate returned unknown id: {sid!r}")
        code = str(raw.get("code") or "semantic")
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"semantic gate returned empty reason for {sid}")
        findings.append(GateFinding(sid, "hard", f"{code}: {reason}"))
    return findings


def literary_gate_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
) -> list[GateFinding]:
    """Literary Russian/voice critic deliberately separated from semantic QA."""
    if not originals:
        return []
    system = """You are a strict bilingual EN→RU LITERARY QUALITY auditor. Do NOT rewrite.
Semantic correctness is checked elsewhere; still flag obvious meaning damage if noticed, but focus on whether the Russian
reads like intentional literary prose rather than a translation draft.
For each pair check:
- English syntactic calques, wooden noun chains, unnatural government/collocations;
- rhythm and sentence architecture: do not flatten deliberate long/short contrasts;
- dry irony, understatement, sardonic timing and punch-line placement;
- narrator and character register/voice, including bluntness, formality and profanity;
- idioms/metaphors/jokes: preserve their function/effect rather than awkward literal wording;
- dialogue punctuation and natural spoken Russian;
- inconsistent names/technical terminology against the translation bible.
Do NOT demand prettier prose than the source and do NOT rewrite deliberate awkwardness that is clearly authorial.
Severity hard only for severe voice/function destruction or obvious semantic corruption; medium for a real literary defect.
Return ONLY JSON with key "issues". "issues" MUST ALWAYS BE A JSON ARRAY; for no problems use exactly [].
Schema: {"issues":[{"id":"...","severity":"medium|hard","code":"calque|rhythm|voice|irony|idiom|dialogue|term|meaning|other","reason":"specific concise defect"}]}.
Never return an integer issue count instead of the array."""
    pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
    user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    valid_ids = {s.id for s in originals}
    findings: list[GateFinding] = []
    for raw in issue_rows(obj, "literary gate"):
        sid = str(raw.get("id") or "")
        if sid not in valid_ids:
            raise ValueError(f"literary gate returned unknown id: {sid!r}")
        severity = str(raw.get("severity") or "medium").lower()
        if severity not in {"medium", "hard"}:
            raise ValueError(f"literary gate returned invalid severity for {sid}: {severity!r}")
        code = str(raw.get("code") or "literary")
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"literary gate returned empty reason for {sid}")
        findings.append(GateFinding(sid, severity, f"{code}: {reason}"))
    return findings
