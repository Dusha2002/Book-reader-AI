from __future__ import annotations

import json

from .llm import _memory_prompt, extract_json
from .models import BookMemory, LLMProvider, Segment
from .quality import assert_exact_ids, is_heading


def literary_polish_batch(
    provider: LLMProvider,
    originals: list[Segment],
    draft: dict[str, str],
    memory: BookMemory,
    *,
    context: list[dict] | None = None,
) -> dict[str, str]:
    """Second, mandatory Flash pass: turn a faithful draft into Russian prose.

    The source remains visible, so this is not free-form paraphrasing. Separating
    fidelity and literary expression is deliberate: a small model is more reliable
    when those objectives are solved in two bounded steps rather than one prompt.
    """
    if not originals:
        return {}

    system = """LITERARY POLISH PASS — EN→RU fiction.
You receive the English source and an already faithful Russian draft. Rewrite the draft so it reads as if the scene
was originally written by a skilled Russian-language novelist, while the English source remains the absolute semantic
contract.

Preserve exactly: actors and referents, actions, causality, chronology, negation, modality, uncertainty, numbers,
technical meaning, deliberate repetitions, ambiguity, jokes, irony, internal-thought distance, brutality and profanity.
Do NOT summarize, explain, embellish, censor, invent motives, or add imagery absent from the source.

Actively remove translationese: English word order, excessive pronouns, noun chains, literal idioms, repeated
"это было/он был" constructions, unnatural participles and mechanically mirrored punctuation. Reconstruct sentence
rhythm in idiomatic literary Russian. Preserve the FUNCTION of metaphors and dry jokes rather than their word order.
Keep established names and terminology from the translation bible. Dialogue must follow normal Russian literary
punctuation and the speaker's established register.

For a heading, output only a concise Russian heading — never body prose.
Context is read-only. Return ONLY a JSON object mapping EXACTLY every target id to the polished Russian text."""

    pairs = {
        s.id: {
            "kind": "heading" if is_heading(s) else "prose",
            "source_en": s.text,
            "faithful_draft_ru": draft[s.id],
        }
        for s in originals
    }
    user = (
        f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\n"
        f"SURROUNDING_CONTEXT:{json.dumps(context or [], ensure_ascii=False)}\n"
        f"TARGETS:{json.dumps(pairs, ensure_ascii=False)}\n"
        "Silently compare the final Russian against every proposition in source_en before returning it."
    )
    obj = extract_json(provider.complete(system, user, temperature=0.25))
    return assert_exact_ids(originals, obj, "literary polisher")
