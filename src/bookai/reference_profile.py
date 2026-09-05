from __future__ import annotations

from dataclasses import replace

from .models import BookMemory, StyleGuide


# Derived from the user-provided GPT-5.6 Sol Russian reference translation.
# It is intentionally compact: the full reference book is NOT copied into prompts.
REFERENCE_STYLE_CONTRACT = """Target Russian literary register for this novel:
- restrained, precise, dryly ironic prose; intelligent but not ornate, archaizing, or self-consciously "beautiful";
- preserve the author's sardonic timing and understatement: the punch line often lands at the end of a sentence/paragraph;
- natural Russian syntax is mandatory. Never mirror English grammar when it produces calques, bureaucratese, cognate-literal wording or awkward pronoun chains;
- preserve long sentence architecture and accumulating lists when they carry rhythm, but rebuild them with idiomatic Russian government and punctuation;
- prefer concrete, period-neutral literary vocabulary over modern corporate/technical Anglicisms unless the source itself is technical;
- translate idioms by function and context, never by lexical resemblance (e.g. avoid literal constructions like «ради чёрта» for English idioms);
- action and engineering prose must remain physically precise: actor, motion, direction, mechanism, scale and cause/effect cannot blur;
- irony must stay dry rather than become louder, slangier or more emotional than the source;
- internal thought should remain close, concise and unsentimental; do not explain the joke or character psychology;
- dialogue should sound spoken yet controlled, with standard Russian literary punctuation and stable character register;
- preserve deliberate repetitions when they create rhythm or irony; remove only accidental translationese repetitions;
- do not upgrade neutral words into melodramatic ones and do not flatten sharp images into generic phrasing;
- terminology and names must remain stable across the book. If uncertain, prefer continuity over improvisation.
"""

REFERENCE_GLOSSARY_SEED = {
    "Valens": "Валенс",
    "Orsea": "Орсеа",
    "Melancton": "Меланктон",
    "Syracoelus": "Сиракоэл",
    "Eremia": "Эремия",
    "Eremians": "эремийцы",
    "Perpetual Republic": "Вечная Республика",
}


def apply_reference_profile(memory: BookMemory) -> BookMemory:
    style = StyleGuide(
        narrative_voice=(
            memory.style.narrative_voice
            + "\nREFERENCE TARGET: "
            + REFERENCE_STYLE_CONTRACT
        )[-9000:],
        rhythm=(
            memory.style.rhythm
            + "\nKeep Parker's long accumulative sentences when functional; rebuild them as idiomatic Russian rather than chopping or mirroring English."
        )[-4000:],
        dialogue=(
            memory.style.dialogue
            + "\nUse restrained, natural Russian dialogue; do not make characters more colloquial, theatrical, or modern than the source."
        )[-4000:],
        humor=(
            memory.style.humor
            + "\nDry irony and understatement outrank literal wording. Keep punch-line placement and tonal restraint."
        )[-4000:],
        taboos=list(dict.fromkeys([
            *memory.style.taboos,
            "No English-syntax calques or bureaucratic Russian unless deliberately present in the source.",
            "No literal idiom translation when natural Russian would express the same function.",
            "No modern corporate/consumer vocabulary in neutral historical-fantasy narration unless required by the source.",
            "Do not intensify neutral narration into melodrama or slang.",
            "Do not flatten technical or physical relations into generic wording.",
        ])),
    )
    glossary = dict(REFERENCE_GLOSSARY_SEED)
    glossary.update(memory.glossary)
    # Reference spellings win only for the explicit benchmark/major recurring terms above.
    glossary.update(REFERENCE_GLOSSARY_SEED)
    return replace(memory, style=style, glossary=glossary)
