from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class Segment:
    id: str
    text: str
    locator: str
    chapter: str = ""


@dataclass(slots=True)
class StyleGuide:
    narrative_voice: str = "Preserve the original author's voice and register."
    rhythm: str = "Preserve sentence rhythm and paragraph structure where natural in Russian."
    dialogue: str = "Preserve each character's individual speech patterns."
    humor: str = "Adapt jokes and wordplay for meaning and effect rather than literally."
    taboos: list[str] = field(default_factory=lambda: [
        "Do not simplify the author's ideas.",
        "Do not add explanations absent from the original.",
        "Do not censor profanity or intensity.",
    ])


@dataclass(slots=True)
class BookMemory:
    title: str = ""
    author: str = ""
    style: StyleGuide = field(default_factory=StyleGuide)
    glossary: dict[str, str] = field(default_factory=dict)
    # Source-derived publication policy for technical abbreviations. Values are
    # the canonical form expected in Russian text: e.g. preserve a Latin acronym
    # or use an established localized abbreviation. Empty for ordinary fiction.
    acronyms: dict[str, str] = field(default_factory=dict)
    characters: dict[str, str] = field(default_factory=dict)
    rolling_summary: str = ""
    last_chapter: str = ""


@dataclass(slots=True)
class BookDocument:
    source: Path
    format: str
    segments: list[Segment]
    payload: object


@dataclass(slots=True)
class GateFinding:
    id: str
    severity: str = "medium"  # medium | hard
    reason: str = ""


class LLMProvider(Protocol):
    model: str
    usage: dict

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str: ...


class SegmentTranslator(Protocol):
    name: str

    def translate(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        context_before: list[Segment] | None = None,
        context_after: list[Segment] | None = None,
    ) -> dict[str, str]: ...
