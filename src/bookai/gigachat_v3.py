from __future__ import annotations

import json
import re

from .gigachat_mt import GigaChatLightningBackend
from .models import BookMemory, Segment
from .quality_v3 import speaker_metadata


_CHAPTER_RU = {
    "one": "один", "two": "два", "three": "три", "four": "четыре", "five": "пять",
    "six": "шесть", "seven": "семь", "eight": "восемь", "nine": "девять", "ten": "десять",
    "eleven": "одиннадцать", "twelve": "двенадцать", "thirteen": "тринадцать",
    "fourteen": "четырнадцать", "fifteen": "пятнадцать", "sixteen": "шестнадцать",
    "seventeen": "семнадцать", "eighteen": "восемнадцать", "nineteen": "девятнадцать",
    "twenty": "двадцать", "twenty-one": "двадцать один", "twenty one": "двадцать один",
    "twenty-two": "двадцать два", "twenty two": "двадцать два",
    "twenty-three": "двадцать три", "twenty three": "двадцать три",
    "twenty-four": "двадцать четыре", "twenty four": "двадцать четыре",
    "twenty-five": "двадцать пять", "twenty five": "двадцать пять",
    "twenty-six": "двадцать шесть", "twenty six": "двадцать шесть",
    "twenty-seven": "двадцать семь", "twenty seven": "двадцать семь",
    "twenty-eight": "двадцать восемь", "twenty eight": "двадцать восемь",
    "twenty-nine": "двадцать девять", "twenty nine": "двадцать девять",
    "thirty": "тридцать",
}


class GigaChatLightningV3Backend(GigaChatLightningBackend):
    """Quality-v3 wrapper around the hardened Lightning backend.

    Changes are deliberately small and measurable:
    * chapter headings are deterministic based on text, not fragile XML locator shape;
    * each target may carry first-person speaker/gender metadata inferred from recent
      epistolary salutations, preventing `я уверен` for a female letter writer;
    * context and speaker metadata remain explicitly non-translatable.
    """

    name = "gigachat-3-lightning-v3"

    @staticmethod
    def _deterministic_heading(segment: Segment) -> str | None:
        match = re.fullmatch(r"\s*Chapter\s+(.+?)\s*", segment.text, flags=re.I)
        if not match:
            return None
        raw = " ".join(match.group(1).strip().casefold().split())
        if raw.isdigit():
            return f"Глава {raw}"
        translated = _CHAPTER_RU.get(raw)
        return f"Глава {translated}" if translated else None

    def _prompt(
        self,
        batch: list[Segment],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
        minimal: bool = False,
    ) -> str:
        base = super()._prompt(
            batch,
            memory,
            source_segments=source_segments,
            minimal=minimal,
        )
        metadata = speaker_metadata(batch, source_segments, memory)
        if not metadata:
            return base
        return (
            base
            + "\n\nTARGET_METADATA (НЕ ПЕРЕВОДИТЬ; использовать только для согласования рода/голоса):\n"
            + json.dumps(metadata, ensure_ascii=False)
            + "\nДля target без записи в TARGET_METADATA ничего не выдумывай."
        )
