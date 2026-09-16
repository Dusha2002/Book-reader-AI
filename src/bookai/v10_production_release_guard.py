from __future__ import annotations

import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue


_SOURCE_ATTRIBUTION = (
    r"said|asked|replied|answered|thought|remarked|observed|whispered|"
    r"shouted|called|added|continued"
)
_RU_FEMININE_ATTRIBUTION = (
    r"сказала|говорила|ответила|спросила|подумала|заметила|добавила|"
    r"продолжила|прошептала|крикнула"
)
_RU_MASCULINE_ATTRIBUTION = (
    r"сказал|говорил|ответил|спросил|подумал|заметил|добавил|"
    r"продолжил|прошептал|крикнул"
)

_EN_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_EN_SIMPLE_NUMBER = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"thirty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"forty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"fifty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"sixty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"seventy(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"eighty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"ninety(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
)
_BETWEEN_RANGE_RE = re.compile(
    rf"\bbetween\s+(?P<left>{_EN_SIMPLE_NUMBER})\s+and\s+(?P<right>{_EN_SIMPLE_NUMBER})\b",
    re.I,
)


def _simple_number_value(text: str) -> int | None:
    parts = re.split(r"[-\s]+", str(text or "").casefold().strip())
    if not parts:
        return None
    if len(parts) == 1:
        return _EN_ONES.get(parts[0], _EN_TENS.get(parts[0]))
    if len(parts) == 2 and parts[0] in _EN_TENS and parts[1] in _EN_ONES:
        return _EN_TENS[parts[0]] + _EN_ONES[parts[1]]
    return None


def spurious_between_range_sums(source: str) -> set[int]:
    """Values a legacy parser can invent by summing both endpoints of a range."""
    out: set[int] = set()
    for match in _BETWEEN_RANGE_RE.finditer(str(source or "")):
        left = _simple_number_value(match.group("left"))
        right = _simple_number_value(match.group("right"))
        if left is not None and right is not None:
            out.add(left + right)
    return out


def clean_numeric_result(source: str, result: dict[str, Any]) -> dict[str, Any]:
    """Drop only impossible synthetic range sums from numeric missing-values."""
    cleaned = dict(result)
    spurious = spurious_between_range_sums(source)
    if not spurious:
        return cleaned
    cleaned["missing"] = [
        value for value in list(cleaned.get("missing") or []) if value not in spurious
    ]
    cleaned["ok"] = not cleaned["missing"]
    cleaned["production_suppressed_range_sums"] = sorted(spurious)
    return cleaned


def clean_quantity_result(source: str, result: dict[str, Any]) -> dict[str, Any]:
    """Apply the same fail-closed range correction to v10 quantity diagnostics."""
    cleaned = dict(result)
    spurious = spurious_between_range_sums(source)
    if not spurious:
        return cleaned
    cleaned["base_missing"] = [
        value for value in list(cleaned.get("base_missing") or []) if value not in spurious
    ]
    cleaned["missing_mentions"] = [
        row
        for row in list(cleaned.get("missing_mentions") or [])
        if row.get("value") not in spurious
    ]
    base = cleaned.get("base")
    if isinstance(base, dict):
        base_clean = dict(base)
        base_clean["missing"] = [
            value for value in list(base_clean.get("missing") or []) if value not in spurious
        ]
        base_clean["ok"] = not base_clean["missing"]
        cleaned["base"] = base_clean
    cleaned["ok"] = (
        not cleaned["base_missing"]
        and not cleaned["missing_mentions"]
        and not cleaned.get("numbered_choice_missing")
    )
    cleaned["production_suppressed_range_sums"] = sorted(spurious)
    return cleaned


def _source_has_direct_actor(source: str, name: str) -> bool:
    escaped = re.escape(name)
    return bool(
        re.search(
            rf"(?<![A-Za-z]){escaped}(?![A-Za-z])[^.!?]{{0,42}}\b(?:{_SOURCE_ATTRIBUTION})\b",
            source,
            re.I,
        )
        or re.search(
            rf"\b(?:{_SOURCE_ATTRIBUTION})\b[^.!?]{{0,24}}(?<![A-Za-z]){escaped}(?![A-Za-z])",
            source,
            re.I,
        )
    )


def _russian_canon_stem(name: str, desc: str, memory: BookMemory) -> str:
    match = re.search(r"(?:^|;)ru=([^;]+)", str(desc or ""), re.I)
    canon = str(match.group(1) if match else memory.glossary.get(name, "") or "").strip()
    words = re.findall(r"[А-Яа-яЁё]+", canon)
    if not words:
        return ""
    word = words[0].casefold().replace("ё", "е")
    return word[: max(3, len(word) - 2)]


def localized_gender_issues(
    segment: Segment,
    target: str,
    memory: BookMemory,
) -> list[V10Issue]:
    """Flag gender only when the wrong Russian attribution is local to that character.

    A paragraph can contain several speakers. Scanning the whole paragraph for any
    feminine/masculine speech verb creates false positives when a different speaker
    appears earlier in the same segment.
    """
    source = str(segment.text or "")
    low = str(target or "").casefold().replace("ё", "е")
    out: list[V10Issue] = []

    for name, desc in memory.characters.items():
        name_s = str(name or "").strip()
        if not name_s:
            continue
        if not re.search(
            rf"(?<![A-Za-z]){re.escape(name_s)}(?![A-Za-z])", source, re.I
        ):
            continue
        if not _source_has_direct_actor(source, name_s):
            continue

        gender_match = re.search(r"gender=(male|female)", str(desc or ""), re.I)
        if not gender_match:
            continue
        stem = _russian_canon_stem(name_s, str(desc or ""), memory)
        if not stem:
            continue

        name_pattern = rf"\b{re.escape(stem)}[а-я]*\b"
        gender = gender_match.group(1).casefold()
        wrong = (
            _RU_FEMININE_ATTRIBUTION
            if gender == "male"
            else _RU_MASCULINE_ATTRIBUTION
        )
        mismatch = bool(
            re.search(
                rf"{name_pattern}[^.!?…]{{0,42}}\b(?:{wrong})\b",
                low,
            )
            or re.search(
                rf"\b(?:{wrong})\b[^.!?…]{{0,24}}{name_pattern}",
                low,
            )
        )
        if mismatch:
            out.append(
                V10Issue(
                    segment.id,
                    "character_gender",
                    "local",
                    "hard",
                    f"{name_s} has a locally mismatched Russian attribution gender",
                )
            )
    return out
