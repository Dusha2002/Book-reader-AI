from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


_EN_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")
_RU_TOKEN_RE = re.compile(r"[А-Яа-яЁё]+|\d+(?:[.,]\d+)?")
_DIGIT_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_JOINER_RE = re.compile(r"^[\s-]+$")

_EN_CARDINAL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_ORDINAL = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80,
    "ninetieth": 90,
}
_EN_SCALE = {"hundred": 100, "thousand": 1000}
_EN_WORDS = set(_EN_CARDINAL) | set(_EN_ORDINAL) | set(_EN_SCALE) | {"and"}
_EN_MARKERS = {
    "number", "no", "chapter", "street", "road", "avenue", "lane", "house", "door",
    "address", "room", "rooms", "floor", "floors", "storey", "storeys", "story", "stories",
    "page", "pages", "section", "sections", "part", "parts", "volume", "volumes", "book", "books",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "century", "centuries",
    "island", "rank", "grade",
}

_RU_CARDINAL = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одно": 1, "одну": 1, "одного": 1, "одному": 1, "одним": 1, "одном": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2,
    "три": 3, "трех": 3, "трёх": 3, "трем": 3, "трём": 3, "тремя": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4, "четырем": 4, "четырём": 4, "четырьмя": 4,
    "пять": 5, "пяти": 5, "пятью": 5, "шесть": 6, "шести": 6, "шестью": 6,
    "семь": 7, "семи": 7, "семью": 7, "восемь": 8, "восьми": 8, "восемью": 8,
    "девять": 9, "девяти": 9, "девятью": 9,
    "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12, "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16, "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18, "девятнадцать": 19, "девятнадцати": 19,
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
    "сто": 100, "ста": 100, "двести": 200, "двухсот": 200,
    "триста": 300, "трехсот": 300, "трёхсот": 300,
    "четыреста": 400, "четырехсот": 400, "четырёхсот": 400,
    "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "шестисот": 600,
    "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800,
    "девятьсот": 900, "девятисот": 900,
    "двое": 2, "двоих": 2, "трое": 3, "троих": 3, "четверо": 4, "четверых": 4,
    "пятеро": 5, "пятерых": 5, "шестеро": 6, "шестерых": 6,
    "семеро": 7, "семерых": 7, "восьмеро": 8, "восьмерых": 8,
    "девятеро": 9, "девятерых": 9, "десятеро": 10, "десятерых": 10,
}
_RU_SCALE = {"тысяча": 1000, "тысячи": 1000, "тысячу": 1000, "тысяч": 1000}
# Do not include "тысячн-": in "тридцать тысячных" it is a fractional
# denominator, not multiplication by 1000.
_RU_ORDINAL_STEMS = [
    ("девяност", 90), ("восьмидесят", 80), ("семидесят", 70),
    ("шестидесят", 60), ("пятидесят", 50), ("сороков", 40),
    ("тридцат", 30), ("двадцат", 20), ("девятнадцат", 19),
    ("восемнадцат", 18), ("семнадцат", 17), ("шестнадцат", 16),
    ("пятнадцат", 15), ("четырнадцат", 14), ("тринадцат", 13),
    ("двенадцат", 12), ("одиннадцат", 11), ("десят", 10),
    ("девят", 9), ("восьм", 8), ("седьм", 7), ("шест", 6),
    ("пят", 5), ("четверт", 4), ("треть", 3), ("трет", 3),
    ("втор", 2), ("перв", 1),
]


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


def _tokens(text: str, pattern: re.Pattern[str]) -> list[_Token]:
    return [_Token(m.group(0).casefold(), m.start(), m.end()) for m in pattern.finditer(text or "")]


def _digit_value(token: str) -> int | float | None:
    if not _DIGIT_RE.fullmatch(token):
        return None
    try:
        value = float(token.replace(",", "."))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _joined(text: str, left: _Token, right: _Token) -> bool:
    return bool(_JOINER_RE.fullmatch(text[left.end:right.start]))


def _parse_en(words: list[str]) -> int | None:
    total = 0
    current = 0
    seen = False
    for word in words:
        if word == "and":
            continue
        if word in _EN_CARDINAL:
            current += _EN_CARDINAL[word]
            seen = True
        elif word in _EN_ORDINAL:
            current += _EN_ORDINAL[word]
            seen = True
        elif word == "hundred":
            current = max(1, current) * 100
            seen = True
        elif word == "thousand":
            total += max(1, current) * 1000
            current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def _english_values(text: str) -> list[int | float]:
    toks = _tokens(text, _EN_TOKEN_RE)
    out: list[int | float] = []
    i = 0
    while i < len(toks):
        digit = _digit_value(toks[i].text)
        if digit is not None:
            out.append(digit)
            i += 1
            continue
        if toks[i].text not in _EN_WORDS:
            i += 1
            continue

        start = i
        words = [toks[i].text]
        seen_ordinal = toks[i].text in _EN_ORDINAL
        j = i + 1
        while j < len(toks):
            if not _joined(text, toks[j - 1], toks[j]):
                break
            nxt = toks[j].text
            if nxt not in _EN_WORDS or seen_ordinal:
                break
            words.append(nxt)
            if nxt in _EN_ORDINAL:
                seen_ordinal = True
            j += 1

        value = _parse_en(words)
        if value is not None:
            previous = toks[start - 1].text if start else ""
            following = toks[j].text if j < len(toks) else ""
            numeric_words = [word for word in words if word != "and"]
            has_ordinal = any(word in _EN_ORDINAL for word in words)
            marker_context = previous in _EN_MARKERS or following in _EN_MARKERS
            if has_ordinal:
                # Generic "first time", "at first sight", "third cousin" may
                # legitimately lexicalize in Russian. Hard-lock ordinals only in
                # numbered entities or true compound ordinals (e.g. twenty-eighth).
                strong = marker_context or len(numeric_words) >= 2
            else:
                # Single "one" is frequently pronominal; other cardinals are
                # overwhelmingly real quantities. Compounds such as one hundred
                # remain strict facts.
                strong = value != 1 or len(numeric_words) >= 2 or marker_context
            if strong:
                out.append(value)
        i = max(j, i + 1)
    return out


def _ru_word_value(word: str) -> tuple[int, str] | None:
    normalized = word.replace("ё", "е")
    for key, value in _RU_CARDINAL.items():
        if normalized == key.replace("ё", "е"):
            return value, "cardinal"
    for key, value in _RU_SCALE.items():
        if normalized == key.replace("ё", "е"):
            return value, "scale"
    for stem, value in _RU_ORDINAL_STEMS:
        if normalized.startswith(stem.replace("ё", "е")) and len(normalized) >= len(stem) + 1:
            return value, "ordinal"
    return None


def _parse_ru(items: list[tuple[int, str]]) -> int | None:
    if not items:
        return None
    total = 0
    current = 0
    for value, kind in items:
        if kind == "scale" and value == 1000:
            total += max(1, current) * 1000
            current = 0
        else:
            current += value
    return total + current


def _russian_values(text: str) -> list[int | float]:
    toks = _tokens(text, _RU_TOKEN_RE)
    out: list[int | float] = []
    i = 0
    while i < len(toks):
        digit = _digit_value(toks[i].text)
        if digit is not None:
            out.append(digit)
            i += 1
            continue
        first = _ru_word_value(toks[i].text)
        if first is None:
            i += 1
            continue

        items = [first]
        seen_ordinal = first[1] == "ordinal"
        j = i + 1
        while j < len(toks):
            if not _joined(text, toks[j - 1], toks[j]):
                break
            nxt = _ru_word_value(toks[j].text)
            if nxt is None or seen_ordinal:
                break
            items.append(nxt)
            if nxt[1] == "ordinal":
                seen_ordinal = True
            j += 1
        value = _parse_ru(items)
        if value is not None:
            out.append(value)
        i = max(j, i + 1)
    return out


def compare_numeric_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    """Return deterministic EN→RU numeric-fact preservation diagnostics.

    This is deliberately narrower than semantic QA: it locks objective quantities
    and numbered entities while avoiding generic ordinals that Russian may render
    idiomatically ("first time"→"впервые"). Punctuation is part of the grammar,
    so independent expressions separated by commas are never accidentally added.
    """
    source_values = _english_values(source_en)
    target_values = _russian_values(target_ru)
    if not source_values:
        return {"ok": True, "source_values": [], "target_values": target_values, "missing": []}

    src = Counter(source_values)
    dst = Counter(target_values)
    missing: list[int | float] = []
    for value, count in src.items():
        missing.extend([value] * max(0, count - dst.get(value, 0)))
    return {
        "ok": not missing,
        "source_values": source_values,
        "target_values": target_values,
        "missing": missing,
    }
