from __future__ import annotations

import re
from collections import Counter
from typing import Any


_EN_TOKEN = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")
_RU_TOKEN = re.compile(r"[А-Яа-яЁё]+|\d+(?:[.,]\d+)?")
_DIGIT = re.compile(r"^\d+(?:[.,]\d+)?$")

_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
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
_EN_SCALES = {"hundred": 100, "thousand": 1000}
_EN_NUMBER_WORDS = set(_EN_UNITS) | set(_EN_TENS) | set(_EN_ORDINAL) | set(_EN_SCALES) | {"and"}
_EN_MARKERS = {"number", "no", "chapter", "street", "road", "avenue", "house", "door", "address", "room"}

_RU_EXACT = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одно": 1, "одну": 1, "одного": 1, "одному": 1, "одним": 1, "одном": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2,
    "три": 3, "трех": 3, "трёх": 3, "трем": 3, "трём": 3, "тремя": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4, "четырем": 4, "четырём": 4, "четырьмя": 4,
    "пять": 5, "пяти": 5, "пятью": 5,
    "шесть": 6, "шести": 6, "шестью": 6,
    "семь": 7, "семи": 7, "семью": 7,
    "восемь": 8, "восьми": 8, "восемью": 8,
    "девять": 9, "девяти": 9, "девятью": 9,
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
    "двадцать": 20, "двадцати": 20,
    "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60,
    "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80,
    "девяносто": 90, "девяноста": 90,
    "сто": 100, "ста": 100,
    "двести": 200, "двухсот": 200,
    "триста": 300, "трехсот": 300, "трёхсот": 300,
    "четыреста": 400, "четырехсот": 400, "четырёхсот": 400,
    "пятьсот": 500, "пятисот": 500,
    "шестьсот": 600, "шестисот": 600,
    "семьсот": 700, "семисот": 700,
    "восемьсот": 800, "восьмисот": 800,
    "девятьсот": 900, "девятисот": 900,
    "тысяча": 1000, "тысячи": 1000, "тысячу": 1000, "тысяч": 1000,
}

# Ordered longest/most specific stems first. These cover inflected Russian
# ordinals such as "тридцатом", "двадцать восьмой", "сто первым".
_RU_ORDINAL_STEMS = [
    ("девяност", 90), ("восьмидесят", 80), ("семидесят", 70),
    ("шестидесят", 60), ("пятидесят", 50), ("сороков", 40),
    ("тридцат", 30), ("двадцат", 20),
    ("девятнадцат", 19), ("восемнадцат", 18), ("семнадцат", 17),
    ("шестнадцат", 16), ("пятнадцат", 15), ("четырнадцат", 14),
    ("тринадцат", 13), ("двенадцат", 12), ("одиннадцат", 11),
    ("десят", 10), ("девят", 9), ("восьм", 8), ("седьм", 7),
    ("шест", 6), ("пят", 5), ("четверт", 4), ("треть", 3),
    ("трет", 3), ("втор", 2), ("перв", 1),
    ("тысячн", 1000), ("сот", 100),
]

_RU_MARKERS = {
    "номер", "номера", "номере", "главе", "глава", "улица", "улице", "улицы",
    "дом", "дома", "доме", "квартира", "квартире", "комната", "комнате", "адрес", "адресу",
}


def _digit_value(token: str) -> int | float | None:
    if not _DIGIT.fullmatch(token):
        return None
    raw = token.replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _parse_en_run(tokens: list[str]) -> int | None:
    total = 0
    current = 0
    seen = False
    for token in tokens:
        if token == "and":
            continue
        if token in _EN_UNITS:
            current += _EN_UNITS[token]
            seen = True
        elif token in _EN_TENS:
            current += _EN_TENS[token]
            seen = True
        elif token in _EN_ORDINAL:
            current += _EN_ORDINAL[token]
            seen = True
        elif token == "hundred":
            current = max(1, current) * 100
            seen = True
        elif token == "thousand":
            total += max(1, current) * 1000
            current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def _ru_token_value(token: str) -> tuple[int, bool] | None:
    token = token.casefold().replace("ё", "е")
    exact = {k.replace("ё", "е"): v for k, v in _RU_EXACT.items()}
    if token in exact:
        return exact[token], False
    for stem, value in _RU_ORDINAL_STEMS:
        stem_n = stem.replace("ё", "е")
        if token.startswith(stem_n) and len(token) >= len(stem_n) + 1:
            return value, True
    return None


def _parse_ru_run(tokens: list[str]) -> int | None:
    values: list[tuple[int, bool]] = []
    for token in tokens:
        parsed = _ru_token_value(token)
        if parsed is None:
            return None
        values.append(parsed)
    if not values:
        return None

    # Russian hundreds are encoded directly ("двести"=200); thousands behave as
    # a scale. Everything below 100 is additive for ordinary cardinal/ordinal
    # compounds, so "тридцать первом" resolves to 31 and cannot masquerade as 30.
    total = 0
    current = 0
    for value, _ordinal in values:
        if value == 1000:
            total += max(1, current) * 1000
            current = 0
        elif value >= 100:
            current += value
        else:
            current += value
    return total + current


def _english_values(text: str) -> list[int | float]:
    tokens = [m.group(0).casefold() for m in _EN_TOKEN.finditer(text or "")]
    out: list[int | float] = []
    i = 0
    while i < len(tokens):
        digit = _digit_value(tokens[i])
        if digit is not None:
            out.append(digit)
            i += 1
            continue
        if tokens[i] not in _EN_NUMBER_WORDS:
            i += 1
            continue
        start = i
        run: list[str] = []
        while i < len(tokens) and tokens[i] in _EN_NUMBER_WORDS:
            run.append(tokens[i])
            i += 1
        value = _parse_en_run(run)
        if value is None:
            continue
        previous = tokens[start - 1] if start else ""
        following = tokens[i] if i < len(tokens) else ""
        strong = (
            value >= 10
            or len([x for x in run if x != "and"]) >= 2
            or any(x in _EN_ORDINAL for x in run)
            or previous in _EN_MARKERS
            or following in _EN_MARKERS
        )
        if strong:
            out.append(value)
    return out


def _russian_values(text: str) -> list[int | float]:
    tokens = [m.group(0).casefold() for m in _RU_TOKEN.finditer(text or "")]
    out: list[int | float] = []
    i = 0
    while i < len(tokens):
        digit = _digit_value(tokens[i])
        if digit is not None:
            out.append(digit)
            i += 1
            continue
        if _ru_token_value(tokens[i]) is None:
            i += 1
            continue
        start = i
        run: list[str] = []
        while i < len(tokens) and _ru_token_value(tokens[i]) is not None:
            run.append(tokens[i])
            i += 1
        value = _parse_ru_run(run)
        if value is None:
            continue
        previous = tokens[start - 1] if start else ""
        following = tokens[i] if i < len(tokens) else ""
        strong = value >= 10 or len(run) >= 2 or previous in _RU_MARKERS or following in _RU_MARKERS
        if strong:
            out.append(value)
    return out


def compare_numeric_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    """Compare explicit/certain EN numeric facts with their RU rendering.

    The function is intentionally conservative about ambiguous single words such
    as English "one": it becomes a numeric obligation only in a strong numeric
    context. Values >=10, compounds, ordinals, digit tokens and numbered entities
    are treated as hard facts.
    """
    source_values = _english_values(source_en)
    if not source_values:
        return {
            "ok": True,
            "source_values": [],
            "target_values": _russian_values(target_ru),
            "missing": [],
        }
    target_values = _russian_values(target_ru)
    src = Counter(source_values)
    dst = Counter(target_values)
    missing: list[int | float] = []
    for value, count in src.items():
        deficit = max(0, count - dst.get(value, 0))
        missing.extend([value] * deficit)
    return {
        "ok": not missing,
        "source_values": source_values,
        "target_values": target_values,
        "missing": missing,
    }
