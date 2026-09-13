from __future__ import annotations

import re

# Narrow deterministic cleanup for untranslated English numeric address labels.
# This is intentionally not a general translator. It only fires when the English
# source proves that a number-word phrase is an ordinal street/avenue/road/lane
# name (or an Island + cardinal label) and the same English words leaked into RU.

_EN_UNITS_ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}
_EN_TEENS_ORD = {
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
}
_EN_TENS_CARD = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_TENS_ORD = {
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_EN_CARDINAL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, **_EN_TENS_CARD,
}

_RU_CARDINAL_1_19 = {
    1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
    6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
    11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
    15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
    19: "девятнадцать",
}
_RU_TENS_CARD = {
    20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
    60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
}
_RU_ORD_FEM_NOM = {
    1: "первая", 2: "вторая", 3: "третья", 4: "четвертая", 5: "пятая",
    6: "шестая", 7: "седьмая", 8: "восьмая", 9: "девятая", 10: "десятая",
    11: "одиннадцатая", 12: "двенадцатая", 13: "тринадцатая", 14: "четырнадцатая",
    15: "пятнадцатая", 16: "шестнадцатая", 17: "семнадцатая", 18: "восемнадцатая",
    19: "девятнадцатая", 20: "двадцатая", 30: "тридцатая", 40: "сороковая",
    50: "пятидесятая", 60: "шестидесятая", 70: "семидесятая",
    80: "восьмидесятая", 90: "девяностая",
}
_RU_ORD_FEM_INST = {
    1: "первой", 2: "второй", 3: "третьей", 4: "четвертой", 5: "пятой",
    6: "шестой", 7: "седьмой", 8: "восьмой", 9: "девятой", 10: "десятой",
    11: "одиннадцатой", 12: "двенадцатой", 13: "тринадцатой", 14: "четырнадцатой",
    15: "пятнадцатой", 16: "шестнадцатой", 17: "семнадцатой", 18: "восемнадцатой",
    19: "девятнадцатой", 20: "двадцатой", 30: "тридцатой", 40: "сороковой",
    50: "пятидесятой", 60: "шестидесятой", 70: "семидесятой",
    80: "восьмидесятой", 90: "девяностой",
}

_ORD_WORD = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    r"thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
    r"twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth))"
)
_CARD_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
    r"(?:one|two|three|four|five|six|seven|eight|nine))"
)
_SOURCE_STREET_RE = re.compile(rf"\b(?P<num>{_ORD_WORD})\s+(?P<kind>Street|Avenue|Road|Lane)\b", re.I)
_SOURCE_ISLAND_RE = re.compile(rf"\bIsland\s+(?P<num>{_CARD_WORD})\b", re.I)


def _parse_ordinal(phrase: str) -> int | None:
    parts = re.split(r"[- ]+", str(phrase or "").casefold().strip())
    if len(parts) == 1:
        word = parts[0]
        return _EN_UNITS_ORD.get(word) or _EN_TEENS_ORD.get(word) or _EN_TENS_ORD.get(word)
    if len(parts) == 2 and parts[0] in _EN_TENS_CARD and parts[1] in _EN_UNITS_ORD:
        return _EN_TENS_CARD[parts[0]] + _EN_UNITS_ORD[parts[1]]
    return None


def _parse_cardinal(phrase: str) -> int | None:
    parts = re.split(r"[- ]+", str(phrase or "").casefold().strip())
    if len(parts) == 1:
        return _EN_CARDINAL.get(parts[0])
    if len(parts) == 2 and parts[0] in _EN_TENS_CARD and parts[1] in _EN_CARDINAL and _EN_CARDINAL[parts[1]] < 10:
        return _EN_TENS_CARD[parts[0]] + _EN_CARDINAL[parts[1]]
    return None


def _ru_cardinal(value: int) -> str:
    if value in _RU_CARDINAL_1_19:
        return _RU_CARDINAL_1_19[value]
    if value in _RU_TENS_CARD:
        return _RU_TENS_CARD[value]
    tens = value // 10 * 10
    unit = value % 10
    if tens in _RU_TENS_CARD and unit in _RU_CARDINAL_1_19:
        return f"{_RU_TENS_CARD[tens]} {_RU_CARDINAL_1_19[unit]}"
    return str(value)


def _ru_ordinal_fem(value: int, *, instrumental: bool = False) -> str:
    table = _RU_ORD_FEM_INST if instrumental else _RU_ORD_FEM_NOM
    if value in table:
        return table[value]
    tens = value // 10 * 10
    unit = value % 10
    if tens in _RU_TENS_CARD and unit in table:
        return f"{_RU_TENS_CARD[tens]} {table[unit]}"
    return str(value)


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def normalize_numbered_address_literals(source_en: str, target_ru: str) -> tuple[str, int]:
    """Replace only source-proven English numeric address labels leaked into RU.

    Example: `Sixty-Seventh Street` -> `Шестьдесят седьмая улица`.
    The function does nothing unless the exact number-word label is present in the
    English source as an address/street entity, which keeps the transform narrow.
    """
    source = str(source_en or "")
    out = str(target_ru or "")
    fixes = 0

    streets: list[tuple[str, int, str]] = []
    for match in _SOURCE_STREET_RE.finditer(source):
        phrase = match.group("num")
        value = _parse_ordinal(phrase)
        if value is not None:
            streets.append((phrase, value, match.group("kind")))

    # Handle the common coordinated address form first so Russian case is natural:
    # "between Sixty-Sixth and Sixty-Eighth Street" ->
    # "между Шестьдесят шестой и Шестьдесят восьмой улицами".
    if len(streets) >= 2:
        source_values = {phrase.casefold(): value for phrase, value, _ in streets}
        pair_re = re.compile(
            rf"\bмежду\s+(?P<a>{_ORD_WORD})(?:\s+Street)?\s+и\s+(?P<b>{_ORD_WORD})(?:\s+Street)?\b",
            re.I,
        )

        def repl_pair(match: re.Match[str]) -> str:
            nonlocal fixes
            a_raw, b_raw = match.group("a"), match.group("b")
            a = source_values.get(a_raw.casefold())
            b = source_values.get(b_raw.casefold())
            if a is None or b is None:
                return match.group(0)
            fixes += 1
            return f"между {_cap_first(_ru_ordinal_fem(a, instrumental=True))} и {_cap_first(_ru_ordinal_fem(b, instrumental=True))} улицами"

        out = pair_re.sub(repl_pair, out)

    for phrase, value, kind in streets:
        phrase_re = re.escape(phrase).replace(r"\-", r"[- ]")
        # Full English address label.
        full_re = re.compile(rf"\b{phrase_re}\s+{re.escape(kind)}\b", re.I)
        replacement = _cap_first(_ru_ordinal_fem(value)) + " улица"
        out, n = full_re.subn(replacement, out)
        fixes += n

        # Common mixed leak: "улица Sixty-Seventh".
        mixed_re = re.compile(rf"\bулиц\w*\s+{phrase_re}\b", re.I)
        out, n = mixed_re.subn(replacement, out)
        fixes += n

        # If the street noun was elided in the RU draft but the exact English
        # ordinal still leaked, remove the Latin residue while preserving value.
        bare_re = re.compile(rf"\b{phrase_re}\b", re.I)
        out, n = bare_re.subn(_cap_first(_ru_ordinal_fem(value)), out)
        fixes += n

    for match in _SOURCE_ISLAND_RE.finditer(source):
        phrase = match.group("num")
        value = _parse_cardinal(phrase)
        if value is None:
            continue
        phrase_re = re.escape(phrase).replace(r"\-", r"[- ]")
        full_re = re.compile(rf"\bIsland\s+{phrase_re}\b", re.I)
        out, n = full_re.subn("Остров " + _cap_first(_ru_cardinal(value)), out)
        fixes += n
        mixed_re = re.compile(rf"\bостров\s+{phrase_re}\b", re.I)
        out, n = mixed_re.subn("Остров " + _cap_first(_ru_cardinal(value)), out)
        fixes += n

    return out, fixes
