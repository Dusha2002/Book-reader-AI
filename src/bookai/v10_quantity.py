from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .v10_numeric import compare_numeric_fidelity_v10


_EN_SMALL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_RU_THOUSAND_PREFIXES = {
    "двухтысяч": 2000, "трехтысяч": 3000, "трёхтысяч": 3000,
    "четырехтысяч": 4000, "четырёхтысяч": 4000, "пятитысяч": 5000,
    "шеститысяч": 6000, "семитысяч": 7000, "восьмитысяч": 8000,
    "девятитысяч": 9000, "десятитысяч": 10000, "одиннадцатитысяч": 11000,
    "двенадцатитысяч": 12000, "тринадцатитысяч": 13000,
    "четырнадцатитысяч": 14000, "пятнадцатитысяч": 15000,
    "шестнадцатитысяч": 16000, "семнадцатитысяч": 17000,
    "восемнадцатитысяч": 18000, "девятнадцатитысяч": 19000,
    "двадцатитысяч": 20000,
}
_RU_CARDINAL_LABELS = {
    1: ("один", "одна", "одно"), 2: ("два", "две"), 3: ("три",), 4: ("четыре",),
    5: ("пять",), 6: ("шесть",), 7: ("семь",), 8: ("восемь",), 9: ("девять",),
    10: ("десять",), 11: ("одиннадцать",), 12: ("двенадцать",),
}
_RU_HUNDRED_GENITIVE = {
    "двух": 200, "трех": 300, "трёх": 300, "четырех": 400, "четырёх": 400,
    "пяти": 500, "шести": 600, "семи": 700, "восьми": 800, "девяти": 900,
}


@dataclass(frozen=True)
class QuantityObligation:
    kind: str
    value: int
    source_phrase: str


def _small_value(token: str) -> int | None:
    value = str(token or "").casefold().strip()
    if value.isdigit():
        return int(value)
    return _EN_SMALL.get(value)


def extract_quantity_obligations(source_en: str) -> list[QuantityObligation]:
    text = str(source_en or "")
    out: list[QuantityObligation] = []
    occupied: list[tuple[int, int]] = []
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bhalf\s+(?:a\s+)?dozen\b", re.I), "half_dozen"),
        (re.compile(r"\b(?P<n>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+dozen\b", re.I), "n_dozen"),
        (re.compile(r"\b(?:a|one)\s+dozen\b", re.I), "dozen"),
        (re.compile(r"\bnumber\s+(?P<n>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b", re.I), "numbered_choice"),
    ]
    for pattern, kind in patterns:
        for match in pattern.finditer(text):
            if any(not (match.end() <= left or match.start() >= right) for left, right in occupied):
                continue
            phrase = match.group(0)
            if kind == "half_dozen":
                value = 6
            elif kind == "dozen":
                value = 12
            elif kind == "n_dozen":
                n = _small_value(match.group("n"))
                if n is None:
                    continue
                value = 12 * n
            else:
                n = _small_value(match.group("n"))
                if n is None:
                    continue
                value = n
            out.append(QuantityObligation(kind, value, phrase))
            occupied.append((match.start(), match.end()))
    return out


def _extra_target_values(target_ru: str) -> list[int]:
    text = str(target_ru or "").casefold().replace("ё", "е")
    out: list[int] = []
    out.extend([6] * len(re.findall(r"\bпол(?:у)?дюжин\w*\b", text)))
    ru_n = {
        "одна": 1, "одну": 1, "одной": 1, "две": 2, "двух": 2, "три": 3,
        "трех": 3, "четыре": 4, "четырех": 4, "пять": 5, "шесть": 6,
    }
    for match in re.finditer(r"\b(одна|одну|одной|две|двух|три|трех|четыре|четырех|пять|шесть)\s+дюжин\w*\b", text):
        out.append(12 * ru_n[match.group(1)])
    for match in re.finditer(r"\bдюжин\w*\b", text):
        prefix = text[max(0, match.start() - 5):match.start()]
        if "пол" not in prefix:
            out.append(12)
    for prefix, value in _RU_THOUSAND_PREFIXES.items():
        out.extend([value] * len(re.findall(rf"\b{re.escape(prefix)}[а-я]+\b", text)))
    for prefix, value in (("двухвек", 2), ("трехвек", 3), ("трёхвек", 3), ("четырехвек", 4), ("четырёхвек", 4)):
        out.extend([value] * len(re.findall(rf"\b{prefix}[а-я]+\b", text)))
    for word, value in _RU_HUNDRED_GENITIVE.items():
        out.extend([value] * len(re.findall(rf"\b{word}\s+сот(?:ен|ни|ням|нями|нях)?\b", text)))
    return out


def _number_label_present(value: int, target_ru: str) -> bool:
    low = str(target_ru or "").casefold().replace("ё", "е")
    if re.search(rf"\b(?:номер\s*)?{value}\b", low):
        return True
    for word in _RU_CARDINAL_LABELS.get(value, ()):
        if re.search(rf"\b(?:номер\s+)?{re.escape(word)}\b", low):
            return True
    stems = {
        1: "перв", 2: "втор", 3: "трет", 4: "четвер", 5: "пят", 6: "шест",
        7: "седьм", 8: "восьм", 9: "девят", 10: "десят", 11: "одиннадцат",
        12: "двенадцат",
    }
    stem = stems.get(value)
    return bool(stem and re.search(rf"\b{stem}[а-я]*\b", low))


def _numbered_choice_present(value: int, source_en: str, target_ru: str) -> bool:
    """Require both the label and its high-confidence governing proposition."""
    if not _number_label_present(value, target_ru):
        return False
    source = str(source_en or "").casefold()
    low = str(target_ru or "").casefold().replace("ё", "е")
    if re.search(r"\bmarry(?:ing|ied)?\s+number\s+", source):
        return bool(re.search(r"\b(?:женюсь|женишься|женится|жениться|женился|женат\w*|замуж\w*|брак\w*)\b", low))
    return True


def _approximate_values(source_en: str) -> set[int]:
    text = str(source_en or "").casefold()
    values: set[int] = set()
    if re.search(r"\b(?:a\s+)?(?:word|flight|step|day|minute|hour)\s+or\s+two\b", text):
        values.add(2)
    if re.search(r"\bone\s+or\s+two\b", text):
        values.update({1, 2})
    if re.search(r"\ba\s+week\s+or\s+ten\s+days\b", text):
        values.add(10)
    return values


def _symbolic_digit_counts(source_en: str) -> Counter[int]:
    """Count source digits that are acting as notation, not quantities.

    EPUB/PDF extraction can degrade Greek/math glyphs into ordinary digits. We only
    suppress quantity obligations for strongly symbolic shapes: a one-digit function
    token/argument such as `7(4)`, a digit adjacent to an operator, or the same kind of
    digit explicitly used as a variable (`where 4 denotes ...`). Real prose quantities
    like `9 models`, years, section numbers and decimal values remain untouched.
    """
    text = str(source_en or "")
    symbolic: Counter[int] = Counter()
    seen_spans: set[tuple[int, int]] = set()

    def mark(match: re.Match[str], group: int | str = 0) -> None:
        start, end = match.span(group)
        token = match.group(group)
        if (start, end) in seen_spans or not str(token).isdigit() or len(str(token)) != 1:
            return
        seen_spans.add((start, end))
        symbolic[int(token)] += 1

    for match in re.finditer(r"(?<![\d.])(?P<fn>\d)\s*\(\s*(?P<arg>\d)\s*\)(?![\d.])", text):
        mark(match, "fn")
        mark(match, "arg")

    for match in re.finditer(
        r"\b(?:where|variable|parameter|symbol|index)\s+(?P<n>\d)\s+"
        r"(?:contains?|denotes?|represents?|indexes?|is|are|stands?\s+for)\b",
        text,
        re.I,
    ):
        mark(match, "n")

    for match in re.finditer(r"(?<![\d.])(?P<n>\d)(?=\s*(?:=|\+|\*|/|\^|≤|≥|<|>))", text):
        mark(match, "n")
    for match in re.finditer(r"(?<=(?:=|\+|\*|/|\^|≤|≥|<|>))\s*(?P<n>\d)(?![\d.])", text):
        mark(match, "n")

    return symbolic


def compare_quantity_fidelity_v2(source_en: str, target_ru: str) -> dict[str, Any]:
    """Proposition-aware quantity fidelity layered on top of numeric_fidelity."""
    base = compare_numeric_fidelity_v10(source_en, target_ru)
    obligations = extract_quantity_obligations(source_en)
    target_values = list(base.get("target_values") or []) + _extra_target_values(target_ru)
    target_counts = Counter(target_values)

    dozen_multipliers: list[int] = []
    for obligation in obligations:
        if obligation.kind == "n_dozen":
            multiplier = obligation.value // 12
            if multiplier > 0:
                dozen_multipliers.append(multiplier)

    base_missing = list(base.get("missing") or [])
    source_counts = Counter(base.get("source_values") or [])
    for multiplier in dozen_multipliers:
        if source_counts.get(multiplier, 0) > 0:
            source_counts[multiplier] -= 1
            if source_counts[multiplier] <= 0:
                source_counts.pop(multiplier, None)
        if multiplier in base_missing:
            base_missing.remove(multiplier)

    for value in _approximate_values(source_en):
        source_counts.pop(value, None)
        base_missing = [row for row in base_missing if row != value]

    symbolic_counts = _symbolic_digit_counts(source_en)
    for value, count in symbolic_counts.items():
        if source_counts.get(value, 0) > 0:
            source_counts[value] -= min(count, source_counts[value])
            if source_counts[value] <= 0:
                source_counts.pop(value, None)

    # Preserve the legacy tolerance for repeated lexicalized values: once Russian
    # visibly carries a value, a second English mention need not be duplicated merely
    # to satisfy the numeric parser. Symbolic occurrences are first removed from the
    # source count, so a missing real quantity that only shares a digit with notation
    # still remains a failure.
    base_missing = [
        value
        for value in base_missing
        if source_counts.get(value, 0) > 0 and target_counts.get(value, 0) <= 0
    ]

    for obligation in obligations:
        if obligation.kind != "numbered_choice":
            source_counts[obligation.value] += 1

    missing_mentions: list[dict[str, Any]] = []
    for value, required in source_counts.items():
        present = target_counts.get(value, 0)
        if present < required:
            lexical_for_value = [o for o in obligations if o.value == value and o.kind != "numbered_choice"]
            if value in set(base_missing) or lexical_for_value:
                missing_mentions.append({"value": value, "required": required, "present": present})

    numbered_missing: list[dict[str, Any]] = []
    for obligation in obligations:
        if obligation.kind == "numbered_choice" and not _numbered_choice_present(obligation.value, source_en, target_ru):
            numbered_missing.append({"value": obligation.value, "source_phrase": obligation.source_phrase})

    return {
        "ok": not base_missing and not missing_mentions and not numbered_missing,
        "base": base,
        "base_missing": base_missing,
        "obligations": [o.__dict__ for o in obligations],
        "target_values": target_values,
        "symbolic_digit_counts": dict(symbolic_counts),
        "missing_mentions": missing_mentions,
        "numbered_choice_missing": numbered_missing,
    }
