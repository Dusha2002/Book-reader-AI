from __future__ import annotations

import re
from typing import Any

from .numeric_fidelity import compare_numeric_fidelity


# Forms deliberately missing from the legacy parser but common in literary Russian.
# This layer only *adds evidence that a source value is present*; it never invents
# a new source obligation, so it can safely suppress false positives without
# weakening substitutions such as six -> five.
_RU_HUNDRED_COMPOUNDS = (
    (re.compile(r"\bдвухсот[а-яё]+\b", re.I), 200),
    (re.compile(r"\bтр[её]хсот[а-яё]+\b", re.I), 300),
    (re.compile(r"\bчетыр[её]хсот[а-яё]+\b", re.I), 400),
    (re.compile(r"\bпятисот[а-яё]+\b", re.I), 500),
    (re.compile(r"\bшестисот[а-яё]+\b", re.I), 600),
    (re.compile(r"\bсемисот[а-яё]+\b", re.I), 700),
    (re.compile(r"\bвосьмисот[а-яё]+\b", re.I), 800),
    (re.compile(r"\bдевятисот[а-яё]+\b", re.I), 900),
)
_RU_HUNDRED_PREPOSITIONAL = {
    "двухстах": 200,
    "трехстах": 300,
    "трёхстах": 300,
    "четырехстах": 400,
    "четырёхстах": 400,
    "пятистах": 500,
    "шестистах": 600,
    "семистах": 700,
    "восьмистах": 800,
    "девятистах": 900,
}
_RU_ONE_OBLIQUE_RE = re.compile(r"\b(?:одной|одною)\b", re.I)


def _extra_ru_values(target_ru: str) -> set[int]:
    text = str(target_ru or "").casefold()
    values: set[int] = set()
    for pattern, value in _RU_HUNDRED_COMPOUNDS:
        if pattern.search(text):
            values.add(value)
    for token, value in _RU_HUNDRED_PREPOSITIONAL.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            values.add(value)
    # The legacy parser covers один/одна/одно/одну/одного/... but historically
    # missed the very common feminine genitive/dative/instrumental form «одной».
    # Treat it as evidence for source value 1 rather than forcing an editor to add
    # a bogus digit or parenthetical marker merely to satisfy the numeric gate.
    if _RU_ONE_OBLIQUE_RE.search(text):
        values.add(1)
    return values


def compare_numeric_fidelity_v10(source_en: str, target_ru: str) -> dict[str, Any]:
    base = dict(compare_numeric_fidelity(source_en, target_ru))
    missing = list(base.get("missing") or [])
    if not missing:
        return base

    extra = _extra_ru_values(target_ru)
    low_source = str(source_en or "").casefold()
    low_target = str(target_ru or "").casefold().replace("ё", "е")

    residual: list[int | float] = []
    for value in missing:
        if isinstance(value, int) and value in extra:
            continue
        # Natural Russian lexicalization of the English alternative "one or two".
        if value == 2 and re.search(r"\bone\s+or\s+two\b", low_source) and re.search(r"\bпар(?:а|у|ой|е)\b", low_target):
            continue
        # Legacy parser knows тысяча/тысячи/тысячу/тысяч but not oblique forms.
        # If the source value is an exact multiple of 1000 and Russian visibly
        # contains the same leading cardinal plus тысячам/тысячами/тысячах, treat
        # the scale as preserved. This is intentionally narrow.
        if isinstance(value, int) and value >= 1000 and value % 1000 == 0:
            if re.search(r"\bтысяч(?:ам|ами|ах)\b", low_target):
                lead = value // 1000
                # Base parser often still sees the leading cardinal (e.g. сорока=40).
                target_values = set(base.get("target_values") or [])
                if lead in target_values:
                    continue
        residual.append(value)

    base["missing"] = residual
    base["ok"] = not residual
    base["v10_extra_ru_values"] = sorted(extra)
    return base
