from __future__ import annotations

import re
from typing import Any


# These contracts are intentionally narrow. They are not a semantic evaluator;
# they only prove a few publication facts cheaply enough to run over every row.

_RU_NEGATIVE_SPACE = re.compile(r"\s+")

_MATERIAL_OBJECT = (
    r"(?:plate|plates|armou?r|blade|blades|wire|bar|bars|rod|rods|sheet|sheets|"
    r"screw|screws|gear|gears|tool|tools|nail|nails|ring|rings|chain|chains|"
    r"door|doors|frame|frames|wheel|wheels|bolt|bolts|spring|springs|pipe|pipes|"
    r"tube|tubes|helmet|helmets|shield|shields|knife|knives|sword|swords|pin|pins|"
    r"axle|axles|shaft|shafts|beam|beams|mesh|link|links)"
)

# Require a physical-object context so verbs/metaphors such as "steel yourself"
# are not treated as material obligations.
_MATERIAL_RULES: dict[str, tuple[re.Pattern[str], tuple[str, ...]]] = {
    "steel": (
        re.compile(rf"\bsteel[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+steel\b", re.I),
        ("сталь", "стальн"),
    ),
    "iron": (
        re.compile(rf"\biron[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+iron\b", re.I),
        ("желез", "чугун"),
    ),
    "copper": (
        re.compile(rf"\bcopper[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+copper\b", re.I),
        ("мед", "медн"),
    ),
    "brass": (
        re.compile(rf"\bbrass[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+brass\b", re.I),
        ("латун",),
    ),
    "bronze": (
        re.compile(rf"\bbronze[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+bronze\b", re.I),
        ("бронз",),
    ),
    "aluminium": (
        re.compile(rf"\balumin(?:ium|um)[- ]+{_MATERIAL_OBJECT}\b|\b{_MATERIAL_OBJECT}\s+of\s+alumin(?:ium|um)\b", re.I),
        ("алюмин",),
    ),
}

_LEAD_SCREW_SOURCE = re.compile(r"\blead[- ]screws?\b", re.I)
_LEAD_SCREW_RU = re.compile(r"\bходов\w*\s+винт\w*\b", re.I)


def compare_question_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    """Require every explicit source question mark to survive.

    Literary Russian can restructure a question internally, but silently collapsing
    one of several explicit interrogatives in the same paragraph is a strong,
    objective omission signal. Extra target questions are not blocked here.
    """
    source_count = str(source_en or "").count("?")
    target_count = str(target_ru or "").count("?")
    return {
        "ok": target_count >= source_count,
        "source_questions": source_count,
        "target_questions": target_count,
        "missing_questions": max(0, source_count - target_count),
    }


def _contains_ru_stem(text: str, stems: tuple[str, ...]) -> bool:
    normalized = _RU_NEGATIVE_SPACE.sub(" ", str(text or "").casefold().replace("ё", "е"))
    words = re.findall(r"[а-я]+", normalized)
    return any(any(word.startswith(stem) for stem in stems) for word in words)


def compare_material_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    """Check only unambiguous physical material+noun constructions.

    This deliberately does *not* treat English `lead` as the metal by default.
    In machine-tool prose, `lead-screw` is correctly translated as `ходовой винт`;
    forcing a literal `свинцовый` there would actively damage the translation.
    """
    source = str(source_en or "")
    target = str(target_ru or "")
    required: list[str] = []
    missing: list[str] = []

    for name, (pattern, stems) in _MATERIAL_RULES.items():
        if not pattern.search(source):
            continue
        required.append(name)
        if not _contains_ru_stem(target, stems):
            missing.append(name)

    return {
        "ok": not missing,
        "required": required,
        "missing": missing,
    }


def lexicalized_technical_compound_preserved(source_en: str, target_ru: str) -> bool:
    """Recognize source compounds whose correct RU term does not mirror tokens."""
    source = str(source_en or "")
    target = str(target_ru or "")
    return bool(_LEAD_SCREW_SOURCE.search(source) and _LEAD_SCREW_RU.search(target))
