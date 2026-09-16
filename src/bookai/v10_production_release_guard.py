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

# Direction relations are only HARD when the source contains explicit locomotion.
# Bare spatial phrases such as "look up the hill" or "looking down" are visual
# orientation, not movement, and must not be cross-matched across a long paragraph.
_SOURCE_UP_MOTION_RE = re.compile(
    r"\b(?:walk(?:ed|ing)?|went|go(?:ing)?|ran|run(?:ning)?|climb(?:ed|ing)?|came|come|"
    r"rode|ride|riding|moved?|moving|headed?|heading|trudged?|rushed?|started?|set\s+off|"
    r"made\s+(?:his|her|their)\s+way)\b[^.!?;:]{0,24}\bup\b",
    re.I,
)
_SOURCE_DOWN_MOTION_RE = re.compile(
    r"\b(?:walk(?:ed|ing)?|went|go(?:ing)?|ran|run(?:ning)?|climb(?:ed|ing)?|came|come|"
    r"rode|ride|riding|moved?|moving|headed?|heading|trudged?|rushed?|started?|set\s+off|"
    r"made\s+(?:his|her|their)\s+way)\b[^.!?;:]{0,24}\bdown\b",
    re.I,
)
_RU_UP_MOTION_RE = re.compile(
    r"\b(?:поднял(?:ся|ась|ись)?|поднимал(?:ся|ась|ись)?|поднима(?:лся|лась|лись|ется|ются)|"
    r"взош[её]л|взошла|пош[её]л\s+вверх|пошла\s+вверх|двинул(?:ся|ась)\s+вверх|"
    r"поехал\s+вверх|поехала\s+вверх|карабкал(?:ся|ась)|взбирал(?:ся|ась))\b",
    re.I,
)
_RU_DOWN_MOTION_RE = re.compile(
    r"\b(?:спустил(?:ся|ась|ись)?|спускал(?:ся|ась|ись)?|спуска(?:лся|лась|лись|ется|ются)|"
    r"сош[её]л|сошла|пош[её]л\s+вниз|пошла\s+вниз|двинул(?:ся|ась)\s+вниз|"
    r"поехал\s+вниз|поехала\s+вниз)\b",
    re.I,
)

_RU_REPEAT_STOP = {
    "и", "а", "но", "или", "он", "она", "они", "его", "ее", "её", "их", "ему", "ей", "им",
    "это", "этот", "эта", "эти", "что", "как", "когда", "если", "бы", "же", "не", "ни", "на",
    "в", "во", "с", "со", "к", "ко", "по", "за", "из", "от", "до", "у", "для", "о", "об", "про",
    "был", "была", "были", "есть", "было", "все", "всё", "только", "уже", "ему", "себя", "свой",
}


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


def strict_direction_mismatch(source: str, target: str) -> bool:
    """Confirm only explicit locomotion reversals, never gaze/orientation phrases."""
    source_text = str(source or "")
    target_text = str(target or "").casefold().replace("ё", "е")
    source_up = bool(_SOURCE_UP_MOTION_RE.search(source_text))
    source_down = bool(_SOURCE_DOWN_MOTION_RE.search(source_text))
    target_up = bool(_RU_UP_MOTION_RE.search(target_text))
    target_down = bool(_RU_DOWN_MOTION_RE.search(target_text))

    # A long paragraph can legitimately contain both directions. In that case this
    # coarse deterministic guard is not strong enough to issue a HARD verdict.
    if source_up and source_down:
        return False
    if source_up:
        return target_down and not target_up
    if source_down:
        return target_up and not target_down
    return False


def target_has_substantial_repeat(target: str) -> bool:
    """Confirm exact repeated Russian phrase evidence before keeping duplicate_content HARD.

    Repetition detectors on long literary paragraphs can overfire on repeated function
    words or recurring nouns. A hard duplicate verdict requires an exact repeated 4–6
    token span with enough lexical content and separation to plausibly be a duplicated
    clause rather than normal prose recurrence.
    """
    words = [w.casefold().replace("ё", "е") for w in re.findall(r"[А-Яа-яЁё]+", str(target or ""))]
    if len(words) < 12:
        return False
    for n in (6, 5, 4):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            content = [w for w in gram if w not in _RU_REPEAT_STOP and len(w) >= 4]
            if len(content) < 3 or len(" ".join(gram)) < 20:
                continue
            previous = seen.get(gram)
            if previous is not None and i - previous >= n + 2:
                return True
            seen.setdefault(gram, i)
    return False


def _entity_root_variants(root: str) -> set[str]:
    normalized = re.sub(r"[^а-яё]", "", str(root or "").casefold()).replace("ё", "е")
    if not normalized:
        return set()
    variants = {normalized}
    # Source-derived fictional names can legitimately oscillate at word-initial E
    # between Russian Е/Э transliteration. Treat that narrow pair as the same family
    # for chapter QA; the final merge canonicalizer below rewrites it to one spelling.
    if normalized.startswith("е") and len(normalized) >= 4:
        variants.add("э" + normalized[1:])
    elif normalized.startswith("э") and len(normalized) >= 4:
        variants.add("е" + normalized[1:])
    return variants


def entity_family_equivalent_present(segment: Segment, target: str, memory: BookMemory) -> bool:
    """Return True if every relevant family root is present, allowing only initial Е/Э variation."""
    source = str(segment.text or "")
    low = str(target or "").casefold().replace("ё", "е")
    relevant = 0
    for source_form, desc in memory.characters.items():
        if "kind=demonym_family" not in str(desc):
            continue
        if not re.search(rf"(?<![A-Za-z]){re.escape(str(source_form))}(?![A-Za-z])", source, re.I):
            continue
        match = re.search(r"(?:^|;)ru_root=([^;]+)", str(desc), re.I)
        variants = _entity_root_variants(match.group(1) if match else "")
        if not variants:
            continue
        relevant += 1
        if not any(re.search(rf"\b{re.escape(root)}[а-я]*\b", low) for root in variants):
            return False
    return relevant > 0


def canonicalize_entity_family_spelling(target: str, families: list[dict[str, Any]]) -> str:
    """Normalize accepted leading Е/Э family variants to the cached canonical root."""
    value = str(target or "")
    for row in families:
        root = re.sub(r"[^а-яё]", "", str(row.get("ru_root") or "").casefold()).replace("ё", "е")
        variants = _entity_root_variants(root) - {root}
        if not root or not variants:
            continue
        for variant in variants:
            pattern = re.compile(rf"\b{re.escape(variant)}(?P<tail>[А-Яа-яЁё]*)\b", re.I)

            def repl(match: re.Match[str], *, canonical: str = root) -> str:
                observed = match.group(0)
                tail = match.group("tail") or ""
                head = canonical
                if observed[:1].isupper():
                    head = canonical[:1].upper() + canonical[1:]
                return head + tail

            value = pattern.sub(repl, value)
    return value


def filter_release_false_positives(
    segment: Segment,
    target: str,
    memory: BookMemory,
    issues: list[V10Issue],
) -> list[V10Issue]:
    """Suppress only release-gate findings that fail a stricter deterministic confirmation."""
    filtered: list[V10Issue] = []
    for issue in issues:
        if issue.code == "direction_relation" and not strict_direction_mismatch(segment.text, target):
            continue
        if issue.code == "duplicate_content" and not target_has_substantial_repeat(target):
            continue
        if issue.code == "entity_family_canon" and entity_family_equivalent_present(segment, target, memory):
            continue
        filtered.append(issue)
    return filtered


__all__ = [
    "canonicalize_entity_family_spelling",
    "clean_numeric_result",
    "clean_quantity_result",
    "entity_family_equivalent_present",
    "filter_release_false_positives",
    "localized_gender_issues",
    "spurious_between_range_sums",
    "strict_direction_mismatch",
    "target_has_substantial_repeat",
]
