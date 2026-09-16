from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import Segment


_DONT_SEE_WHY_NOT_RE = re.compile(
    r"\b(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+(?:really\s+)?see\s+why\s+not\b|"
    r"\b(?:there(?:'s| is)\s+)?no\s+reason\s+(?:why\s+)?not\b",
    re.I,
)
_BRIGANDINE_RE = re.compile(r"\bbrigandine\b", re.I)
_DARNING_NEEDLE_EYE_RE = re.compile(r"\beye\s+of\s+(?:a|the)\s+darning[-\s]?needle\b", re.I)
_LAST_LESSON_BUT_ONE_RE = re.compile(r"\blast\s+lesson\s+but\s+one\b", re.I)
_BRIGANDINE_RU_RE = re.compile(r"\b(?:бригандин|бригантин)[а-яё]*\b", re.I)
_NEEDLE_EYE_RU_RE = re.compile(r"\bушк[а-яё]*\b.{0,40}\bигл[а-яё]*\b", re.I | re.S)
_PENULTIMATE_RU_RE = re.compile(r"\bпредпослед[а-яё]*\b", re.I)
_WORD_RE = re.compile(r"[А-Яа-яЁё]{3,}")


def source_fingerprint(text: str) -> str:
    """Stable short fingerprint used to invalidate stale cached translations."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:20]


def provenance_entry(
    segment: Segment,
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
    *,
    stage: str = "draft",
) -> dict[str, Any]:
    return {
        "source_id": segment.id,
        "source_hash": source_fingerprint(segment.text),
        "context_ids": [
            s.id for s in [*(context_before or []), *(context_after or [])]
        ],
        "stage": stage,
    }


def _word_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = [word.casefold() for word in _WORD_RE.findall(text or "")]
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _neighbor_overlap(translations: dict[str, str], targets: list[Segment], index: int) -> float:
    current = _word_ngrams(str(translations.get(targets[index].id) or ""))
    if len(current) < 8:
        return 0.0
    best = 0.0
    for j in range(max(0, index - 2), min(len(targets), index + 3)):
        if j == index:
            continue
        other = _word_ngrams(str(translations.get(targets[j].id) or ""))
        if not other:
            continue
        best = max(best, len(current & other) / max(1, len(current)))
    return best


def specialist_guard_routes(
    targets: list[Segment],
    translated: dict[str, str],
) -> list[dict[str, Any]]:
    """High-confidence signals that MUST reach the existing semantic specialist.

    These are routing signals, not a second quality layer. They deliberately reuse
    the already configured specialist for verification/repair.
    """
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(targets):
        source = str(segment.text or "")
        current = str(translated.get(segment.id) or "")
        if not source or not current:
            continue

        signals: list[tuple[str, int, str]] = []
        if _DONT_SEE_WHY_NOT_RE.search(source):
            signals.append(
                (
                    "polarity_scope",
                    26,
                    "Double-negation/polarity scope is easy to invert; verify the complete proposition against SOURCE.",
                )
            )

        # Source-grounded lexical/idiomatic invariants. These are intentionally
        # narrow: they fire only when the English source contains an unambiguous
        # construction and the Russian target demonstrably lost that meaning.
        if _BRIGANDINE_RE.search(source) and not _BRIGANDINE_RU_RE.search(current):
            signals.append(
                (
                    "armor_terminology",
                    30,
                    "SOURCE says 'brigandine' (brigandine armour). Preserve that armour type explicitly as "
                    "бригантина/бригандин; do NOT generalize it to кольчуга or generic armour.",
                )
            )
        if _DARNING_NEEDLE_EYE_RE.search(source) and not _NEEDLE_EYE_RU_RE.search(current):
            signals.append(
                (
                    "needle_eye_idiom",
                    30,
                    "SOURCE 'eye of a darning-needle' means the needle's eye: render naturally as "
                    "'ушко штопальной иглы', not 'глазок/глаз иглы'.",
                )
            )
        if _LAST_LESSON_BUT_ONE_RE.search(source) and not _PENULTIMATE_RU_RE.search(current):
            signals.append(
                (
                    "penultimate_idiom",
                    30,
                    "SOURCE 'last lesson but one' means the penultimate lesson. Russian must preserve that ordinal sense "
                    "with 'предпоследний/предпоследним', not a literal 'последний ... один'.",
                )
            )

        ratio = len(current) / max(1, len(source))
        suspicious_expansion = (
            (len(source) >= 160 and ratio >= 1.27)
            or (80 <= len(source) < 160 and ratio >= 1.45)
        )
        overlap = _neighbor_overlap(translated, targets, index)
        if suspicious_expansion or overlap >= 0.30:
            reason_bits = []
            if suspicious_expansion:
                reason_bits.append(f"target/source length ratio={ratio:.3f}")
            if overlap >= 0.30:
                reason_bits.append(f"neighbor Russian 5-gram overlap={overlap:.3f}")
            signals.append(
                (
                    "source_contamination",
                    26,
                    "Possible cross-segment carry-over: current_ru must contain ONLY propositions entailed by this SOURCE; "
                    "neighboring text is context for disambiguation, never content to import. " + "; ".join(reason_bits),
                )
            )

        if not signals:
            continue
        code, priority, _ = max(signals, key=lambda item: item[1])
        rows.append(
            {
                "id": segment.id,
                "index": index,
                "priority": priority,
                "code": code,
                "reason": " | ".join(item[2] for item in signals),
                "guard_codes": [item[0] for item in signals],
                "glossary_hits": [],
            }
        )
    return rows
