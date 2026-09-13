from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import BookMemory, Segment
from .v10_quantity import extract_quantity_obligations


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.U)
_RU_STOP = {
    "и", "а", "но", "или", "что", "как", "это", "он", "она", "они", "мы", "вы", "я", "ты",
    "его", "ее", "её", "их", "ему", "ей", "мне", "тебе", "вам", "нас", "вас", "в", "во", "на",
    "под", "над", "к", "ко", "от", "до", "за", "из", "у", "с", "со", "по", "для", "же", "бы",
}


@dataclass(frozen=True)
class ClauseFidelityIssue:
    code: str
    reason: str
    evidence: dict[str, Any]


def _norm_words(text: str) -> list[str]:
    return [w.casefold().replace("ё", "е") for w in _WORD_RE.findall(str(text or ""))]


def _repeated_ngram(text: str) -> tuple[str, int] | None:
    """Return a suspicious long repeated Russian phrase, if any.

    The contract is deliberately narrow: at least three words, at least 18 visible
    characters, at least one substantial content word, and non-overlapping repeats.
    This catches accidental clause duplication while avoiding tiny discourse phrases.
    """
    words = _norm_words(text)
    if len(words) < 9:
        return None
    for n in (6, 5, 4, 3):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(0, len(words) - n + 1):
            gram = tuple(words[i:i + n])
            phrase = " ".join(gram)
            if len(phrase) < 18:
                continue
            content = [w for w in gram if w not in _RU_STOP]
            if not content or max(map(len, content)) < 7:
                continue
            prev = seen.get(gram)
            if prev is not None and i - prev >= n:
                return phrase, 2
            seen.setdefault(gram, i)
    return None


def compare_duplicate_content_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    target_repeat = _repeated_ngram(target_ru)
    if not target_repeat:
        return {"ok": True, "repeated_phrase": None}

    # If the English source itself contains an obvious exact repeated multiword
    # phrase, do not call Russian repetition invented. This is conservative rather
    # than trying to align arbitrary paraphrases across languages.
    source_repeat = _repeated_ngram(source_en)
    if source_repeat:
        return {"ok": True, "repeated_phrase": target_repeat[0], "source_repetition": source_repeat[0]}
    return {
        "ok": False,
        "repeated_phrase": target_repeat[0],
        "reason": "target repeats a substantial clause/phrase while source has no comparable exact repetition",
    }


def _find_once(text: str, needle: str) -> int | None:
    hay = str(text or "").casefold().replace("ё", "е")
    ndl = str(needle or "").casefold().replace("ё", "е").strip()
    if not ndl:
        return None
    first = hay.find(ndl)
    if first < 0 or hay.find(ndl, first + max(1, len(ndl))) >= 0:
        return None
    return first


def _quantity_target_position(value: int, target_ru: str) -> int | None:
    low = str(target_ru or "").casefold().replace("ё", "е")
    forms = {
        1: (r"\b1\b", r"\bодин\w*\b", r"\bперв\w*\b"),
        2: (r"\b2\b", r"\bдва\b", r"\bдве\b", r"\bвтор\w*\b"),
        3: (r"\b3\b", r"\bтри\b", r"\bтрет\w*\b"),
        4: (r"\b4\b", r"\bчетыре\b", r"\bчетвер\w*\b"),
        5: (r"\b5\b", r"\bпять\b", r"\bпят\w*\b"),
        6: (r"\b6\b", r"\bшесть\b", r"\bшест\w*\b", r"\bпол(?:у)?дюжин\w*\b"),
        12: (r"\b12\b", r"\bдвенадцат\w*\b", r"\bдюжин\w*\b"),
        24: (r"\b24\b", r"\bдвадцат\w*\s+четыр\w*\b", r"\bдве\s+дюжин\w*\b"),
    }
    patterns = forms.get(value, (rf"\b{value}\b",))
    hits: list[int] = []
    for pattern in patterns:
        hits.extend(m.start() for m in re.finditer(pattern, low, re.I))
    hits = sorted(set(hits))
    return hits[0] if len(hits) == 1 else None


def _canon_from_memory(memory: BookMemory) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for source, ru in memory.glossary.items():
        src = str(source or "").strip()
        dst = str(ru or "").strip()
        if src and dst and len(src) >= 3 and len(dst) >= 2:
            rows.append((src, dst))
    for source, desc in memory.characters.items():
        match = re.search(r"(?:^|;)ru=([^;]+)", str(desc or ""), re.I)
        if match:
            rows.append((str(source or "").strip(), match.group(1).strip()))
    return rows


def compare_clause_order_fidelity(source_en: str, target_ru: str, memory: BookMemory) -> dict[str, Any]:
    """Check order only for uniquely alignable, source-grounded anchors.

    Anchors are exact quantities plus confirmed Book Bible names/glossary entries.
    We require unique occurrences on both sides; ambiguous/repeated anchors are ignored.
    This makes the contract sparse but high precision.
    """
    source = str(source_en or "")
    target = str(target_ru or "")
    anchors: list[tuple[str, int, int]] = []

    for obligation in extract_quantity_obligations(source):
        src_match = re.search(re.escape(obligation.source_phrase), source, re.I)
        if not src_match:
            continue
        target_pos = _quantity_target_position(obligation.value, target)
        if target_pos is not None:
            anchors.append((f"quantity:{obligation.value}:{obligation.kind}", src_match.start(), target_pos))

    source_low = source.casefold()
    for src, ru in _canon_from_memory(memory):
        src_low = src.casefold()
        if source_low.count(src_low) != 1:
            continue
        target_pos = _find_once(target, ru)
        if target_pos is None:
            continue
        anchors.append((f"canon:{src}", source_low.find(src_low), target_pos))

    # Collapse exact duplicate positions/keys and keep source order.
    unique: dict[str, tuple[str, int, int]] = {}
    for row in anchors:
        unique.setdefault(row[0], row)
    ordered = sorted(unique.values(), key=lambda row: row[1])
    if len(ordered) < 2:
        return {"ok": True, "anchors": ordered, "inversions": []}

    inversions: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        if left[2] > right[2]:
            inversions.append({"left": left[0], "right": right[0], "source_positions": [left[1], right[1]], "target_positions": [left[2], right[2]]})
    return {"ok": not inversions, "anchors": ordered, "inversions": inversions}


def scan_clause_fidelity(segment: Segment, target_ru: str, memory: BookMemory) -> list[ClauseFidelityIssue]:
    out: list[ClauseFidelityIssue] = []
    duplicate = compare_duplicate_content_fidelity(segment.text, target_ru)
    if not duplicate.get("ok", True):
        out.append(ClauseFidelityIssue("duplicate_content", str(duplicate.get("reason") or "invented target repetition"), duplicate))
    order = compare_clause_order_fidelity(segment.text, target_ru, memory)
    if not order.get("ok", True):
        out.append(ClauseFidelityIssue("clause_order", "reliable semantic anchors appear in a different order from source", order))
    return out
