from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import BookMemory, Segment
from .v10_quantity import extract_quantity_obligations


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.U)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;]+(?:\s+|$)")
_RU_STOP = {
    "и", "а", "но", "или", "что", "как", "это", "он", "она", "они", "мы", "вы", "я", "ты",
    "его", "ее", "её", "их", "ему", "ей", "мне", "тебе", "вам", "нас", "вас", "в", "во", "на",
    "под", "над", "к", "ко", "от", "до", "за", "из", "у", "с", "со", "по", "для", "же", "бы",
}
_EN_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "from", "of", "for",
    "with", "without", "as", "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "their", "they", "we", "you", "he", "she", "not", "can", "could",
}


@dataclass(frozen=True)
class ClauseFidelityIssue:
    code: str
    reason: str
    evidence: dict[str, Any]


def _norm_words(text: str) -> list[str]:
    return [w.casefold().replace("ё", "е") for w in _WORD_RE.findall(str(text or ""))]


def _repeated_ngram(text: str) -> tuple[str, int] | None:
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


def _source_has_local_substantial_repeat(source_en: str) -> dict[str, Any] | None:
    """Find a compact repeated EN content phrase inside one source sentence.

    Russian often expands an English two-word term into a three/four-word phrase.
    The target duplicate detector therefore cannot prove invention merely because its
    repeated n-gram is longer. We only use this exemption for a repeated English
    content bigram/trigram *inside the same sentence*; target-only repetition across
    discourse clauses remains a hard failure.
    """
    for sentence in re.split(r"[.!?;]+", str(source_en or "")):
        words = [w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)]
        if len(words) < 6:
            continue
        for n in (3, 2):
            seen: dict[tuple[str, ...], int] = {}
            for i in range(0, len(words) - n + 1):
                gram = tuple(words[i:i + n])
                content = [w for w in gram if w not in _EN_STOP and len(w) >= 4]
                if len(content) < n:
                    continue
                phrase = " ".join(gram)
                if len(phrase) < 10:
                    continue
                prev = seen.get(gram)
                if prev is not None and i - prev >= n:
                    return {"source_repetition": phrase, "ngram": n}
                seen.setdefault(gram, i)
    return None


def compare_duplicate_content_fidelity(source_en: str, target_ru: str) -> dict[str, Any]:
    target_repeat = _repeated_ngram(target_ru)
    if not target_repeat:
        return {"ok": True, "repeated_phrase": None}
    source_repeat = _repeated_ngram(source_en)
    if source_repeat:
        return {"ok": True, "repeated_phrase": target_repeat[0], "source_repetition": source_repeat[0]}
    local_source_repeat = _source_has_local_substantial_repeat(source_en)
    if local_source_repeat:
        return {
            "ok": True,
            "repeated_phrase": target_repeat[0],
            **local_source_repeat,
            "expanded_translation_repetition": True,
        }
    return {
        "ok": False,
        "repeated_phrase": target_repeat[0],
        "reason": "target repeats a substantial clause/phrase while source has no comparable exact repetition",
    }


def _glossary_supports_repetition(source_en: str, target_ru: str, memory: BookMemory) -> dict[str, str] | None:
    """Allow repeated Russian terminology when the same source term also repeats.

    The duplicate n-gram detector is intentionally language-agnostic, so a repeated
    multiword Russian translation can look suspicious even when English repeats a
    one-token/hyphenated technical term. Runtime BookMemory gives us a source-derived
    alignment without hard-coding any book vocabulary.
    """
    source_low = str(source_en or "").casefold()
    target_low = " ".join(str(target_ru or "").casefold().replace("ё", "е").split())
    for en, ru in memory.glossary.items():
        en_s = str(en or "").strip().casefold()
        ru_s = " ".join(str(ru or "").strip().casefold().replace("ё", "е").split())
        if not en_s or not ru_s or len(ru_s) < 8:
            continue
        source_count = len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(en_s)}(?![A-Za-z0-9])", source_low, re.I))
        target_count = target_low.count(ru_s)
        if source_count >= 2 and target_count >= 2:
            return {"source_term": en_s, "target_term": ru_s}
    return None


def _find_once(text: str, needle: str) -> int | None:
    hay = str(text or "").casefold().replace("ё", "е")
    ndl = str(needle or "").casefold().replace("ё", "е").strip()
    if not ndl:
        return None
    first = hay.find(ndl)
    if first < 0 or hay.find(ndl, first + max(1, len(ndl))) >= 0:
        return None
    return first


def _find_proper_once(text: str, canonical: str) -> int | None:
    exact = _find_once(text, canonical)
    if exact is not None:
        return exact
    low = str(text or "").casefold().replace("ё", "е")
    word = re.sub(r"[^а-я]", "", str(canonical or "").casefold().replace("ё", "е"))
    if len(word) < 4:
        return None
    stem = word
    if stem[-1:] in {"ь", "й", "а", "я"} and len(stem) >= 5:
        stem = stem[:-1]
    if len(stem) < 4:
        return None
    hits = [m.start() for m in re.finditer(rf"\b{re.escape(stem)}[а-я]*\b", low)]
    return hits[0] if len(hits) == 1 else None


def _clause_index(text: str, position: int) -> int:
    return sum(1 for match in _CLAUSE_BOUNDARY_RE.finditer(str(text or "")) if match.end() <= position)


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


def _canon_from_memory(memory: BookMemory) -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    character_names = {str(name or "").strip() for name in memory.characters}
    for source, ru in memory.glossary.items():
        src = str(source or "").strip()
        dst = str(ru or "").strip()
        if src and dst and len(src) >= 3 and len(dst) >= 2:
            rows.append((src, dst, bool(src[:1].isupper() or src in character_names)))
    for source, desc in memory.characters.items():
        match = re.search(r"(?:^|;)ru=([^;]+)", str(desc or ""), re.I)
        if match:
            rows.append((str(source or "").strip(), match.group(1).strip(), True))
    return rows


def compare_clause_order_fidelity(source_en: str, target_ru: str, memory: BookMemory) -> dict[str, Any]:
    """Compare order of uniquely alignable anchors across discourse clauses."""
    source = str(source_en or "")
    target = str(target_ru or "")
    anchors: list[tuple[str, int, int, int, int]] = []

    for obligation in extract_quantity_obligations(source):
        src_match = re.search(re.escape(obligation.source_phrase), source, re.I)
        if not src_match:
            continue
        target_pos = _quantity_target_position(obligation.value, target)
        if target_pos is not None:
            anchors.append((
                f"quantity:{obligation.value}:{obligation.kind}",
                src_match.start(), target_pos,
                _clause_index(source, src_match.start()), _clause_index(target, target_pos),
            ))

    source_low = source.casefold()
    for src, ru, proper in _canon_from_memory(memory):
        src_low = src.casefold()
        if source_low.count(src_low) != 1:
            continue
        target_pos = _find_proper_once(target, ru) if proper else _find_once(target, ru)
        if target_pos is None:
            continue
        source_pos = source_low.find(src_low)
        anchors.append((
            f"canon:{src}", source_pos, target_pos,
            _clause_index(source, source_pos), _clause_index(target, target_pos),
        ))

    unique: dict[str, tuple[str, int, int, int, int]] = {}
    for row in anchors:
        unique.setdefault(row[0], row)
    ordered = sorted(unique.values(), key=lambda row: row[1])
    if len(ordered) < 2:
        return {"ok": True, "anchors": ordered, "inversions": []}

    inversions: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        if left[3] == right[3]:
            continue
        if left[3] < right[3] and left[4] > right[4]:
            inversions.append({
                "left": left[0], "right": right[0],
                "source_clause_indexes": [left[3], right[3]],
                "target_clause_indexes": [left[4], right[4]],
            })
    return {"ok": not inversions, "anchors": ordered, "inversions": inversions}


def scan_clause_fidelity(segment: Segment, target_ru: str, memory: BookMemory) -> list[ClauseFidelityIssue]:
    out: list[ClauseFidelityIssue] = []
    duplicate = compare_duplicate_content_fidelity(segment.text, target_ru)
    if not duplicate.get("ok", True):
        support = _glossary_supports_repetition(segment.text, target_ru, memory)
        if not support:
            out.append(ClauseFidelityIssue(
                "duplicate_content",
                str(duplicate.get("reason") or "invented target repetition"),
                duplicate,
            ))
    order = compare_clause_order_fidelity(segment.text, target_ru, memory)
    if not order.get("ok", True):
        out.append(ClauseFidelityIssue(
            "clause_order",
            "reliable semantic anchors appear in a different discourse-clause order from source",
            order,
        ))
    return out
