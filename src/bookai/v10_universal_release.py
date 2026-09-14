from __future__ import annotations

import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _norm
from .v10_general_release import (
    FinalBookBibleBuilder as _GeneralBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _GeneralQualityQA,
    ResidualHardDeepSeekRepair,
)
from .v10_source_bible import SourceOnlyBookBibleBuilder


_LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
_MIXED_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9-]*")
_SHORT_DEFINITION_RE = re.compile(
    r"\b([a-z][a-z'-]{2,8})\b\s+(?:had|has|have|is|was|were)\s+(?:no|not|only|a|an|the)\b",
    re.I,
)
_COMMON_SHORT = {
    "one", "thing", "man", "woman", "time", "year", "day", "way", "part", "side",
    "good", "great", "same", "other", "only", "just", "still", "even", "very",
}


def _is_acronym_or_identifier(token: str) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) >= 2 and all(ch.isupper() for ch in letters):
        return True
    if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
        return True
    # CamelCase/model/dataset identifier such as ImageNet or Word2Vec.
    if len(token) >= 4 and any(ch.isupper() for ch in token[1:]):
        return True
    return False


def _citation_tokens(source: str) -> set[str]:
    """Latin surnames immediately tied to a bibliographic year may remain Latin.

    This is deliberately contextual: a fictional proper name in narrative prose does
    not become exempt merely because it is capitalized.
    """
    out: set[str] = set()
    text = str(source or "")
    for match in re.finditer(r"\b([A-Z][A-Za-z'-]{2,})\b(?=[^.!?\n]{0,70}\b(?:18|19|20)\d{2}[a-z]?\b)", text):
        out.add(match.group(1))
    for match in re.finditer(r"\b([A-Z][A-Za-z'-]{2,})\s+et\s+al\.?", text):
        out.add(match.group(1))
    return out


def _source_has_exact_token(source: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", str(source or "")))


class FinalBookBibleBuilder(_GeneralBookBibleBuilder):
    """Universal source-only memory with slightly more recall for short smoke excerpts."""

    @staticmethod
    def _accept_name_item(
        item: dict[str, Any],
        allowed: dict[str, dict[str, Any]],
        *,
        threshold: float,
    ) -> tuple[str, str, str, str, str] | None:
        # On short excerpts the model has less evidence and systematically reports
        # lower confidence for genuine names. Keep the original spelling/Russian
        # sanity guards, but lower only the confidence floor; classification and
        # transliteration are still model-derived from source context.
        relaxed = 0.60 if threshold >= 0.70 else 0.56
        return SourceOnlyBookBibleBuilder._accept_name_item(
            item,
            allowed,
            threshold=min(float(threshold), relaxed),
        )

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        records = [dict(row) for row in _GeneralBookBibleBuilder._candidate_records(segments)]
        existing = {str(row.get("candidate") or "").casefold() for row in records}

        # A short rare noun may be technically important even when it occurs once.
        # Rather than hard-code vocabulary, harvest only source-defined nouns whose
        # immediate clause provides definitional evidence ("X had no edge, only...").
        for segment in segments:
            text = str(segment.text or "")
            for match in _SHORT_DEFINITION_RE.finditer(text):
                candidate = match.group(1).casefold()
                if candidate in existing or candidate in _COMMON_SHORT:
                    continue
                records.append({
                    "candidate": candidate,
                    "kind_hint": "technical_term",
                    "frequency": 1,
                    "contexts": [{"chapter": str(segment.chapter or ""), "text": _norm(text)[:520]}],
                })
                existing.add(candidate)
        return records


class FinalV10QualityQA(_GeneralQualityQA):
    """Publication QA that distinguishes protected technical notation from leaks."""

    @staticmethod
    def _protected_latin(source: str) -> set[str]:
        protected = set(_citation_tokens(source))
        for token in _LATIN_TOKEN_RE.findall(str(source or "")):
            if _is_acronym_or_identifier(token):
                protected.add(token)
        return protected

    @classmethod
    def _latin_issues(cls, segment: Segment, target: str) -> list[V10Issue]:
        source = str(segment.text or "")
        protected = cls._protected_latin(source)
        out: list[V10Issue] = []

        mixed = []
        for token in _MIXED_TOKEN_RE.findall(str(target or "")):
            if not re.search(r"[A-Za-z]", token) or not re.search(r"[А-Яа-яЁё]", token):
                continue
            # Standard one-letter engineering notation is not an untranslated leak.
            if re.fullmatch(r"[A-Za-z]-[А-Яа-яЁё-]+", token):
                continue
            mixed.append(token)
        if mixed:
            out.append(V10Issue(
                segment.id,
                "latin_leak",
                "local",
                "hard",
                "mixed Cyrillic/Latin token remains: " + ", ".join(dict.fromkeys(mixed[:6])),
            ))

        leaks = []
        for token in _LATIN_TOKEN_RE.findall(str(target or "")):
            if token.casefold() == "chapter":
                continue
            if token in protected and _source_has_exact_token(source, token):
                continue
            leaks.append(token)
        if leaks:
            out.append(V10Issue(
                segment.id,
                "latin_leak",
                "local",
                "hard",
                "untranslated Latin prose/name remains: " + ", ".join(dict.fromkeys(leaks[:8])),
            ))
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = [row for row in super().scan_segment(segment, target, memory) if row.code != "latin_leak"]
        issues.extend(self._latin_issues(segment, target))
        unique = {(row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
