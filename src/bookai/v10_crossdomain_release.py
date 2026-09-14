from __future__ import annotations

import json
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json, _norm
from .v10_source_bible import _has_clean_russian
from .v10_universal_release import (
    FinalBookBibleBuilder as _UniversalBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist as _UniversalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _UniversalQualityQA,
    ResidualHardDeepSeekRepair,
)

_CROSSDOMAIN_CACHE_MARKER = "v10-crossdomain-1"
_TECH_STYLE_WORDS = (
    "technical", "academic", "scientific", "expository", "textbook", "engineering", "research",
)
_RU_ENDINGS = (
    "ыми", "ими", "ого", "ему", "ому", "ую", "юю", "ами", "ями", "ая", "яя", "ой", "ей",
    "ый", "ий", "ое", "ее", "ые", "ие", "ых", "их", "ам", "ям", "ах", "ях", "ом", "ем",
    "ов", "ев", "ью", "ы", "и", "а", "я", "у", "ю", "е", "о", "ь",
)


def _technical_style(memory: BookMemory) -> bool:
    style = memory.style
    blob = " ".join(
        str(value or "")
        for value in (style.narrative_voice, style.rhythm, style.dialogue, style.humor)
    ).casefold()
    return any(word in blob for word in _TECH_STYLE_WORDS)


def _ru_stem(word: str) -> str:
    value = str(word or "").casefold().replace("ё", "е")
    if len(value) <= 4:
        return value
    for ending in sorted(_RU_ENDINGS, key=len, reverse=True):
        if value.endswith(ending) and len(value) - len(ending) >= 4:
            return value[: -len(ending)]
    return value


def _ru_phrase_present(expected: str, target: str) -> bool:
    expected_words = [
        _ru_stem(word)
        for word in re.findall(r"[А-Яа-яЁё]+", str(expected or ""))
        if len(word) >= 3
    ]
    if not expected_words:
        return True
    target_stems = {
        _ru_stem(word)
        for word in re.findall(r"[А-Яа-яЁё]+", str(target or ""))
        if len(word) >= 3
    }
    return all(stem in target_stems for stem in expected_words if len(stem) >= 3)


def _source_phrase_present(source: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z]){re.escape(str(phrase or '').strip())}(?![A-Za-z])", source, re.I))


def _citation_surnames(source: str) -> set[str]:
    text = str(source or "")
    out: set[str] = set()
    patterns = (
        r"\b([A-Z][A-Za-z'-]{2,})\s*(?:et\s+al\.)?\s*\((?:18|19|20)\d{2}[a-z]?\)",
        r"\b([A-Z][A-Za-z'-]{2,})(?:\s+et\s+al\.)?\s*,\s*(?:18|19|20)\d{2}[a-z]?\b",
    )
    for pattern in patterns:
        out.update(match.group(1) for match in re.finditer(pattern, text))
    for match in re.finditer(
        r"\b([A-Z][A-Za-z'-]{2,})\s+and\s+([A-Z][A-Za-z'-]{2,})\s*\((?:18|19|20)\d{2}[a-z]?\)",
        text,
    ):
        out.update((match.group(1), match.group(2)))
    return out


def _source_acronyms(source: str) -> set[str]:
    out: set[str] = set()
    for match in re.finditer(r"\b([A-Z]{3,})(?:s)?\b", str(source or "")):
        out.add(match.group(1))
    return out


class FinalBookBibleBuilder(_UniversalBookBibleBuilder):
    """Adds one source-only domain terminology pass for academic/technical books."""

    def _augment_domain_glossary(self, segments: list[Segment], memory: BookMemory) -> int:
        if not _technical_style(memory):
            return 0

        joined = "\n".join(str(segment.text or "") for segment in segments)
        if not joined.strip():
            return 0

        sample_count = min(20, len(segments))
        excerpts: list[str] = []
        for n in range(sample_count):
            index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
            text = _norm(segments[index].text)
            if text:
                excerpts.append(text[:700])

        system = """SOURCE-ONLY EN→RU terminology canon builder for an academic or technical book.
Extract only specialist terms whose mistranslation would materially change technical meaning or established Russian usage.
Include important one-off terms, not only repeated terms. Prefer established professional Russian terminology over literal calques.
Copy `source` EXACTLY from the supplied English excerpts. Do not include author names, brands, institutions, ordinary prose, or standalone acronyms.
Return ONLY JSON {\"terms\":[{\"source\":\"exact English term\",\"ru\":\"canonical Russian term\",\"confidence\":0.0}]}.
Return at most 18 terms and omit anything genuinely ambiguous."""
        try:
            obj = _giga_json(self.backend, system, {"excerpts": excerpts}, max_tokens=2200)
            self.stats["analysis_calls"] = int(self.stats.get("analysis_calls") or 0) + 1
            self.stats["domain_term_calls"] = int(self.stats.get("domain_term_calls") or 0) + 1
        except Exception as exc:
            print(f"[v10-crossdomain-terms] error={type(exc).__name__}", flush=True)
            return 0

        added = 0
        for item in obj.get("terms") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            ru = _norm(item.get("ru") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if not source or confidence < 0.82 or not _source_phrase_present(joined, source):
                continue
            if not _has_clean_russian(ru) or len(ru.split()) > 8:
                continue
            key = source.casefold()
            if key in memory.glossary:
                continue
            memory.glossary[key] = ru
            added += 1
        return added

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        memory, stats = super().build(segments)
        stats = dict(stats)

        cached_marker = ""
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
            cached_marker = str(data.get("crossdomain_release_schema") or "")
        except Exception:
            data = {}

        added = 0
        if cached_marker != _CROSSDOMAIN_CACHE_MARKER:
            added = self._augment_domain_glossary(segments, memory)
            try:
                if not isinstance(data, dict):
                    data = {}
                data["glossary"] = dict(memory.glossary)
                data["crossdomain_release_schema"] = _CROSSDOMAIN_CACHE_MARKER
                self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            except Exception:
                pass

        stats["domain_terms_added"] = added
        stats["crossdomain_release_schema"] = _CROSSDOMAIN_CACHE_MARKER
        return memory, stats


class FinalV10QualityQA(_UniversalQualityQA):
    """Cross-domain publication QA: relations, terminology, citations and boundaries."""

    @staticmethod
    def _dimension_relation_issues(segment: Segment, target: str) -> list[V10Issue]:
        out = list(_UniversalQualityQA._dimension_relation_issues(segment, target))
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")

        # Generic attributive inch measurement plus a separate explicit feet-long
        # measurement. The first quantity describes section/width/diameter unless
        # source explicitly says otherwise; merely preserving both numbers is not enough.
        dual = bool(re.search(
            r"\b[a-z0-9/]+[- ]inch\b[^.!?]{0,120},[^.!?]{0,120}\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+feet?\s+long\b",
            source,
            re.I,
        ))
        bad_binding = bool(
            re.search(r"\b[а-яё]*дюйм\w*\s+в\s+длин\w*\b", low)
            or re.search(r"\bдлин\w*\s+(?:в\s+)?(?:\w+\s+){0,2}[а-яё]*дюйм\w*\b", low)
        )
        if dual and bad_binding and not any(row.code == "dimension_relation" for row in out):
            out.append(V10Issue(
                segment.id,
                "dimension_relation",
                "semantic",
                "hard",
                "source has separate inch cross-section and feet-long dimensions, but Russian attaches the inch value to length",
            ))
        return out

    @staticmethod
    def _academic_notation_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        if not _technical_style(memory):
            return []
        source = str(segment.text or "")
        target_text = str(target or "")
        out: list[V10Issue] = []

        missing_acronyms = [token for token in sorted(_source_acronyms(source)) if token not in target_text]
        if missing_acronyms:
            out.append(V10Issue(
                segment.id,
                "acronym_fidelity",
                "local",
                "hard",
                "source technical acronym/identifier lost: " + ", ".join(missing_acronyms[:6]),
            ))

        missing_citations = [token for token in sorted(_citation_surnames(source)) if token not in target_text]
        if missing_citations:
            out.append(V10Issue(
                segment.id,
                "citation_fidelity",
                "local",
                "hard",
                "academic citation surname spelling should remain source-stable: " + ", ".join(missing_citations[:6]),
            ))
        return out

    @staticmethod
    def _technical_term_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        if not _technical_style(memory):
            return []
        source = str(segment.text or "")
        out: list[V10Issue] = []
        for en, ru in memory.glossary.items():
            term = str(en or "").strip()
            expected = str(ru or "").strip()
            if not term or not expected or not _source_phrase_present(source, term):
                continue
            if " " not in term and "-" not in term:
                continue
            if _ru_phrase_present(expected, target):
                continue
            out.append(V10Issue(
                segment.id,
                "technical_term",
                "semantic",
                "hard",
                f"source technical term {term!r} should preserve dynamic book glossary term {expected!r}",
            ))
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan_segment(segment, target, memory))
        issues.extend(self._academic_notation_issues(segment, target, memory))
        issues.extend(self._technical_term_issues(segment, target, memory))
        unique = {(row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())

    def scan(self, segments: list[Segment], translated: dict[str, str], memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan(segments, translated, memory))
        for current, nxt in zip(segments, segments[1:]):
            source = str(current.text or "").rstrip()
            next_source = str(nxt.text or "").lstrip()
            target = str(translated.get(current.id) or "").rstrip()
            if not source or not next_source or not target:
                continue
            source_continues = source.endswith((",", ";", ":")) and bool(re.match(r"^[a-z]", next_source))
            target_closed = bool(re.search(r"[.!?][»\"']?$", target))
            if source_continues and target_closed:
                issues.append(V10Issue(
                    current.id,
                    "segment_boundary",
                    "semantic",
                    "hard",
                    "source sentence continues into the next segment but Russian closes it early",
                ))
        unique = {(row.id, row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


class FinalDeepSeekSemanticSpecialist(_UniversalDeepSeekSemanticSpecialist):
    _CROSSDOMAIN_CODES = frozenset({
        "dimension_relation",
        "technical_term",
        "acronym_fidelity",
        "citation_fidelity",
        "segment_boundary",
    })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        out = _UniversalDeepSeekSemanticSpecialist._proven_codes_by_id(issues)
        for issue in issues:
            if issue.severity == "hard" and issue.code in FinalDeepSeekSemanticSpecialist._CROSSDOMAIN_CODES:
                out.setdefault(issue.id, set()).add(issue.code)
        return out


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
