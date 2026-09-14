from __future__ import annotations

import json
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json, _norm
from .v10_source_bible import _has_clean_russian
from .v10_crossdomain_release import (
    FinalBookBibleBuilder as _CrossDomainBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard as _CrossDomainDialogueGuard,
    FinalV10QualityQA as _CrossDomainQualityQA,
    ResidualHardDeepSeekRepair,
    _source_acronyms,
    _source_phrase_present,
    _technical_style,
)

_PUBLICATION_CACHE_MARKER = "v10-publication-2"
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
_SRC_INLINE_CITATION_RE = re.compile(
    r"\b([A-Z][A-Za-z'’-]+(?:\s+(?:and\s+[A-Z][A-Za-z'’-]+|et\s+al\.))?)\s*"
    r"\(((?:18|19|20)\d{2}[a-z]?)\)"
)
_ACRONYM_DEFINITION_RE = re.compile(
    r"\b((?:[a-z][a-z'-]*\s+){1,6}[a-z][a-z'-]*)\s+or\s+([A-Z][A-Z0-9-]{1,})\b"
)
_CITATION_ADJACENT_TERM_RE = re.compile(
    r"\b([A-Z][a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){1,3})\s*"
    r"\([^()]{0,140}\b(?:18|19|20)\d{2}[a-z]?\b"
)
_TERM_PREFIX_DROP = {
    "introduced", "introduces", "introduce", "called", "known", "named", "the", "a", "an",
    "their", "this", "that", "these", "those",
}


def _clean_acronym_target(value: str) -> str:
    target = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.+/-]{0,15}", target):
        return ""
    return target


def _contains_form(text: str, form: str) -> bool:
    value = str(form or "").strip()
    if not value:
        return False
    return bool(re.search(rf"(?<![A-Za-zА-Яа-яЁё0-9]){re.escape(value)}(?![A-Za-zА-Яа-яЁё0-9])", str(text or "")))


def _source_contexts(segments: list[Segment], token: str, limit: int = 2) -> list[str]:
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])", re.I)
    rows: list[str] = []
    for segment in segments:
        if pattern.search(str(segment.text or "")):
            rows.append(_norm(segment.text)[:620])
            if len(rows) >= limit:
                break
    return rows


def _trim_definition_term(value: str) -> str:
    words = str(value or "").strip().split()
    while words and words[0].casefold() in _TERM_PREFIX_DROP:
        words.pop(0)
    if len(words) > 5:
        words = words[-5:]
        while words and words[0].casefold() in _TERM_PREFIX_DROP:
            words.pop(0)
    return " ".join(words).strip()


def _priority_term_candidates(segments: list[Segment], memory: BookMemory) -> list[dict[str, Any]]:
    """Build source-only high-value term rows without title/book-specific vocabulary."""
    joined = "\n".join(str(segment.text or "") for segment in segments)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source: str, evidence: str, current_ru: str = "") -> None:
        term = str(source or "").strip()
        key = term.casefold()
        if not term or key in seen or not _source_phrase_present(joined, term):
            return
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", term)
        if len(words) < 2 or len(words) > 6:
            return
        seen.add(key)
        rows.append({
            "source": term,
            "current_ru": str(current_ru or ""),
            "evidence": evidence,
            "contexts": _source_contexts(segments, term, limit=2),
        })

    # Explicit source definitions such as `long short-term memory or LSTM` are
    # high-value even when they occur only once.
    for segment in segments:
        text = str(segment.text or "")
        for match in _ACRONYM_DEFINITION_RE.finditer(text):
            term = _trim_definition_term(match.group(1))
            add(term, f"explicit_acronym_definition:{match.group(2)}", memory.glossary.get(term.casefold(), ""))

    # A capitalized multiword concept directly introducing an author-year citation
    # is often a named method/family rather than ordinary prose (e.g. a method class).
    for segment in segments:
        for match in _CITATION_ADJACENT_TERM_RE.finditer(str(segment.text or "")):
            term = match.group(1).strip()
            add(term, "citation_adjacent_concept", memory.glossary.get(term.casefold(), ""))

    # Existing multiword technical canon must be explicitly reviewed: the earlier
    # candidate pass may find the right source phrase but choose an over-literal RU canon.
    for source, ru in memory.glossary.items():
        term = str(source or "").strip()
        if " " not in term and "-" not in term:
            continue
        add(term, "existing_multitoken_glossary", str(ru or ""))

    return rows[:24]


def _restore_parenthetical_citations(source: str, target: str) -> str:
    value = str(target or "")
    source_groups: list[tuple[str, tuple[str, ...]]] = []
    for match in re.finditer(r"\(([^()\n]{1,260})\)", str(source or "")):
        body = match.group(1)
        years = tuple(_YEAR_RE.findall(body))
        if not years or not re.search(r"\b[A-Z][A-Za-z'’-]{2,}\b", body):
            continue
        source_groups.append((match.group(0), years))

    for source_group, years in source_groups:
        for match in list(re.finditer(r"\(([^()\n]{1,300})\)", value)):
            if tuple(_YEAR_RE.findall(match.group(1))) != years:
                continue
            value = value[: match.start()] + source_group + value[match.end() :]
            break
    return value


def _restore_inline_citations(source: str, target: str) -> str:
    value = str(target or "")
    for match in _SRC_INLINE_CITATION_RE.finditer(str(source or "")):
        source_label = match.group(1)
        year = match.group(2)
        target_pattern = re.compile(
            r"(?P<label>[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'’.-]+"
            r"(?:\s+(?:и|and)\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'’.-]+|"
            r"\s+(?:с\s+соавторами|et\s+al\.))?)\s*"
            + re.escape(f"({year})")
        )
        target_match = target_pattern.search(value)
        if not target_match:
            continue
        replacement = f"{source_label} ({year})"
        value = value[: target_match.start()] + replacement + value[target_match.end() :]
    return value


def _restore_academic_citations(source: str, target: str) -> str:
    value = _restore_parenthetical_citations(source, target)
    return _restore_inline_citations(source, value)


class FinalDialogueDiscourseGuard(_CrossDomainDialogueGuard):
    """Normalizes dialogue and deterministically preserves author-year citations."""

    @staticmethod
    def _normalize(segment: Segment, text: str) -> str:
        value = _CrossDomainDialogueGuard._normalize(segment, text)
        if not value:
            return value
        return _restore_academic_citations(str(segment.text or ""), value)


class FinalBookBibleBuilder(_CrossDomainBookBibleBuilder):
    """Adds source-derived publication policies for acronyms and high-risk terminology."""

    def _publication_audit(self, segments: list[Segment], memory: BookMemory) -> tuple[int, int, int]:
        if not _technical_style(memory):
            return 0, 0, 0

        acronyms = sorted({token for segment in segments for token in _source_acronyms(segment.text)})
        priority_terms = _priority_term_candidates(segments, memory)
        sample_count = min(20, len(segments))
        excerpts: list[str] = []
        for n in range(sample_count):
            index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
            text = _norm(segments[index].text)
            if text:
                excerpts.append(text[:700])
        if not excerpts:
            return 0, 0, 0

        acronym_rows = [
            {"source": token, "contexts": _source_contexts(segments, token)}
            for token in acronyms
        ]
        system = """You are a Russian technical-publication terminology editor working SOURCE-ONLY.
For every supplied English acronym, determine the conventional printed form in modern Russian specialist literature. Actively distinguish acronyms normally localized into Cyrillic from hardware/model/dataset/organization abbreviations normally preserved in Latin. Do not mechanically preserve the English token. Do not spell an acronym out in `target`.

PRIORITY_TERMS require an explicit editorial decision. For each row decide `keep`, `change`, or `add`. Preserve the semantic breadth of the English category: a broad method/model family MUST NOT be narrowed to one subtype. Prefer the established professional Russian term over a literal calque. Explicit acronym-definition terms and citation-adjacent concepts deserve special attention even if they occur once.

You may additionally return up to 10 missing high-value specialist terms from excerpts. Do not return ordinary prose, people, organizations or brands as terms. Copy every `source` EXACTLY from the supplied source text.
Return ONLY JSON {
  "acronyms":[{"source":"CPU","target":"CPU","confidence":0.0}],
  "priority_terms":[{"source":"exact source term","decision":"keep|change|add","ru":"best Russian canonical or empty for keep","confidence":0.0}],
  "terms":[{"source":"exact English term","ru":"best established Russian term","confidence":0.0}]
}.
Return every supplied acronym and every PRIORITY_TERMS row. Prefer what a professional Russian technical editor would actually print."""
        try:
            obj = _giga_json(
                self.backend,
                system,
                {
                    "acronyms": acronym_rows,
                    "priority_terms": priority_terms,
                    "excerpts": excerpts,
                    "existing_glossary": dict(memory.glossary),
                },
                max_tokens=3600,
            )
            self.stats["analysis_calls"] = int(self.stats.get("analysis_calls") or 0) + 1
            self.stats["publication_policy_calls"] = int(self.stats.get("publication_policy_calls") or 0) + 1
            self.stats["publication_priority_terms"] = len(priority_terms)
        except Exception as exc:
            print(f"[v10-publication-policy] error={type(exc).__name__}", flush=True)
            return 0, 0, 0

        by_source = {
            str(item.get("source") or "").strip(): item
            for item in obj.get("acronyms") or []
            if isinstance(item, dict)
        }
        acronym_updates = 0
        for token in acronyms:
            item = by_source.get(token) or {}
            target = _clean_acronym_target(item.get("target") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0

            # Localized forms are useful hard canon when confidence is solid.
            # Preserved Latin forms need stronger evidence; for short two-letter
            # abbreviations we leave the translator free unless a localized form
            # was actually proposed, avoiding accidental anglicization of Russian prose.
            canonical = ""
            if target and target != token and confidence >= 0.80:
                canonical = target
            elif target == token and len(token) >= 3 and confidence >= 0.90:
                canonical = token
            if canonical:
                if memory.acronyms.get(token) != canonical:
                    acronym_updates += 1
                memory.acronyms[token] = canonical

        joined = "\n".join(str(segment.text or "") for segment in segments)
        terms_added = 0
        terms_refined = 0
        priority_allowed = {row["source"].casefold(): row for row in priority_terms}
        priority_seen = 0
        for item in obj.get("priority_terms") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            key = source.casefold()
            if key not in priority_allowed or not _source_phrase_present(joined, source):
                continue
            priority_seen += 1
            decision = str(item.get("decision") or "").strip().casefold()
            ru = _norm(item.get("ru") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if decision == "keep":
                continue
            if decision not in {"change", "add"} or confidence < 0.80:
                continue
            if not _has_clean_russian(ru) or len(ru.split()) > 9:
                continue
            old = str(memory.glossary.get(key) or "").strip()
            if old:
                if confidence >= 0.82 and old.casefold() != ru.casefold():
                    memory.glossary[key] = ru
                    terms_refined += 1
            else:
                memory.glossary[key] = ru
                terms_added += 1
        self.stats["publication_priority_terms_returned"] = priority_seen

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
            if not _has_clean_russian(ru) or len(ru.split()) > 9:
                continue
            key = source.casefold()
            old = str(memory.glossary.get(key) or "").strip()
            if old:
                if confidence >= 0.88 and old.casefold() != ru.casefold():
                    memory.glossary[key] = ru
                    terms_refined += 1
            else:
                memory.glossary[key] = ru
                terms_added += 1

        print(
            f"[v10-publication-policy] acronyms={len(acronyms)} "
            f"acronym_updates={acronym_updates} priority={len(priority_terms)}/{priority_seen} "
            f"terms_added={terms_added} terms_refined={terms_refined}",
            flush=True,
        )
        return acronym_updates, terms_added, terms_refined

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        memory, stats = super().build(segments)
        stats = dict(stats)
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        marker = str(data.get("publication_release_schema") or "")
        cached_acronyms = data.get("acronyms") or {}
        if marker == _PUBLICATION_CACHE_MARKER and isinstance(cached_acronyms, dict):
            memory.acronyms.update({str(k): str(v) for k, v in cached_acronyms.items() if k and v})
            acronym_updates = terms_added = terms_refined = 0
            cache_hit = True
        else:
            memory.acronyms.clear()
            acronym_updates, terms_added, terms_refined = self._publication_audit(segments, memory)
            data["glossary"] = dict(memory.glossary)
            data["acronyms"] = dict(memory.acronyms)
            data["publication_release_schema"] = _PUBLICATION_CACHE_MARKER
            try:
                self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            except Exception:
                pass
            cache_hit = False

        stats["publication_policy_cache_hit"] = cache_hit
        stats["publication_acronyms"] = len(memory.acronyms)
        stats["publication_acronym_updates"] = acronym_updates
        stats["publication_terms_added"] = terms_added
        stats["publication_terms_refined"] = terms_refined
        stats["publication_priority_terms"] = int(self.stats.get("publication_priority_terms") or 0)
        stats["publication_priority_terms_returned"] = int(self.stats.get("publication_priority_terms_returned") or 0)
        stats["publication_release_schema"] = _PUBLICATION_CACHE_MARKER
        return memory, stats


class FinalV10QualityQA(_CrossDomainQualityQA):
    """Uses per-book publication policy plus stricter cross-segment continuity."""

    @staticmethod
    def _academic_notation_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        base = [
            issue
            for issue in _CrossDomainQualityQA._academic_notation_issues(segment, target, memory)
            if issue.code != "acronym_fidelity"
        ]
        if not _technical_style(memory):
            return base
        source = str(segment.text or "")
        for token in sorted(_source_acronyms(source)):
            # Only enforce abbreviations for which the source-only publication audit
            # established a confident canonical. Uncertain tokens remain translator choice.
            expected = str(memory.acronyms.get(token) or "").strip()
            if not expected or _contains_form(target, expected):
                continue
            base.append(V10Issue(
                segment.id,
                "acronym_fidelity",
                "local",
                "hard",
                f"source acronym {token!r} should use publication canonical {expected!r}",
            ))
        return base

    def scan(self, segments: list[Segment], translated: dict[str, str], memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan(segments, translated, memory))
        for current, nxt in zip(segments, segments[1:]):
            source = str(current.text or "").rstrip()
            next_source = str(nxt.text or "").lstrip()
            target = str(translated.get(current.id) or "").rstrip()
            if not source or not next_source or not target:
                continue
            source_continues = source.endswith((",", ";", ":")) and bool(re.match(r"^[a-z]", next_source))
            if not source_continues:
                continue
            target_is_open = bool(re.search(r"[,;:—–-]\s*$", target))
            if target_is_open:
                continue
            issues.append(V10Issue(
                current.id,
                "segment_boundary",
                "semantic",
                "hard",
                "source sentence remains syntactically open across the next segment, but Russian loses the joining punctuation/open syntax",
            ))
        unique = {(row.id, row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]