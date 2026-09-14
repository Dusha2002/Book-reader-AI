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

_PUBLICATION_CACHE_MARKER = "v10-publication-1"
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
_SRC_INLINE_CITATION_RE = re.compile(
    r"\b([A-Z][A-Za-z'’-]+(?:\s+(?:and\s+[A-Z][A-Za-z'’-]+|et\s+al\.))?)\s*"
    r"\(((?:18|19|20)\d{2}[a-z]?)\)"
)


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
    pattern = re.compile(rf"\b{re.escape(token)}(?:s)?\b")
    rows: list[str] = []
    for segment in segments:
        if pattern.search(str(segment.text or "")):
            rows.append(_norm(segment.text)[:520])
            if len(rows) >= limit:
                break
    return rows


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
For every supplied English acronym, choose the conventional printed form in modern Russian technical literature: use an established Russian abbreviation when that is the normal form, otherwise preserve the Latin acronym. Do not spell it out in the `target` field.
Also audit the excerpts for up to 10 high-risk specialist terms that are missing from the existing glossary or whose existing Russian canonical is misleading, incomplete, overly literal, or incorrectly narrows a broad source category to one subtype. Include important one-off methods, model families and architecture names.
Do not return ordinary prose, people, organizations or brands as terms. Copy every term `source` exactly from the English excerpts.
Return ONLY JSON {"acronyms":[{"source":"CPU","target":"CPU","confidence":0.0}],"terms":[{"source":"exact English term","ru":"best established Russian term","confidence":0.0}]}.
Return every supplied acronym. Prefer terminology that a professional Russian technical editor would print."""
        try:
            obj = _giga_json(
                self.backend,
                system,
                {
                    "acronyms": acronym_rows,
                    "excerpts": excerpts,
                    "existing_glossary": dict(memory.glossary),
                },
                max_tokens=2600,
            )
            self.stats["analysis_calls"] = int(self.stats.get("analysis_calls") or 0) + 1
            self.stats["publication_policy_calls"] = int(self.stats.get("publication_policy_calls") or 0) + 1
        except Exception as exc:
            print(f"[v10-publication-policy] error={type(exc).__name__}", flush=True)
            for token in acronyms:
                memory.acronyms.setdefault(token, token)
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
            canonical = target if target and confidence >= 0.72 else token
            if memory.acronyms.get(token) != canonical:
                acronym_updates += 1
            memory.acronyms[token] = canonical

        joined = "\n".join(str(segment.text or "") for segment in segments)
        terms_added = 0
        terms_refined = 0
        for item in obj.get("terms") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            ru = _norm(item.get("ru") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if not source or confidence < 0.80 or not _source_phrase_present(joined, source):
                continue
            if not _has_clean_russian(ru) or len(ru.split()) > 9:
                continue
            key = source.casefold()
            old = str(memory.glossary.get(key) or "").strip()
            if old:
                if confidence >= 0.86 and old.casefold() != ru.casefold():
                    memory.glossary[key] = ru
                    terms_refined += 1
            else:
                memory.glossary[key] = ru
                terms_added += 1

        print(
            f"[v10-publication-policy] acronyms={len(acronyms)} "
            f"acronym_updates={acronym_updates} terms_added={terms_added} terms_refined={terms_refined}",
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
        stats["publication_release_schema"] = _PUBLICATION_CACHE_MARKER
        return memory, stats


class FinalV10QualityQA(_CrossDomainQualityQA):
    """Uses per-book acronym policy instead of requiring every Latin source acronym verbatim."""

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
            expected = str(memory.acronyms.get(token) or token).strip()
            if _contains_form(target, expected):
                continue
            base.append(V10Issue(
                segment.id,
                "acronym_fidelity",
                "local",
                "hard",
                f"source acronym {token!r} should use publication canonical {expected!r}",
            ))
        return base


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
