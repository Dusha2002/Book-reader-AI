from __future__ import annotations

import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json, _norm
from .v10_general_release import (
    FinalBookBibleBuilder as _GeneralBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _GeneralQualityQA,
    ResidualHardDeepSeekRepair,
)
from .v10_source_bible import SourceOnlyBookBibleBuilder, _has_clean_russian


_LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
_MIXED_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9-]*")
_SHORT_DEFINITION_RE = re.compile(
    r"\b([a-z][a-z'-]{2,8})\b\s+(?:had|has|have|is|was|were)\s+(?:no|not|only|a|an|the)\b",
    re.I,
)
_ACRONYM_TERM_RE = re.compile(
    r"\b((?:[a-z][a-z'-]*\s+){1,4}[a-z][a-z'-]*)\s+or\s+([A-Z][A-Z0-9-]{1,})\b"
)
_COMMON_SHORT = {
    "one", "thing", "man", "woman", "time", "year", "day", "way", "part", "side",
    "good", "great", "same", "other", "only", "just", "still", "even", "very",
}
_ACRONYM_PREFIX_DROP = {
    "introduced", "introduces", "introduce", "called", "known", "named", "the", "a", "an",
}
_TECHNICAL_STYLE_WORDS = (
    "technical", "academic", "scientific", "expository", "textbook", "engineering", "research",
)


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
    """Latin surnames immediately tied to a bibliographic year may remain Latin."""
    out: set[str] = set()
    text = str(source or "")
    for match in re.finditer(r"\b([A-Z][A-Za-z'-]{2,})\b(?=[^.!?\n]{0,70}\b(?:18|19|20)\d{2}[a-z]?\b)", text):
        out.add(match.group(1))
    for match in re.finditer(r"\b([A-Z][A-Za-z'-]{2,})\s+et\s+al\.?", text):
        out.add(match.group(1))
    return out


def _quoted_latin_symbols(source: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"[\"'“‘]\s*([A-Za-z])\s*[\"'”’]", str(source or ""))
    }


def _source_has_exact_token(source: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", str(source or "")))


def _technical_style(memory: BookMemory) -> bool:
    style = memory.style
    blob = " ".join(
        str(value or "")
        for value in (style.narrative_voice, style.rhythm, style.dialogue, style.humor)
    ).casefold()
    return any(word in blob for word in _TECHNICAL_STYLE_WORDS)


def _technical_source_names(source: str) -> set[str]:
    """Protect brands/organizations in an academic source only from local context.

    Fictional names do not qualify merely by capitalization. We require an English
    relational cue such as `at Google`, `from NVIDIA`, or `by IBM` and later also
    require the whole-book style classifier to identify technical/academic prose.
    """
    out: set[str] = set()
    for match in re.finditer(
        r"\b(?:at|from|by|with|using|via|company|organization|laboratory|lab)\s+([A-Z][A-Za-z0-9-]{2,})\b",
        str(source or ""),
    ):
        out.add(match.group(1))
    return out


def _trim_acronym_term(value: str) -> str:
    words = str(value or "").strip().split()
    while words and words[0].casefold() in _ACRONYM_PREFIX_DROP:
        words.pop(0)
    # A greedy five-word regex can capture one ordinary verb before a compact
    # technical phrase. Keep the semantically dense tail rather than that verb.
    if len(words) > 4:
        words = words[-4:]
    return " ".join(words).strip()


class FinalBookBibleBuilder(_GeneralBookBibleBuilder):
    """Universal source-only memory with stronger short-excerpt recall."""

    @staticmethod
    def _accept_name_item(
        item: dict[str, Any],
        allowed: dict[str, dict[str, Any]],
        *,
        threshold: float,
    ) -> tuple[str, str, str, str, str] | None:
        # Short excerpts give less repeated evidence. Keep spelling/Russian sanity
        # guards, but lower only the confidence floor; the canon remains source-only.
        relaxed = 0.60 if threshold >= 0.70 else 0.56
        return SourceOnlyBookBibleBuilder._accept_name_item(
            item,
            allowed,
            threshold=min(float(threshold), relaxed),
        )

    def _name_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        # The base recovery pass only revisits frequency>=4. A smoke excerpt may
        # contain an important recurring name 2-3 times, so make every already-
        # detected proper-name candidate eligible for that *same* source-only pass.
        boosted: list[dict[str, Any]] = []
        for row in rows:
            copy = dict(row)
            copy["frequency"] = max(4, int(copy.get("frequency") or 0))
            boosted.append(copy)
        super()._name_batches(boosted, aggregate)

        # Composite names must be compositionally consistent when their components
        # were independently accepted (e.g. Firstname Surname). No fixture spellings.
        canon = aggregate.get("canonicals") or {}
        for source in list(canon):
            parts = str(source).split()
            if len(parts) < 2 or not all(part in canon for part in parts):
                continue
            canon[source] = " ".join(str(canon[part]) for part in parts)

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        records = [dict(row) for row in _GeneralBookBibleBuilder._candidate_records(segments)]
        existing = {str(row.get("candidate") or "").casefold() for row in records}

        for segment in segments:
            text = str(segment.text or "")
            # Source-defined specialist noun: `X had no cutting edge, only...`.
            for match in _SHORT_DEFINITION_RE.finditer(text):
                candidate = match.group(1).casefold()
                if candidate in existing or candidate in _COMMON_SHORT:
                    continue
                records.append({
                    "candidate": candidate,
                    "kind_hint": "technical_term",
                    "frequency": 1,
                    "evidence": "local_definition",
                    "contexts": [{"chapter": str(segment.chapter or ""), "text": _norm(text)[:520]}],
                })
                existing.add(candidate)

            # Explicit technical definition: `long short-term memory or LSTM`.
            for match in _ACRONYM_TERM_RE.finditer(text):
                candidate = _trim_acronym_term(match.group(1)).casefold()
                acronym = match.group(2)
                if not candidate or candidate in existing or len(candidate.split()) < 2:
                    continue
                records.append({
                    "candidate": candidate,
                    "kind_hint": "technical_term",
                    "frequency": 1,
                    "evidence": "acronym_definition",
                    "acronym": acronym,
                    "contexts": [{"chapter": str(segment.chapter or ""), "text": _norm(text)[:520]}],
                })
                existing.add(candidate)
        return records

    def _term_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        super()._term_batches(rows, aggregate)

        # A second, tiny recovery pass is reserved for terms with explicit source
        # evidence. This is cheaper and safer than lowering the threshold for every
        # random n-gram generated from a technical book.
        residual = [
            row for row in rows
            if row.get("evidence") in {"local_definition", "acronym_definition"}
            and str(row.get("candidate") or "").casefold() not in aggregate["glossary"]
        ]
        if not residual:
            return
        system = """SOURCE-ONLY EN→RU terminology recovery. Each candidate is backed by an explicit local definition or acronym definition in the English source. Decide whether it has a stable specialist Russian term in that context. Translate the TERM, not the surrounding sentence; do not transliterate an ordinary English common noun. Preserve standard acronyms separately. Return ONLY JSON {"terms":[{"source":"...","ru":"...","confidence":0.0}]} and omit genuinely ambiguous candidates."""
        try:
            obj = _giga_json(self.backend, system, {"candidates": residual}, max_tokens=1800)
            self.stats["analysis_calls"] += 1
            self.stats["definition_term_recovery_calls"] = int(self.stats.get("definition_term_recovery_calls") or 0) + 1
        except Exception as exc:
            print(f"[v10-universal-term-recovery] error={type(exc).__name__}", flush=True)
            return
        allowed = {str(row.get("candidate") or "").casefold() for row in residual}
        recovered = 0
        for item in obj.get("terms") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip().casefold()
            ru = _norm(item.get("ru") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if source not in allowed or confidence < 0.76 or not _has_clean_russian(ru):
                continue
            if any(ch in ru for ch in "()[]{}") or len(ru.split()) > 6:
                continue
            aggregate["glossary"][source] = ru
            recovered += 1
        self.stats["definition_terms_recovered"] = int(self.stats.get("definition_terms_recovered") or 0) + recovered

    def _style(self, segments: list[Segment], aggregate: dict[str, Any]) -> None:
        sample_count = min(24, len(segments))
        excerpts: list[dict[str, str]] = []
        if sample_count:
            for n in range(sample_count):
                index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
                segment = segments[index]
                excerpts.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:650]})
        system = """Infer SOURCE-ONLY EN→RU translation guidance from excerpts of an arbitrary book. The source may be literary fiction, narrative nonfiction, academic/technical prose, or general nonfiction. Do not assume a novel. Identify the observable domain and describe what the translator must preserve: register, sentence rhythm, dialogue behavior if any, humor/irony if any, terminology/notation discipline, and a short continuity summary. ONLY JSON {"domain":"literary_fiction|narrative_nonfiction|academic_technical|general_nonfiction|other","style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},"summary":"..."}."""
        try:
            obj = _giga_json(self.backend, system, {"excerpts": excerpts}, max_tokens=2200)
            self.stats["analysis_calls"] += 1
            if isinstance(obj.get("style"), dict):
                aggregate["style"] = {k: _norm(v) for k, v in obj["style"].items() if _norm(v)}
            aggregate["domain"] = _norm(obj.get("domain") or "")
            aggregate["summary"] = _norm(obj.get("summary") or "")
        except Exception as exc:
            print(f"[v10-universal-style] error={type(exc).__name__}", flush=True)


class FinalV10QualityQA(_GeneralQualityQA):
    """Publication QA that distinguishes notation from untranslated prose."""

    @staticmethod
    def _protected_latin(source: str, memory: BookMemory) -> set[str]:
        protected = set(_citation_tokens(source)) | _quoted_latin_symbols(source)
        for token in _LATIN_TOKEN_RE.findall(str(source or "")):
            if _is_acronym_or_identifier(token):
                protected.add(token)
        if _technical_style(memory):
            protected.update(_technical_source_names(source))
        return protected

    @classmethod
    def _latin_issues(cls, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        target_text = str(target or "")
        protected = cls._protected_latin(source, memory)
        out: list[V10Issue] = []

        mixed = []
        for token in _MIXED_TOKEN_RE.findall(target_text):
            if not re.search(r"[A-Za-z]", token) or not re.search(r"[А-Яа-яЁё]", token):
                continue
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
        for match in _LATIN_TOKEN_RE.finditer(target_text):
            token = match.group(0)
            if token.casefold() == "chapter":
                continue
            if token in protected and _source_has_exact_token(source, token):
                continue
            if len(token) == 1 and re.match(r"-[А-Яа-яЁё]", target_text[match.end():]):
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

    @staticmethod
    def _dimension_relation_issues(segment: Segment, target: str) -> list[V10Issue]:
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")
        out: list[V10Issue] = []
        # If a source object is half-inch in section/diameter but later explicitly
        # `N feet long`, Russian must not turn both measurements into its length.
        has_dual_dimension = bool(re.search(
            r"\bhalf[- ]inch\b[^.!?]{0,140}\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+feet?\s+long\b",
            source,
            re.I,
        ))
        if has_dual_dimension and re.search(r"\bдлин\w*\s+(?:в\s+)?полдюйм\w*\b", low):
            out.append(V10Issue(
                segment.id,
                "half_inch",
                "semantic",
                "hard",
                "half-inch object dimension was attached to length even though source separately states an N-foot length",
            ))
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = [row for row in super().scan_segment(segment, target, memory) if row.code != "latin_leak"]
        issues.extend(self._latin_issues(segment, target, memory))
        issues.extend(self._dimension_relation_issues(segment, target))
        unique = {(row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
