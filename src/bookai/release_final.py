from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter

# Compatibility facade. Production release logic remains source-only and
# book-agnostic; publication policy is inferred per book at runtime.
from .v10 import _giga_json, _norm
from .v10_crossdomain_release import _ru_phrase_present, _source_acronyms, _technical_style
from .v10_entity_graph import harmonize_composite_entities
from .v10_general_release import _STOPWORDS, _TOKEN_RE
from .v10_publication_release import (
    FinalBookBibleBuilder as _PublicationBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _PublicationQualityQA,
    ResidualHardDeepSeekRepair,
)
from .v10_source_bible import _has_clean_russian


_YEAR = r"(?:18|19|20)\d{2}[a-z]?"
_PAREN_CITATION_RE = re.compile(
    rf"\((?=[^()\n]{{0,320}}\b{_YEAR}\b)(?=[^()\n]{{0,320}}\b[A-Z][A-Za-z'’.-]{{2,}}\b)[^()\n]{{1,320}}\)"
)
_INLINE_CITATION_RE = re.compile(
    rf"\b[A-Z][A-Za-z'’.-]+(?:\s+(?:and\s+[A-Z][A-Za-z'’.-]+|et\s+al\.))?\s*\({_YEAR}\)"
)
_RELEASE_MEMORY_SCHEMA = "v10-release-memory-2-compact"


def _citation_spans(source: str) -> list[tuple[int, int]]:
    text = str(source or "")
    spans = [(match.start(), match.end()) for match in _PAREN_CITATION_RE.finditer(text)]
    spans.extend((match.start(), match.end()) for match in _INLINE_CITATION_RE.finditer(text))
    return spans


def _entity_occurs_only_in_citations(source: str, entity: str) -> bool:
    text = str(source or "")
    name = str(entity or "").strip()
    if not text or not name:
        return False
    occurrences = list(re.finditer(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, re.I))
    if not occurrences:
        return False
    spans = _citation_spans(text)
    if not spans:
        return False
    return all(any(start <= match.start() and match.end() <= end for start, end in spans) for match in occurrences)


def _source_fingerprint(segments) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        text = _norm(str(segment.text or ""))
        if not text:
            continue
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _explicit_term_ok(row: dict) -> bool:
    candidate = str(row.get("candidate") or "").strip()
    if not candidate:
        return False
    words = [token.casefold() for token in _TOKEN_RE.findall(candidate)]
    if not words or all(word in _STOPWORDS for word in words):
        return False
    if row.get("evidence") == "local_definition" and len(words) == 1 and candidate[:1].isupper():
        return False
    return True


def _technical_source_identifiers(source: str) -> set[str]:
    """Source-grounded Latin notation/identifiers that are legitimate in RU tech prose.

    This deliberately does not whitelist arbitrary capitalized words. Single uppercase
    letters exclude A/I (English article/pronoun), and named multi-letter identifiers
    need a nearby software/library/framework cue. The rule is book-agnostic and relies
    only on the current English source segment.
    """
    text = str(source or "")
    out = {
        match.group(1)
        for match in re.finditer(r"(?<![A-Za-z0-9])([B-HJ-Z])(?![A-Za-z0-9])", text)
    }
    cue_re = re.compile(
        r"\b(?:libraries?|frameworks?|packages?|software|toolkits?|platforms?|APIs?|SDKs?)\b",
        re.I,
    )
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9.+_-]{2,})\b", text):
        token = match.group(1)
        before = text[max(0, match.start() - 110):match.start()]
        after = text[match.end():min(len(text), match.end() + 100)]
        if cue_re.search(before) and (
            re.match(r"\s*(?:\([^)]*\b(?:18|19|20)\d{2}[a-z]?\b[^)]*\))", after)
            or re.search(r"\b(?:such\s+as|including|like|using|via)\s+$", before, re.I)
            or re.search(r"(?:,|\band\b)\s*$", before, re.I)
        ):
            out.add(token)
    return out


class FinalBookBibleBuilder(_PublicationBookBibleBuilder):
    """Final book-adaptive memory with compact, fingerprinted whole-book analysis.

    Proper names are discovered from the complete source. Hard terminology candidates
    are either explicitly evidenced by the source or highly recurrent lexical phrases;
    old title/domain seed lists never enter the release candidate set. The cache is
    bound to the actual memory source fingerprint so a per-book BookMemory can safely
    be reused across many chapter/window translations.
    """

    @staticmethod
    def _candidate_records(segments):
        inherited = _PublicationBookBibleBuilder._candidate_records(segments)
        proper_rows = [dict(row) for row in inherited if row.get("kind_hint") == "proper"]

        explicit_rows = [
            dict(row)
            for row in inherited
            if row.get("kind_hint") == "technical_term"
            and row.get("evidence") in {"local_definition", "acronym_definition"}
            and _explicit_term_ok(row)
        ]
        explicit_rows.sort(
            key=lambda row: (
                row.get("evidence") == "acronym_definition",
                len(str(row.get("candidate") or "").split()) > 1,
                int(row.get("frequency") or 0),
                len(str(row.get("candidate") or "")),
            ),
            reverse=True,
        )
        explicit_rows = explicit_rows[:24]

        records = proper_rows + explicit_rows
        existing = {str(row.get("candidate") or "").casefold() for row in records}
        proper_lower = {str(row.get("candidate") or "").casefold() for row in proper_rows}

        joined = "\n".join(str(segment.text or "") for segment in segments)
        component_rows = []
        for row in list(proper_rows):
            composite = str(row.get("candidate") or "").strip()
            if not (2 <= len(composite.split()) <= 4):
                continue
            full_count = len(re.findall(rf"(?<![A-Za-z]){re.escape(composite)}(?![A-Za-z])", joined))
            if full_count < 1:
                continue
            for part in composite.split():
                key = part.casefold()
                if key in existing or key in _STOPWORDS or len(part) < 3:
                    continue
                total = len(re.findall(rf"(?<![A-Za-z]){re.escape(part)}(?![A-Za-z])", joined))
                standalone = total - full_count
                if standalone < 1:
                    continue
                contexts = []
                for segment in segments:
                    text = str(segment.text or "")
                    if re.search(rf"(?<![A-Za-z]){re.escape(part)}(?![A-Za-z])", text) and not re.fullmatch(
                        rf"\s*{re.escape(composite)}[\s'’.,;:!?-]*", text
                    ):
                        contexts.append({"chapter": str(segment.chapter or ""), "text": _norm(text)[:520]})
                    if len(contexts) >= 3:
                        break
                component_rows.append({
                    "candidate": part,
                    "kind_hint": "proper",
                    "frequency": standalone,
                    "mid_frequency": standalone,
                    "title_evidence": 0,
                    "component_of": composite,
                    "contexts": contexts,
                })
                existing.add(key)
                proper_lower.add(key)
        records.extend(component_rows)

        counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}

        def remember(key: str, segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 3:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:520]})

        for segment in segments:
            raw_tokens = _TOKEN_RE.findall(str(segment.text or ""))
            tokens = [token.casefold() for token in raw_tokens]
            for n in (1, 2, 3):
                for i in range(0, max(0, len(tokens) - n + 1)):
                    window = tokens[i:i + n]
                    if n == 1:
                        token = window[0]
                        if token in _STOPWORDS or ("-" not in token and len(token) < 10):
                            continue
                    else:
                        content = [token for token in window if token not in _STOPWORDS and len(token) >= 4]
                        if len(content) < 2:
                            continue
                        if window[0] in _STOPWORDS and window[-1] in _STOPWORDS:
                            continue
                    phrase = " ".join(window)
                    if phrase in proper_lower or phrase in existing:
                        continue
                    counts[phrase] += 1
                    remember(phrase, segment)

        ranked = sorted(
            (
                (phrase, count)
                for phrase, count in counts.items()
                if (phrase.count(" ") and 3 <= count <= 36)
                or (not phrase.count(" ") and 4 <= count <= 24)
            ),
            key=lambda item: (
                item[0].count(" ") >= 1,
                item[0].count(" "),
                min(item[1], 12),
                len(item[0]),
            ),
            reverse=True,
        )
        for phrase, count in ranked[:48]:
            records.append({
                "candidate": phrase,
                "kind_hint": "technical_term",
                "frequency": count,
                "contexts": contexts.get(phrase, []),
                "evidence": "recurring_source_ngram",
            })
        return records

    def _term_batches(self, rows, aggregate):
        batch_size = max(16, min(32, int(os.getenv("BOOKAI_V10_TERM_BATCH") or "24")))
        system = """Build a SOURCE-ONLY HIGH-PRECISION EN→RU terminology glossary for a book of any genre.
Candidates come only from the current English source. Keep an item ONLY when its denotation is stable across the supplied
contexts and a book-wide Russian canon is genuinely useful. Omit ordinary prose, names, sentence fragments, and ambiguous
polysemy. For academic/technical concepts prefer established professional Russian terminology; for historical/literary
specialist objects prefer the precise conventional object name. Preserve category breadth: never replace a broad class with
one subtype. Return ONLY the concise Russian TERM itself, no explanation, alternatives, parentheses or definition.
No Russian reference translation exists. ONLY JSON
{"terms":[{"source":"...","ru":"...","confidence":0.0}]}.
"""
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            allowed = {str(row["candidate"]).casefold(): row for row in batch}
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=3200)
                self.stats["analysis_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible-terms] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                continue
            for item in obj.get("terms") or []:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source") or "").strip().casefold()
                ru = _norm(item.get("ru") or "")
                try:
                    confidence = float(item.get("confidence") or 0)
                except Exception:
                    confidence = 0.0
                if (
                    source not in allowed
                    or confidence < 0.92
                    or not _has_clean_russian(ru)
                    or any(ch in ru for ch in "()[]{}")
                    or len(ru.split()) > 7
                ):
                    self.stats["rejected_terms"] += 1
                    continue
                aggregate["glossary"][source] = ru

    def _style(self, segments, aggregate):
        sample_count = min(20, len(segments))
        excerpts = []
        if sample_count:
            for n in range(sample_count):
                index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
                seg = segments[index]
                excerpts.append({"chapter": str(seg.chapter or ""), "text": _norm(seg.text)[:650]})
        system = """Infer SOURCE-ONLY translation guidance from English excerpts distributed across a whole book.
First identify what the prose is (fiction, memoir/narrative nonfiction, academic/technical exposition, general nonfiction,
or other) from source evidence. Do not invent a Russian reference style. Describe only observable properties to preserve:
narrative/expository register, sentence rhythm, dialogue register if any, humor/irony if any, and a short continuity/topic
summary. ONLY JSON {"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},"summary":"..."}."""
        try:
            obj = _giga_json(self.backend, system, {"excerpts": excerpts}, max_tokens=2200)
            self.stats["analysis_calls"] += 1
            if isinstance(obj.get("style"), dict):
                aggregate["style"] = {k: _norm(v) for k, v in obj["style"].items() if _norm(v)}
            aggregate["summary"] = _norm(obj.get("summary") or "")
        except Exception as exc:
            print(f"[v10-bible-style] error={type(exc).__name__}", flush=True)

    def build(self, segments):
        fingerprint = _source_fingerprint(segments)
        cache_path = getattr(self, "cache_path", None)
        if cache_path and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text("utf-8"))
            except Exception:
                data = {}
            if (
                not isinstance(data, dict)
                or data.get("release_memory_schema") != _RELEASE_MEMORY_SCHEMA
                or data.get("release_source_fingerprint") != fingerprint
            ):
                try:
                    cache_path.unlink()
                except OSError:
                    pass

        memory, stats = super().build(segments)

        source_acronyms: set[str] = set()
        if _technical_style(memory):
            source_acronyms = {
                token
                for segment in segments
                for token in _source_acronyms(str(segment.text or ""))
            }
            for token in source_acronyms:
                memory.acronyms[token] = token

        entity_graph = harmonize_composite_entities(memory, cache_path)

        try:
            data = json.loads(cache_path.read_text("utf-8")) if cache_path and cache_path.exists() else {}
            if isinstance(data, dict):
                if source_acronyms:
                    data["acronyms"] = dict(memory.acronyms)
                    data["source_acronym_fail_closed"] = True
                data["release_memory_schema"] = _RELEASE_MEMORY_SCHEMA
                data["release_source_fingerprint"] = fingerprint
                cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass

        stats = dict(stats)
        stats["entity_graph"] = entity_graph
        stats["source_acronym_fail_closed"] = bool(source_acronyms)
        stats["source_acronyms_preserved"] = len(source_acronyms)
        stats["legacy_term_seeds_used"] = False
        stats["release_memory_schema"] = _RELEASE_MEMORY_SCHEMA
        stats["release_source_fingerprint"] = fingerprint[:16]
        return memory, stats


class MorphologyAwarePublicationQA(_PublicationQualityQA):
    """Final publication QA with morphology, citations and technical notation policy."""

    @staticmethod
    def _protected_latin(source, memory):
        protected = set(_PublicationQualityQA._protected_latin(source, memory))
        if _technical_style(memory):
            protected.update(_technical_source_identifiers(str(source or "")))
        return protected

    def scan_segment(self, segment, target, memory):
        issues = list(super().scan_segment(segment, target, memory))
        filtered = []
        glossary_casefold = {
            str(key or "").casefold(): str(value or "")
            for key, value in memory.glossary.items()
        }
        for issue in issues:
            if issue.code == "name_canon":
                match = re.search(r"source entity '([^']+)' must preserve", str(issue.reason or ""))
                if match and _entity_occurs_only_in_citations(str(segment.text or ""), match.group(1)):
                    continue
                filtered.append(issue)
                continue

            if issue.code != "book_term_canon":
                filtered.append(issue)
                continue
            match = re.search(r"source term '([^']+)' must preserve", str(issue.reason or ""))
            if not match:
                filtered.append(issue)
                continue
            expected = glossary_casefold.get(match.group(1).casefold(), "")
            if expected and _ru_phrase_present(expected, target):
                continue
            filtered.append(issue)
        return filtered


FinalV10QualityQA = MorphologyAwarePublicationQA


__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
