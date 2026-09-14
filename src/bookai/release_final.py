from __future__ import annotations

import re
from collections import Counter

# Compatibility facade. Production release logic remains source-only and
# book-agnostic; publication policy is inferred per book at runtime.
from .v10 import _norm
from .v10_crossdomain_release import _ru_phrase_present
from .v10_entity_graph import harmonize_composite_entities
from .v10_general_release import _STOPWORDS, _TOKEN_RE
from .v10_publication_release import (
    FinalBookBibleBuilder as _PublicationBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _PublicationQualityQA,
    ResidualHardDeepSeekRepair,
)


class FinalBookBibleBuilder(_PublicationBookBibleBuilder):
    """Final book-adaptive memory with no legacy title-specific term seeds.

    Proper-name discovery and *explicit source-evidence* terminology (for example a
    local definition or acronym definition) are reused from the source-only builder.
    All other specialist candidates are rebuilt only from recurring lexical n-grams
    in the CURRENT source, so old book-specific vocabulary cannot influence release.
    Composite entities are harmonized from independently accepted components.
    """

    @staticmethod
    def _candidate_records(segments):
        inherited = _PublicationBookBibleBuilder._candidate_records(segments)
        allowed_evidence = {"local_definition", "acronym_definition"}
        records = [
            dict(row)
            for row in inherited
            if row.get("kind_hint") == "proper" or row.get("evidence") in allowed_evidence
        ]
        existing = {str(row.get("candidate") or "").casefold() for row in records}
        proper_lower = {
            str(row.get("candidate") or "").casefold()
            for row in records
            if row.get("kind_hint") == "proper"
        }

        counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}

        def remember(key: str, segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 3:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:520]})

        for segment in segments:
            tokens = [token.casefold() for token in _TOKEN_RE.findall(str(segment.text or ""))]
            for n in (1, 2, 3):
                for i in range(0, max(0, len(tokens) - n + 1)):
                    window = tokens[i:i + n]
                    if n == 1:
                        token = window[0]
                        if len(token) < 7 or token in _STOPWORDS:
                            continue
                    else:
                        content = [token for token in window if token not in _STOPWORDS and len(token) >= 4]
                        if len(content) < 2:
                            continue
                        if window[0] in _STOPWORDS and window[-1] in _STOPWORDS:
                            continue
                    phrase = " ".join(window)
                    if phrase in proper_lower:
                        continue
                    counts[phrase] += 1
                    remember(phrase, segment)

        ranked = sorted(
            ((phrase, count) for phrase, count in counts.items() if 2 <= count <= 40 and phrase not in existing),
            key=lambda item: (
                item[0].count(" ") >= 1,
                item[0].count(" "),
                min(item[1], 12),
                len(item[0]),
            ),
            reverse=True,
        )
        for phrase, count in ranked[:96]:
            records.append({
                "candidate": phrase,
                "kind_hint": "technical_term",
                "frequency": count,
                "contexts": contexts.get(phrase, []),
            })
        return records

    def build(self, segments):
        memory, stats = super().build(segments)
        entity_graph = harmonize_composite_entities(memory, getattr(self, "cache_path", None))
        stats = dict(stats)
        stats["entity_graph"] = entity_graph
        stats["legacy_term_seeds_used"] = False
        return memory, stats


class MorphologyAwarePublicationQA(_PublicationQualityQA):
    """Suppress only book-term false positives caused by Russian inflection.

    The underlying publication QA still owns terminology policy. This final facade
    replaces its old first-word prefix test with the cross-domain phrase matcher when
    a `book_term_canon` issue is emitted.
    """

    def scan_segment(self, segment, target, memory):
        issues = list(super().scan_segment(segment, target, memory))
        filtered = []
        glossary_casefold = {
            str(key or "").casefold(): str(value or "")
            for key, value in memory.glossary.items()
        }
        for issue in issues:
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
