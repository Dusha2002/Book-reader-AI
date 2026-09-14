from __future__ import annotations

import re

# Compatibility facade. Production release logic remains source-only and
# book-agnostic; publication policy is inferred per book at runtime.
from .v10_crossdomain_release import _ru_phrase_present
from .v10_entity_graph import harmonize_composite_entities
from .v10_publication_release import (
    FinalBookBibleBuilder as _PublicationBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _PublicationQualityQA,
    ResidualHardDeepSeekRepair,
)


class FinalBookBibleBuilder(_PublicationBookBibleBuilder):
    """Publication memory plus a conservative compositional entity graph.

    Full names are not allowed to invent a spelling that conflicts with independently
    accepted component entities. No title-specific names or target spellings live here.
    """

    def build(self, segments):
        memory, stats = super().build(segments)
        entity_graph = harmonize_composite_entities(memory, getattr(self, "cache_path", None))
        stats = dict(stats)
        stats["entity_graph"] = entity_graph
        return memory, stats


class MorphologyAwarePublicationQA(_PublicationQualityQA):
    """Suppress only legacy book-term false positives caused by Russian inflection.

    The underlying publication QA still owns terminology policy. This final facade
    merely replaces its old first-word prefix test with the cross-domain phrase
    matcher when a `book_term_canon` issue is emitted.
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
