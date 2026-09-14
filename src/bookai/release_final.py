from __future__ import annotations

import re

# Compatibility facade. Production release logic remains source-only and
# book-agnostic; publication policy is inferred per book at runtime.
from .v10_crossdomain_release import _ru_phrase_present
from .v10_publication_release import (
    FinalBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA as _PublicationQualityQA,
    ResidualHardDeepSeekRepair,
)


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
