from __future__ import annotations

# Compatibility facade. The production release implementation is intentionally
# book-agnostic. v10_universal_release layers cross-domain notation/name handling
# over the general release core without embedding any fixture-book vocabulary.
from .v10_universal_release import (
    FinalBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA,
    ResidualHardDeepSeekRepair,
)

__all__ = [
    "FinalBookBibleBuilder",
    "FinalDeepSeekSemanticSpecialist",
    "FinalDialogueDiscourseGuard",
    "FinalV10QualityQA",
    "ResidualHardDeepSeekRepair",
]
