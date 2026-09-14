from __future__ import annotations

# Compatibility facade. The actual release implementation is intentionally
# book-agnostic and lives in v10_general_release.
from .v10_general_release import (
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
