from __future__ import annotations

# Compatibility facade. Production release logic stays book-agnostic: the
# cross-domain layer adds terminology, notation, relation and boundary fidelity
# on top of the universal source-only release core, without fixture-book rules.
from .v10_crossdomain_release import (
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
