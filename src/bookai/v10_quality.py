from __future__ import annotations

import re

from .models import BookMemory, Segment
from .v10 import DeterministicQA, V10Issue


class V10QualityQA(DeterministicQA):
    """Clean-v10 QA with risk-specific priorities learned from v9ad.

    Deterministic checks remain narrow; broader lexical/spatial/kinship signals only
    influence which <=8 rows reach the single DeepSeek semantic batch.
    """

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        out = list(super().scan_segment(segment, target, memory))
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")

        # High-confidence physical direction reversal. This catches errors such as
        # "trudged up the stairs" -> "спускаясь", without trying to solve all motion.
        src_up = bool(re.search(r"\b(?:up\s+the\s+(?:stairs|steps|slope|hill)|(?:trudged|walked|went|ran|climbed|came)\s+up)\b", source, re.I))
        src_down = bool(re.search(r"\b(?:down\s+the\s+(?:stairs|steps|slope|hill)|(?:trudged|walked|went|ran|climbed|came)\s+down)\b", source, re.I))
        if src_up and re.search(r"\b(?:спуска\w*|сош[её]л|вниз)\b", low):
            out.append(V10Issue(segment.id, "direction_relation", "semantic", "hard", "source motion is upward but Russian is downward"))
        if src_down and re.search(r"\b(?:поднима\w*|взош[её]л|вверх)\b", low):
            out.append(V10Issue(segment.id, "direction_relation", "semantic", "hard", "source motion is downward but Russian is upward"))

        # Narrow kinship preservation for possessive English relations. Natural
        # Russian may use племянник/племянница, so either explicit or derived form passes.
        sister_son = re.search(r"\b(?:his|her|their)\s+sister(?:'s|’s)\s+(?:eldest\s+|oldest\s+|younger\s+|youngest\s+)?son\b", source, re.I)
        sister_daughter = re.search(r"\b(?:his|her|their)\s+sister(?:'s|’s)\s+(?:eldest\s+|oldest\s+|younger\s+|youngest\s+)?daughter\b", source, re.I)
        brother_son = re.search(r"\b(?:his|her|their)\s+brother(?:'s|’s)\s+(?:eldest\s+|oldest\s+|younger\s+|youngest\s+)?son\b", source, re.I)
        brother_daughter = re.search(r"\b(?:his|her|their)\s+brother(?:'s|’s)\s+(?:eldest\s+|oldest\s+|younger\s+|youngest\s+)?daughter\b", source, re.I)
        if (sister_son or brother_son) and not ("племян" in low or re.search(r"сын\w*.*(?:сестр|брат)", low)):
            out.append(V10Issue(segment.id, "kinship_relation", "semantic", "hard", "nephew/parent-sibling relation may be lost"))
        if (sister_daughter or brother_daughter) and not ("племян" in low or re.search(r"доч\w*.*(?:сестр|брат)", low)):
            out.append(V10Issue(segment.id, "kinship_relation", "semantic", "hard", "niece/parent-sibling relation may be lost"))

        unique = {(row.code, row.mode, row.reason): row for row in out}
        return list(unique.values())

    @staticmethod
    def semantic_risk(segment: Segment, memory: BookMemory) -> int:
        score = DeterministicQA.semantic_risk(segment, memory)
        text = str(segment.text or "")
        # v9ad was strong because it ranked semantic *types*, not merely paragraph length.
        risk_rules = (
            (r"\b(?:up|down)\s+the\s+(?:stairs|steps|slope|hill)\b", 5),
            (r"\b(?:ahead|behind|before|after|towards?|away\s+from|into|out\s+of)\b", 2),
            (r"\b(?:sister|brother|daughter|son|cousin|nephew|niece|marry|married|marriage)\b", 3),
            (r"\b(?:first|second|third|fourth|fifth|sixth|eldest|oldest|youngest)\b", 2),
            (r"\b(?:hunt|hunting|boar|spinney|covert|coverts|wood|woodland|grove)\b", 4),
            (r"\b(?:draw|drive|drove|driven)\b", 1),
            (r"\b(?:elector|chancellor|advocate|counsel|prosecutor|defender)\b", 3),
            (r"\b(?:hear|heard|listen|understand|mean|meant)\b", 1),
        )
        for pattern, weight in risk_rules:
            if re.search(pattern, text, re.I):
                score += weight
        return score
