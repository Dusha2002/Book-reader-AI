from __future__ import annotations

import re

from .models import BookMemory, Segment
from .v10 import DeterministicQA, V10Issue
from .v10_numeric import compare_numeric_fidelity_v10


class V10QualityQA(DeterministicQA):
    """Clean-v10 QA with risk-specific priorities learned from v9ad.

    Deterministic checks remain narrow; broader lexical/spatial/kinship signals only
    influence which <=8 rows reach the single DeepSeek semantic batch.
    """

    @staticmethod
    def _mixed_script_tokens(target: str) -> list[str]:
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]*", str(target or ""))
        return [
            token for token in tokens
            if re.search(r"[A-Za-z]", token) and re.search(r"[А-Яа-яЁё]", token)
        ]

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        out = list(super().scan_segment(segment, target, memory))
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")

        # The legacy numeric detector intentionally remains untouched. v10 only
        # suppresses its known Russian-morphology false positives when an equivalent
        # source number is demonstrably present in the target.
        numeric_v10 = compare_numeric_fidelity_v10(source, target)
        if numeric_v10.get("ok", True):
            out = [row for row in out if row.code != "numeric"]
        else:
            out = [row for row in out if row.code != "numeric"]
            out.append(V10Issue(
                segment.id,
                "numeric",
                "local",
                "hard",
                f"missing source numeric values {numeric_v10.get('missing')}",
            ))

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

        # Hunting vocabulary is highly polysemous. Route only concrete suspicious
        # realizations, not every occurrence of the words, to the semantic specialist.
        if re.search(r"\bdraw\s+the\s+(?:home\s+)?coverts\b", source, re.I):
            if re.search(r"\b(?:домашн\w*\s+птиц\w*|птиц\w*)\b", low):
                out.append(V10Issue(
                    segment.id,
                    "hunting_collocation",
                    "semantic",
                    "hard",
                    "hunting 'draw the coverts' was rendered as birds/domestic animals instead of working through cover",
                ))
        if re.search(r"\bmill-stream\b", source, re.I) and re.search(r"\bзапруд\w*\b", low):
            out.append(V10Issue(
                segment.id,
                "hunting_collocation",
                "semantic",
                "hard",
                "mill-stream is a stream/watercourse in the hunt, not a dam",
            ))

        # Base latin_leak catches standalone Latin tokens but not mixed-script words
        # such as Ветраниio. Treat those as a local publication defect.
        mixed = self._mixed_script_tokens(target)
        if mixed:
            out.append(V10Issue(
                segment.id,
                "latin_leak",
                "local",
                "hard",
                "mixed Cyrillic/Latin token remains: " + ", ".join(mixed[:4]),
            ))

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
            (r"\bdraw\s+the\s+(?:home\s+)?coverts\b", 9),
            (r"\bmill-stream\b", 5),
            (r"\b(?:hunt|hunting|boar|spinney|covert|coverts|wood|woodland|grove)\b", 4),
            (r"\b(?:draw|drive|drove|driven)\b", 1),
            (r"\b(?:elector|chancellor|advocate|counsel|prosecutor|defender)\b", 3),
            (r"\b(?:hear|heard|listen|understand|mean|meant)\b", 1),
        )
        for pattern, weight in risk_rules:
            if re.search(pattern, text, re.I):
                score += weight
        return score
