from __future__ import annotations

import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue
from .v10_deepseek import PROVEN_DEEPSEEK_CODES
from .v10_release import (
    HardenedBookBibleBuilder,
    HardenedDeepSeekSemanticSpecialist,
    HardenedV10QualityQA,
)


_FALSE_CHARACTER_CANON = {"Who", "God", "Duchess"}
_HUNDRED_EVIDENCE: dict[int, tuple[str, ...]] = {
    100: (r"\bсто\b", r"\bсотн\w*\b", r"\bстопроцент\w*\b"),
    200: (r"\bдвест\w*\b",),
    300: (r"\bтрист\w*\b",),
    400: (r"\bчетырест\w*\b",),
    500: (r"\bпятьсот\b", r"\bпятисот\b"),
    600: (r"\bшестьсот\b", r"\bшестисот\b"),
    700: (r"\bсемьсот\b", r"\bсемисот\b"),
    800: (r"\bвосемьсот\b", r"\bвосьмисот\b"),
    900: (r"\bдевятьсот\b", r"\bдевятисот\b"),
}


class FinalBookBibleBuilder(HardenedBookBibleBuilder):
    """Keep useful source-only canon, but do not expose obvious common-word false names as characters."""

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        memory, stats = super().build(segments)
        pruned: list[str] = []
        for name in _FALSE_CHARACTER_CANON:
            if name in memory.characters:
                memory.characters.pop(name, None)
                pruned.append(name)
            # These entries are harmful as forced glossary items too; ordinary Russian handles them better.
            memory.glossary.pop(name, None)
        stats = dict(stats)
        stats["runtime_false_character_canon_pruned"] = sorted(pruned)
        return memory, stats


class FinalV10QualityQA(HardenedV10QualityQA):
    """Release QA with evidence-aware false-positive suppression.

    Hard semantic checks stay strict. Only diagnostics proven noisy on valid Russian
    morphology/phrasing are relaxed, and clause-order can be demoted after one repair pass.
    """

    def __init__(self, *, demote_clause_order: bool = False):
        super().__init__()
        self.demote_clause_order = bool(demote_clause_order)

    @classmethod
    def _clean_quantity(cls, source: str, target: str) -> dict[str, Any]:
        quantity = dict(super()._clean_quantity(source, target))
        low = str(target or "").casefold().replace("ё", "е")
        suppress: set[int] = set()

        for value, patterns in _HUNDRED_EVIDENCE.items():
            if any(re.search(pattern, low) for pattern in patterns):
                suppress.add(value)

        # numeric_fidelity currently tokenises "one tenth" as the spurious sum 11.
        if re.search(r"\bone\s+tenth\b", source, re.I) and re.search(r"\bодн\w*\s+десят\w*\b", low):
            suppress.add(11)

        # Natural range: "another two, three hundred" -> "ещё двести-триста".
        if re.search(r"\btwo\s*,\s*three\s+hundred\b", source, re.I):
            if re.search(r"\bдвест\w*\b", low) and re.search(r"\bтрист\w*\b", low):
                suppress.update({200, 300})

        base_missing = [value for value in quantity.get("base_missing") or [] if value not in suppress]
        missing_mentions = [
            row for row in quantity.get("missing_mentions") or []
            if row.get("value") not in suppress
        ]
        quantity["base_missing"] = base_missing
        quantity["missing_mentions"] = missing_mentions
        quantity["ok"] = not base_missing and not missing_mentions and not quantity.get("numbered_choice_missing")
        return quantity

    @classmethod
    def _name_canon_issues(cls, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = super()._name_canon_issues(segment, target, memory)
        out: list[V10Issue] = []
        for issue in issues:
            match = re.search(r"source name '([^']+)'", issue.reason)
            name = match.group(1) if match else ""
            desc = str(memory.characters.get(name) or "")
            # v10_source_bible inserts every accepted proper token into characters with this fallback role.
            # Enforce character spelling only when the source analysis actually classified it as a character.
            if "role=proper_name" in desc or name in _FALSE_CHARACTER_CANON:
                continue
            out.append(issue)
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan_segment(segment, target, memory))
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")

        # Correct Russian compact form: "полдюйма / полдюймовый".
        if re.search(r"\b(?:half[- ]inch|half an inch)\b", source, re.I) and re.search(r"\bполдюйм\w*\b", low):
            issues = [issue for issue in issues if issue.code != "half_inch"]

        if self.demote_clause_order:
            demoted: list[V10Issue] = []
            for issue in issues:
                if issue.code == "clause_order" and issue.severity == "hard":
                    demoted.append(V10Issue(issue.id, issue.code, issue.mode, "soft", issue.reason))
                else:
                    demoted.append(issue)
            issues = demoted

        unique = {(row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


class FinalDeepSeekSemanticSpecialist(HardenedDeepSeekSemanticSpecialist):
    """Route every release-blocking residual to the one bounded semantic batch."""

    _FINAL_PROVEN = frozenset({
        "latin_leak",
        "question",
        "direction_relation",
        "half_inch",
        "material",
        "name_canon",
        "dialogue_typography",
        "technical_denotation",
        "ethnonym_canon",
        "idiom_last_but_one",
        "hunting_collocation",
    })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        allowed = PROVEN_DEEPSEEK_CODES | HardenedDeepSeekSemanticSpecialist._EXTRA_PROVEN | FinalDeepSeekSemanticSpecialist._FINAL_PROVEN
        out: dict[str, set[str]] = {}
        for issue in issues:
            if issue.severity == "hard" and issue.code in allowed:
                out.setdefault(issue.id, set()).add(issue.code)
        return out
