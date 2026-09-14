from __future__ import annotations

import json
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _json_from_text, _norm
from .v10_deepseek import PROVEN_DEEPSEEK_CODES
from .v10_dialogue import DialogueDiscourseGuard, source_has_dialogue
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
            memory.glossary.pop(name, None)
        stats = dict(stats)
        stats["runtime_false_character_canon_pruned"] = sorted(pruned)
        return memory, stats


class FinalDialogueDiscourseGuard(DialogueDiscourseGuard):
    """Dialogue normalizer that preserves genuine nested Russian guillemets.

    The older guard correctly converted outer English direct-speech quotes to a
    Russian dialogue dash, but then stripped every closing » near punctuation.
    That destroyed valid nested quotes such as «это закон?» and created the last
    release-gate failures in otherwise repaired rows.
    """

    @staticmethod
    def _drop_unmatched_closing_guillemets(value: str) -> str:
        depth = 0
        out: list[str] = []
        for ch in value:
            if ch == "«":
                depth += 1
                out.append(ch)
            elif ch == "»":
                if depth > 0:
                    depth -= 1
                    out.append(ch)
                # An unmatched closing » is normally the obsolete outer quote
                # after its opening quote was converted to a dialogue dash.
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _normalize(segment: Segment, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        source = str(segment.text or "")
        value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r",\s*,+", ",", value)

        if source_has_dialogue(source):
            source_starts_dialogue = bool(re.match(r"^\s*[\"'“‘]", source))
            if source_starts_dialogue:
                if re.match(r"^\s*[\"'«]", value):
                    value = re.sub(r"^\s*[\"'«]\s*", "— ", value, count=1)
                elif not re.match(r"^\s*—", value):
                    value = "— " + value.lstrip()

            value = re.sub(r"—\s*[\"']\s*(?=[А-Яа-яЁё])", "— ", value)
            value = re.sub(r"(?<=[.!?…])\s*[\"']\s*(?=[А-ЯЁ])", " — ", value)

            # Remove only obsolete ASCII quote residue. Genuine paired «...» is
            # preserved; only unmatched closing guillemets are discarded below.
            value = re.sub(r"(?<=[А-Яа-яЁё0-9])['\"](?=[,!?….])", "", value)
            value = re.sub(r"([,!?….])\s*['\"](?=\s*(?:—|-|$))", r"\1", value)
            value = re.sub(r"['\"]\s*$", "", value)
            value = FinalDialogueDiscourseGuard._drop_unmatched_closing_guillemets(value)
        else:
            value = re.sub(r"['\"]([^'\"\n]{1,240})['\"]", r"«\1»", value)

        value = re.sub(r"\s+([,.!?…])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        value = re.sub(r",\s*,+", ",", value)
        return value


class FinalV10QualityQA(HardenedV10QualityQA):
    """Release QA with evidence-aware false-positive suppression."""

    def __init__(self, *, demote_clause_order: bool = False):
        super().__init__()
        self.demote_clause_order = bool(demote_clause_order)

    @staticmethod
    def _mixed_script_tokens(target: str) -> list[str]:
        tokens = HardenedV10QualityQA._mixed_script_tokens(target)
        # Standard Russian technical notation: V-образный, T-образный, X-образный.
        return [token for token in tokens if not re.fullmatch(r"[A-Za-z]-[А-Яа-яЁё-]+", token)]

    @classmethod
    def _clean_quantity(cls, source: str, target: str) -> dict[str, Any]:
        quantity = dict(super()._clean_quantity(source, target))
        low = str(target or "").casefold().replace("ё", "е")
        suppress: set[int] = set()

        for value, patterns in _HUNDRED_EVIDENCE.items():
            if any(re.search(pattern, low) for pattern in patterns):
                suppress.add(value)

        # Legacy numeric parser adds cardinal + ordinal here: one + tenth => 11.
        if re.search(r"\bone\s+tenth\b", source, re.I) and re.search(r"\bодн\w*\s+десят\w*\b", low):
            suppress.add(11)

        # Likewise one sixty-fourth is a fraction 1/64, not 65.
        if re.search(r"\bone\s+sixty[- ]fourth\b", source, re.I):
            fraction_ok = bool(
                re.search(r"\bодн\w*\s+шестьдесят\w*четверт\w*\b", low)
                or re.search(r"\b1\s*/\s*64\b", low)
                or re.search(r"\b0[,.]015625\b", low)
            )
            if fraction_ok:
                suppress.add(65)

        # Natural range: "another two, three hundred" -> "ещё двести-триста".
        if re.search(r"\btwo\s*,\s*three\s+hundred\b", source, re.I):
            if re.search(r"\bдвест\w*\b", low) and re.search(r"\bтрист\w*\b", low):
                suppress.update({2, 3, 200, 300})

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
            if "role=proper_name" in desc or name in _FALSE_CHARACTER_CANON:
                continue
            out.append(issue)
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan_segment(segment, target, memory))
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")

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
    """Route every release-blocking residual to the bounded semantic batch."""

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


class ResidualHardDeepSeekRepair:
    """One small fail-closed rescue call for rows a large release batch rejected.

    It is only invoked when deterministic HARD issues remain. Unlike the main
    specialist, it does not trust a model-reported confidence number; a candidate
    is accepted only when the deterministic release QA proves that HARD errors
    decrease. This handles occasional schema/confidence collapse without reopening
    the old many-call Giga repair cascade.
    """

    def __init__(self, provider: Any, qa: FinalV10QualityQA, max_segments: int = 12):
        self.provider = provider
        self.qa = qa
        self.max_segments = max(1, int(max_segments))
        self.stats: dict[str, Any] = {
            "calls": 0,
            "selected": 0,
            "selected_ids": [],
            "accepted": 0,
            "changed_ids": [],
            "rejected": [],
        }

    def repair(
        self,
        targets: list[Segment],
        translated: dict[str, str],
        memory: BookMemory,
        issues: list[V10Issue],
    ) -> list[str]:
        hard_by_id: dict[str, list[V10Issue]] = {}
        for issue in issues:
            if issue.severity == "hard":
                hard_by_id.setdefault(issue.id, []).append(issue)
        if not hard_by_id:
            return []

        selected = [segment for segment in targets if segment.id in hard_by_id][: self.max_segments]
        if not selected:
            return []
        self.stats["selected"] = len(selected)
        self.stats["selected_ids"] = [segment.id for segment in selected]

        items = []
        for segment in selected:
            items.append({
                "id": segment.id,
                "source_en": segment.text,
                "current_ru": translated.get(segment.id, ""),
                "hard_defects": [f"{row.code}: {row.reason}" for row in hard_by_id[segment.id]],
            })

        system = """Final fail-closed EN→RU literary repair. Every input row still fails a deterministic publication gate.
Return a COMPLETE corrected Russian translation for every id; never return English source unchanged.
Fix every listed hard_defect while preserving every source fact, quantity, speaker attribution, name and technical denotation.
If the current_ru is mostly English, translate the entire source_en from scratch. If a source character name appears in an
attribution such as 'Ziani said', preserve that attribution and the stated Russian canon from hard_defects. Keep Russian nested
quotes balanced («...»). Standard one-letter engineering notation such as V-образный is allowed. Do not add commentary.
ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}"""
        try:
            raw = self.provider.complete(system, json.dumps({"items": items}, ensure_ascii=False), temperature=0.0)
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            self.stats["rejected"].append({"reason": f"call_error:{type(exc).__name__}"})
            return []

        parsed = {str(row.get("id") or ""): row for row in (obj.get("items") or []) if isinstance(row, dict)}
        changed: list[str] = []
        by_id = {segment.id: segment for segment in selected}
        for sid, segment in by_id.items():
            candidate = _norm((parsed.get(sid) or {}).get("corrected_ru") or "")
            if not candidate:
                self.stats["rejected"].append({"id": sid, "reason": "empty_or_missing"})
                continue
            current = str(translated.get(sid) or "")
            before = self.qa.scan_segment(segment, current, memory)
            after = self.qa.scan_segment(segment, candidate, memory)
            before_hard = sum(row.severity == "hard" for row in before)
            after_hard = sum(row.severity == "hard" for row in after)
            if after_hard >= before_hard:
                self.stats["rejected"].append({
                    "id": sid,
                    "reason": "hard_not_reduced",
                    "before": before_hard,
                    "after": after_hard,
                })
                continue
            translated[sid] = candidate
            changed.append(sid)

        self.stats["accepted"] = len(changed)
        self.stats["changed_ids"] = changed
        return changed
