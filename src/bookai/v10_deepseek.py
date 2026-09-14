from __future__ import annotations

import json
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import DeepSeekSemanticSpecialist as _BaseDeepSeekSemanticSpecialist
from .v10 import V10Issue, _json_from_text, _norm
from .v10_quantity import extract_quantity_obligations


PROVEN_DEEPSEEK_CODES = frozenset({
    "numeric",
    "quantity_obligation",
    "numbered_choice",
    "quarter_inch",
    "short_omission",
    "omission",
    "duplicate_content",
    "clause_order",
})
_QUANTITY_CODES = {"numeric", "quantity_obligation", "numbered_choice", "quarter_inch"}


def _semantic_hints_for_segment(segment: Segment, memory: BookMemory) -> dict[str, str]:
    """Return only source-profile hints whose exact English phrase occurs here."""
    source = str(segment.text or "")
    out: dict[str, str] = {}
    for phrase, meaning in dict(getattr(memory, "semantic_hints", {}) or {}).items():
        key = str(phrase or "").strip()
        value = str(meaning or "").strip()
        if not key or not value:
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])", source, re.I):
            out[key] = value
    return out


class DeepSeekSemanticSpecialist(_BaseDeepSeekSemanticSpecialist):
    """One-batch DeepSeek tail with proof-first and source-hint-aware routing."""

    def __init__(self, provider: Any, qa: Any, max_segments: int = 8):
        super().__init__(provider, qa, max_segments=max_segments)
        self.stats.update({
            "forced_proven_selected": 0,
            "forced_proven_ids": [],
            "forced_proven_accepted": 0,
            "forced_proven_rejected": 0,
            "forced_rejection_details": [],
            "semantic_hint_selected": 0,
            "semantic_hint_ids": [],
        })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for issue in issues:
            if issue.severity == "hard" and issue.code in PROVEN_DEEPSEEK_CODES:
                out.setdefault(issue.id, set()).add(issue.code)
        return out

    def _select(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[Segment]:
        proven = self._proven_codes_by_id(issues)
        semantic_hard = {issue.id for issue in issues if issue.severity == "hard" and issue.mode == "semantic"}
        hint_counts = {segment.id: len(_semantic_hints_for_segment(segment, memory)) for segment in targets}
        ranked = sorted(
            targets,
            key=lambda segment: (
                segment.id in proven,
                segment.id in semantic_hard,
                hint_counts.get(segment.id, 0) > 0,
                hint_counts.get(segment.id, 0),
                self.qa.semantic_risk(segment, memory),
                len(segment.text),
            ),
            reverse=True,
        )
        selected: list[Segment] = []
        for segment in ranked:
            risk = self.qa.semantic_risk(segment, memory)
            has_hints = bool(hint_counts.get(segment.id, 0))
            if segment.id not in proven and segment.id not in semantic_hard and not has_hints and risk < 4:
                continue
            selected.append(segment)
            if len(selected) >= self.max_segments:
                break
        return selected

    def repair(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[str]:
        selected = self._select(targets, translated, memory, issues)
        if not selected:
            return []

        proven = self._proven_codes_by_id(issues)
        selected_proven_ids = [segment.id for segment in selected if segment.id in proven]
        self.stats["forced_proven_selected"] = len(selected_proven_ids)
        self.stats["forced_proven_ids"] = selected_proven_ids
        hinted_ids = [segment.id for segment in selected if _semantic_hints_for_segment(segment, memory)]
        self.stats["semantic_hint_selected"] = len(hinted_ids)
        self.stats["semantic_hint_ids"] = hinted_ids

        index = {segment.id: i for i, segment in enumerate(targets)}
        issue_map: dict[str, list[str]] = {}
        for issue in issues:
            issue_map.setdefault(issue.id, []).append(f"{issue.code}: {issue.reason}")

        items = []
        for segment in selected:
            i = index[segment.id]
            must_fix = sorted(proven.get(segment.id, set()))
            obligations = []
            if set(must_fix) & _QUANTITY_CODES:
                obligations = [
                    {"kind": row.kind, "value": row.value, "source_phrase": row.source_phrase}
                    for row in extract_quantity_obligations(segment.text)
                ]
            items.append({
                "id": segment.id,
                "source": segment.text,
                "current_ru": translated.get(segment.id, ""),
                "known_defects": issue_map.get(segment.id, []),
                "must_fix_codes": must_fix,
                "source_semantic_hints": _semantic_hints_for_segment(segment, memory),
                "exact_quantity_obligations": obligations,
                "before_en": [row.text for row in targets[max(0, i - 2):i]],
                "after_en": [row.text for row in targets[i + 1:i + 3]],
            })

        self.stats.update({"selected": len(selected), "selected_ids": [segment.id for segment in selected]})
        style = memory.style
        system = """You are the ONE expensive semantic specialist in a cost-sensitive EN→RU BOOK translation pipeline.
The book may be literary fiction, narrative nonfiction, academic/technical prose, or another genre. Follow the supplied BOOK_PROFILE; never impose a literary voice on technical prose or flatten literary prose into textbook language.
All rows are handled in THIS ONE batch. Never request another pass.

Rows with must_fix_codes contain SOURCE-GROUNDED, DETERMINISTICALLY PROVEN fidelity failures. They are mandatory:
- numeric / quantity_obligation: restore every exact quantity AND its proposition. exact_quantity_obligations gives arithmetic truth. In particular two dozen=24, NOT «два десятка»; half a dozen=6; a dozen=12;
- numbered_choice: preserve the numbered option/label and its governing action;
- quarter_inch: preserve 1/4 inch or faithful 6.35 mm conversion;
- short_omission / omission: restore every missing source beat, clause, dialogue turn and action without invention;
- duplicate_content: remove target-only repeated clauses/phrases while preserving the source proposition once;
- clause_order: restore source discourse-clause order of reliable anchors, but keep natural Russian word order inside a clause.

`source_semantic_hints` are SOURCE-ONLY sense disambiguations produced before translation. Treat them as a compact semantic contract for the exact English phrases shown: verify that current_ru expresses that contextual meaning and repair a calque, category error, archaic misreading or idiom if it does not. Do not copy the English explanation into Russian and do not change already-natural wording merely to paraphrase it.

For other rows, repair only genuine semantic/publication defects: invented facts, actor/action/object reversals, antecedents, chronology, causality, negation/modality, category narrowing/broadening, difficult word sense or technical denotation. Do not rewrite merely for taste.
For academic/technical prose preserve established Russian terminology, formulas, notation, bibliography and ACRONYM_CANON. For fiction preserve authorial voice, rhythm, irony and character speech.
If a source fragment syntactically continues into before_en/after_en, do not close it artificially with a sentence boundary.
If change=true, corrected_ru MUST be the complete publication-ready Russian translation of exactly that source segment. Preserve every fact, number, name and established term. No English residue unless required by ACRONYM_CANON, a citation, formula, identifier or genuine proper name.
Return exactly one row per id.
ONLY JSON {"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}"""

        book_profile = {
            "domain": str(getattr(memory, "domain", "") or ""),
            "voice": str(style.narrative_voice or ""),
            "rhythm": str(style.rhythm or ""),
            "dialogue": str(style.dialogue or ""),
            "humor": str(style.humor or ""),
            "acronym_canon": dict(memory.acronyms),
        }
        try:
            raw = self.provider.complete(
                system,
                json.dumps({"book_profile": book_profile, "items": items}, ensure_ascii=False),
                temperature=0.0,
            )
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            print(f"[v10-deepseek] error={type(exc).__name__}: {exc}", flush=True)
            return []

        parsed = {str(row.get("id") or ""): row for row in (obj.get("items") or []) if isinstance(row, dict)}
        changed: list[str] = []
        by_id = {segment.id: segment for segment in selected}
        for sid, segment in by_id.items():
            row = parsed.get(sid) or {}
            if type(row.get("change")) is not bool or not row.get("change"):
                if sid in proven:
                    self.stats["forced_proven_rejected"] += 1
                    self.stats["forced_rejection_details"].append({"id": sid, "reason": "model_change_false"})
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            candidate = _norm(row.get("corrected_ru") or "")
            if confidence < 0.62 or not candidate:
                if sid in proven:
                    self.stats["forced_proven_rejected"] += 1
                    self.stats["forced_rejection_details"].append({"id": sid, "reason": "low_confidence_or_empty"})
                continue

            current = str(translated.get(sid) or "")
            before = self.qa.scan_segment(segment, current, memory)
            after = self.qa.scan_segment(segment, candidate, memory)
            before_hard = sum(issue.severity == "hard" for issue in before)
            after_hard = sum(issue.severity == "hard" for issue in after)

            required_codes = proven.get(sid, set())
            if required_codes:
                residual_required = {issue.code for issue in after if issue.severity == "hard" and issue.code in required_codes}
                if residual_required or after_hard > before_hard:
                    self.stats["forced_proven_rejected"] += 1
                    self.stats["forced_rejection_details"].append({
                        "id": sid,
                        "reason": "required_code_remains_or_new_hard",
                        "residual_required": sorted(residual_required),
                        "hard_before": before_hard,
                        "hard_after": after_hard,
                    })
                    continue
                self.stats["forced_proven_accepted"] += 1
            elif after_hard > before_hard:
                continue

            translated[sid] = candidate
            changed.append(sid)

        self.stats["accepted"] = len(changed)
        return changed
