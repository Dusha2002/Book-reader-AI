from __future__ import annotations

import json
from typing import Any

from .models import BookMemory, Segment
from .v10 import DeepSeekSemanticSpecialist as _BaseDeepSeekSemanticSpecialist
from .v10 import V10Issue, _json_from_text, _norm


# These defects are not heuristic "maybe bad" signals. They are deterministic,
# source-grounded fidelity failures that survived both cheap Giga repair tiers.
# They therefore get first claim on the existing ONE DeepSeek batch rather than
# causing another Giga loop or an extra DeepSeek request.
PROVEN_DEEPSEEK_CODES = frozenset({
    "numeric",
    "quantity_obligation",
    "numbered_choice",
    "quarter_inch",
    "short_omission",
    "omission",
})


class DeepSeekSemanticSpecialist(_BaseDeepSeekSemanticSpecialist):
    """One-batch DeepSeek tail with proof-first routing.

    Selection priority:
      1. deterministic hard quantity/omission defects that survived Giga;
      2. other deterministic hard semantic defects;
      3. ordinary v9ad-style semantic-risk candidates.

    No second DeepSeek call is ever created. For proof-routed rows, a candidate is
    accepted only if every proven defect code present before the call disappears.
    """

    def __init__(self, provider: Any, qa: Any, max_segments: int = 8):
        super().__init__(provider, qa, max_segments=max_segments)
        self.stats.update({
            "forced_proven_selected": 0,
            "forced_proven_ids": [],
            "forced_proven_accepted": 0,
            "forced_proven_rejected": 0,
        })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for issue in issues:
            if issue.severity == "hard" and issue.code in PROVEN_DEEPSEEK_CODES:
                out.setdefault(issue.id, set()).add(issue.code)
        return out

    def _select(
        self,
        targets: list[Segment],
        translated: dict[str, str],
        memory: BookMemory,
        issues: list[V10Issue],
    ) -> list[Segment]:
        proven = self._proven_codes_by_id(issues)
        semantic_hard = {
            issue.id
            for issue in issues
            if issue.severity == "hard" and issue.mode == "semantic"
        }
        ranked = sorted(
            targets,
            key=lambda segment: (
                segment.id in proven,
                segment.id in semantic_hard,
                self.qa.semantic_risk(segment, memory),
                len(segment.text),
            ),
            reverse=True,
        )

        selected: list[Segment] = []
        for segment in ranked:
            risk = self.qa.semantic_risk(segment, memory)
            if segment.id not in proven and segment.id not in semantic_hard and risk < 4:
                continue
            selected.append(segment)
            if len(selected) >= self.max_segments:
                break
        return selected

    def repair(
        self,
        targets: list[Segment],
        translated: dict[str, str],
        memory: BookMemory,
        issues: list[V10Issue],
    ) -> list[str]:
        selected = self._select(targets, translated, memory, issues)
        if not selected:
            return []

        proven = self._proven_codes_by_id(issues)
        selected_proven_ids = [segment.id for segment in selected if segment.id in proven]
        self.stats["forced_proven_selected"] = len(selected_proven_ids)
        self.stats["forced_proven_ids"] = selected_proven_ids

        index = {segment.id: i for i, segment in enumerate(targets)}
        issue_map: dict[str, list[str]] = {}
        for issue in issues:
            issue_map.setdefault(issue.id, []).append(f"{issue.code}: {issue.reason}")

        items = []
        for segment in selected:
            i = index[segment.id]
            must_fix = sorted(proven.get(segment.id, set()))
            items.append({
                "id": segment.id,
                "source": segment.text,
                "current_ru": translated.get(segment.id, ""),
                "known_defects": issue_map.get(segment.id, []),
                "must_fix_codes": must_fix,
                "before_en": [row.text for row in targets[max(0, i - 2):i]],
                "after_en": [row.text for row in targets[i + 1:i + 3]],
            })

        self.stats.update({
            "selected": len(selected),
            "selected_ids": [segment.id for segment in selected],
        })
        system = """You are the ONE expensive semantic specialist in a cost-sensitive EN→RU literary pipeline.
All rows are handled in THIS ONE batch. Never request another pass.

Rows with must_fix_codes contain SOURCE-GROUNDED, DETERMINISTICALLY PROVEN fidelity failures that already survived cheap Giga repair. They are mandatory, not stylistic suggestions:
- numeric / quantity_obligation: restore every exact quantity and the proposition attached to it;
- numbered_choice: preserve the numbered option/label and its surrounding action (for example, "number six" must not disappear);
- quarter_inch: preserve the exact fraction/unit (quarter inch = 1/4 inch, or 6.35 mm if converted);
- short_omission / omission: restore every missing source beat, clause, dialogue turn and action without inventing text.
For those rows, change=true is expected unless current_ru already demonstrably contains the required meaning.

For other rows, repair only genuine semantic/publication defects: invented facts, actor/action/object reversals, antecedents, chronology, causality, negation/modality, difficult word sense or technical denotation. Do not rewrite merely for taste.
If change=true, corrected_ru MUST be the complete publication-ready Russian translation of exactly that source segment. Preserve every fact, number, name and established term.
Return exactly one row per id.
ONLY JSON {"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}"""

        try:
            raw = self.provider.complete(
                system,
                json.dumps({"items": items}, ensure_ascii=False),
                temperature=0.0,
            )
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            print(f"[v10-deepseek] error={type(exc).__name__}: {exc}", flush=True)
            return []

        parsed = {
            str(row.get("id") or ""): row
            for row in (obj.get("items") or [])
            if isinstance(row, dict)
        }
        changed: list[str] = []
        by_id = {segment.id: segment for segment in selected}
        for sid, segment in by_id.items():
            row = parsed.get(sid) or {}
            if type(row.get("change")) is not bool or not row.get("change"):
                if sid in proven:
                    self.stats["forced_proven_rejected"] += 1
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            candidate = _norm(row.get("corrected_ru") or "")
            if confidence < 0.62 or not candidate:
                if sid in proven:
                    self.stats["forced_proven_rejected"] += 1
                continue

            current = str(translated.get(sid) or "")
            before = self.qa.scan_segment(segment, current, memory)
            after = self.qa.scan_segment(segment, candidate, memory)
            before_hard = sum(issue.severity == "hard" for issue in before)
            after_hard = sum(issue.severity == "hard" for issue in after)

            required_codes = proven.get(sid, set())
            if required_codes:
                residual_required = {
                    issue.code
                    for issue in after
                    if issue.severity == "hard" and issue.code in required_codes
                }
                # Proof-routed rows must actually fix every defect that justified
                # spending a DeepSeek slot; a stylistic rewrite is not enough.
                if residual_required or after_hard > before_hard:
                    self.stats["forced_proven_rejected"] += 1
                    continue
                self.stats["forced_proven_accepted"] += 1
            elif after_hard > before_hard:
                continue

            translated[sid] = candidate
            changed.append(sid)

        self.stats["accepted"] = len(changed)
        return changed
