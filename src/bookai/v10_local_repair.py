from __future__ import annotations

from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json, _norm


class GigaLocalRewriter:
    """Second cheap repair tier for deterministic defects span-patching cannot fix."""

    _CODES = {
        "numeric", "quantity_obligation", "numbered_choice", "quarter_inch", "question",
        "material", "order", "latin_leak", "character_gender", "glossary_term",
    }

    def __init__(self, backend: Any, qa: Any, max_segments: int = 16) -> None:
        self.backend = backend
        self.qa = qa
        self.max_segments = max(1, max_segments)
        self.stats: dict[str, Any] = {"calls": 0, "requested": 0, "accepted": 0, "rejected": 0, "selected_ids": []}

    def repair(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[str]:
        by_id = {s.id: s for s in targets}
        local_by_id: dict[str, list[V10Issue]] = {}
        for issue in issues:
            if issue.mode == "local" and issue.code in self._CODES:
                local_by_id.setdefault(issue.id, []).append(issue)
        rows = []
        for sid, defects in local_by_id.items():
            segment = by_id.get(sid)
            current = str(translated.get(sid) or "")
            if not segment or not current:
                continue
            rows.append({
                "id": sid,
                "source": segment.text,
                "current_ru": current,
                "defects": [{"code": d.code, "reason": d.reason} for d in defects],
            })
            if len(rows) >= self.max_segments:
                break
        if not rows:
            return []

        self.stats["requested"] = len(rows)
        self.stats["selected_ids"] = [row["id"] for row in rows]
        system = """You are a FAST GigaChat EN→RU local fidelity editor. Every item has a PROVEN local defect.
Fix ONLY the listed defect(s) while preserving the current Russian wording, literary tone, paragraph structure and all unrelated facts.
Typical defects: missing/wrong number or unit, repeated/dozen quantity, numbered choice/label, question force/punctuation, physical material/order, raw untranslated Latin, local gender agreement.
For NUMERIC/QUANTITY defects, restore the COMPLETE proposition attached to every missing quantity. Do not satisfy a later quantity merely because the same number appears earlier in the paragraph.
Interpret dozen exactly: a dozen=12, half a dozen=6, two dozen=24. For a numbered choice such as "number six", preserve the choice naturally in Russian (e.g. «номер шесть» or «шестой вариант/кандидат») together with its surrounding clause.
For a LATIN defect, remove mixed-script/transliterated residue without changing the referent.
Do not add interpretations and do not perform broad stylistic rewriting. corrected_ru MUST be the COMPLETE final Russian translation of exactly source.
Return every supplied id. ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}.
"""
        changed: list[str] = []
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            try:
                obj = _giga_json(self.backend, system, {"items": batch}, max_tokens=5000)
                self.stats["calls"] += 1
            except Exception as exc:
                print(f"[v10-local-rewriter] batch={start // 8 + 1} error={type(exc).__name__}", flush=True)
                continue
            parsed = {str(item.get("id") or ""): item for item in (obj.get("items") or []) if isinstance(item, dict)}
            for row in batch:
                sid = row["id"]
                item = parsed.get(sid) or {}
                candidate = _norm(item.get("corrected_ru") or "")
                current = str(translated.get(sid) or "")
                if not candidate or candidate == current or "<s " in candidate or "<src " in candidate:
                    self.stats["rejected"] += 1
                    continue
                segment = by_id[sid]
                before = self.qa.scan_segment(segment, current, memory)
                after = self.qa.scan_segment(segment, candidate, memory)
                before_local = sum(i.mode == "local" and i.severity == "hard" for i in before)
                after_local = sum(i.mode == "local" and i.severity == "hard" for i in after)
                before_sem = sum(i.mode == "semantic" and i.severity == "hard" for i in before)
                after_sem = sum(i.mode == "semantic" and i.severity == "hard" for i in after)
                if after_local < before_local and after_sem <= before_sem:
                    translated[sid] = candidate
                    changed.append(sid)
                    self.stats["accepted"] += 1
                else:
                    self.stats["rejected"] += 1
        return changed
