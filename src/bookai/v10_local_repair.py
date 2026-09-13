from __future__ import annotations

import json
import re
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json, _norm, _usage


_EDITORIAL_RESIDUE_RE = re.compile(
    r"(?:\(\s*(?:вариант|букв\.?|буквально|дословно|option|variant|translation)\s*\)|"
    r"\[\s*(?:вариант|букв\.?|буквально|дословно|option|variant|translation)\s*\])",
    re.I,
)


class GigaLocalRewriter:
    """Second cheap repair tier for deterministic defects span-patching cannot fix."""

    _CODES = {
        "numeric", "quantity_obligation", "numbered_choice", "quarter_inch", "question",
        "material", "order", "latin_leak", "character_gender", "glossary_term",
    }
    _SINGLE_FALLBACK_CODES = {
        "numeric", "quantity_obligation", "numbered_choice", "quarter_inch", "latin_leak",
    }

    def __init__(self, backend: Any, qa: Any, max_segments: int = 16) -> None:
        self.backend = backend
        self.qa = qa
        self.max_segments = max(1, max_segments)
        self.stats: dict[str, Any] = {
            "calls": 0, "requested": 0, "accepted": 0, "rejected": 0,
            "missing_rows": 0, "single_calls": 0, "single_accepted": 0,
            "selected_ids": [], "single_ids": [], "editorial_residue_rejected": 0,
        }

    def _accept(self, segment: Segment, current: str, candidate: str, memory: BookMemory) -> bool:
        if not candidate or candidate == current or "<s " in candidate or "<src " in candidate:
            return False
        # Local repair is final prose, never an editor's note or a list of options.
        # This specifically prevents a bare gloss such as «шестой» (вариант) from
        # fooling the numbered-choice detector while the governing action is absent.
        if _EDITORIAL_RESIDUE_RE.search(candidate):
            self.stats["editorial_residue_rejected"] += 1
            return False
        before = self.qa.scan_segment(segment, current, memory)
        after = self.qa.scan_segment(segment, candidate, memory)
        before_local = sum(i.mode == "local" and i.severity == "hard" for i in before)
        after_local = sum(i.mode == "local" and i.severity == "hard" for i in after)
        before_sem = sum(i.mode == "semantic" and i.severity == "hard" for i in before)
        after_sem = sum(i.mode == "semantic" and i.severity == "hard" for i in after)
        return after_local < before_local and after_sem <= before_sem

    def _single_repair(self, row: dict[str, Any]) -> str:
        client = self.backend._ensure_client()
        system = (
            "Ты точный редактор литературного перевода EN→RU. Дана ОДНА строка с уже доказанными локальными дефектами. "
            "Исправь только их, но верни ПОЛНЫЙ готовый русский перевод SOURCE. Ничего не сокращай и не добавляй. "
            "Для dozen: a dozen=12, half a dozen=6, two dozen=24. Для number six сохрани сам выбор №6 и весь связанный смысл. "
            "Не добавляй скобочные пояснения, пометы 'вариант', альтернативы или комментарии переводчика. "
            "Если дефект Latin — убери латиницу, сохранив имя/значение. Верни только русский текст, без JSON, комментариев и вариантов."
        )
        request = {
            "model": self.backend.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(row, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": max(1400, min(6000, int(self.backend.max_tokens))),
        }
        self.stats["single_calls"] += 1
        try:
            response = client.chat(request)
            self.backend.usage.add(_usage(response), calls=1)
        except Exception as exc:
            print(f"[v10-local-single] id={row['id']} error={type(exc).__name__}", flush=True)
            return ""
        text = str(response.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^\s*(?:перевод|исправленный перевод|translation)\s*:\s*", "", text, flags=re.I)
        return _norm(text)

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
Interpret dozen exactly: a dozen=12, half a dozen=6, two dozen=24. For a numbered choice such as "number six", preserve the choice naturally in Russian together with its governing action and surrounding clause.
For a LATIN defect, remove mixed-script/transliterated residue without changing the referent.
Never append explanations, alternatives, bracketed glosses, translator notes or words such as «вариант» merely to satisfy a detector.
Do not add interpretations and do not perform broad stylistic rewriting. corrected_ru MUST be the COMPLETE final Russian translation of exactly source.
Return every supplied id. ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}.
"""
        changed: list[str] = []
        batch_size = 4
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            try:
                obj = _giga_json(self.backend, system, {"items": batch}, max_tokens=4600)
                self.stats["calls"] += 1
            except Exception as exc:
                print(f"[v10-local-rewriter] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                self.stats["rejected"] += len(batch)
                continue
            parsed = {str(item.get("id") or ""): item for item in (obj.get("items") or []) if isinstance(item, dict)}
            for row in batch:
                sid = row["id"]
                if sid not in parsed:
                    self.stats["missing_rows"] += 1
                    self.stats["rejected"] += 1
                    continue
                candidate = _norm((parsed[sid] or {}).get("corrected_ru") or "")
                current = str(translated.get(sid) or "")
                segment = by_id[sid]
                if self._accept(segment, current, candidate, memory):
                    translated[sid] = candidate
                    changed.append(sid)
                    self.stats["accepted"] += 1
                else:
                    self.stats["rejected"] += 1

        fallback_rows: list[dict[str, Any]] = []
        for row in rows:
            sid = row["id"]
            segment = by_id[sid]
            residual = self.qa.scan_segment(segment, translated.get(sid, ""), memory)
            if any(i.mode == "local" and i.severity == "hard" and i.code in self._SINGLE_FALLBACK_CODES for i in residual):
                fallback_rows.append({
                    "id": sid,
                    "source": segment.text,
                    "current_ru": translated.get(sid, ""),
                    "defects": [{"code": i.code, "reason": i.reason} for i in residual if i.mode == "local" and i.severity == "hard"],
                })
        for row in fallback_rows[:6]:
            sid = row["id"]
            self.stats["single_ids"].append(sid)
            segment = by_id[sid]
            current = str(translated.get(sid) or "")
            candidate = self._single_repair(row)
            if self._accept(segment, current, candidate, memory):
                translated[sid] = candidate
                if sid not in changed:
                    changed.append(sid)
                self.stats["accepted"] += 1
                self.stats["single_accepted"] += 1
            else:
                self.stats["rejected"] += 1
        return changed
