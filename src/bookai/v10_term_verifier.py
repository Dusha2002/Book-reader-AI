from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import _json_from_text, _norm
from .v10_publication_release import _priority_term_candidates, _technical_style
from .v10_source_bible import _has_clean_russian


_VERIFIER_CACHE_MARKER = "v10-deepseek-terminology-1"
_LOCAL_DEFINITION_RE = re.compile(
    r"\b([a-z][a-z'-]{2,16})\b\s+(?:had|has|have|is|was|were)\s+"
    r"(?:no|not|only|a|an|the)\b",
    re.I,
)
_COMMON_DEFINITION_WORDS = {
    "thing", "person", "people", "time", "place", "part", "side", "way", "problem", "result",
    "model", "system", "method", "work", "research", "network", "algorithm",
}


def _local_definition_candidates(segments: list[Segment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in segments:
        text = str(segment.text or "")
        for match in _LOCAL_DEFINITION_RE.finditer(text):
            source = match.group(1).strip()
            key = source.casefold()
            if key in seen or key in _COMMON_DEFINITION_WORDS:
                continue
            seen.add(key)
            rows.append({
                "source": source,
                "current_ru": "",
                "evidence": "local_source_definition",
                "contexts": [_norm(text)[:700]],
            })
    return rows


def _verification_candidates(segments: list[Segment], memory: BookMemory) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _priority_term_candidates(segments, memory)]
    seen = {str(row.get("source") or "").casefold() for row in rows}
    for row in _local_definition_candidates(segments):
        key = str(row.get("source") or "").casefold()
        if key in seen:
            continue
        current = str(memory.glossary.get(key) or "")
        row["current_ru"] = current
        rows.append(row)
        seen.add(key)

    # A nontechnical book only needs the explicit source-defined specialist rows;
    # technical/academic books benefit from the whole compact priority set.
    if not _technical_style(memory):
        rows = [
            row for row in rows
            if str(row.get("evidence") or "").startswith("explicit_acronym_definition")
            or row.get("evidence") == "local_source_definition"
        ]
    return rows[:32]


class DeepSeekPublicationTerminologyVerifier:
    """One source-only semantic pass over a compact set of high-risk book terms.

    This is intentionally not a sentence retranslator. It creates/repairs runtime
    terminology canon before the primary translation, then caches the verified canon
    with the book bible so later chapters do not pay for the pass again.
    """

    def __init__(self, provider: Any, cache_path: Path):
        self.provider = provider
        self.cache_path = Path(cache_path)
        self.stats: dict[str, Any] = {
            "cache_hit": False,
            "calls": 0,
            "candidates": 0,
            "returned": 0,
            "added": 0,
            "refined": 0,
            "kept": 0,
            "omitted": 0,
            "applied": {},
        }

    def _load_cache(self, memory: BookMemory) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict) or data.get("deepseek_terminology_schema") != _VERIFIER_CACHE_MARKER:
            return False
        verified = data.get("deepseek_verified_glossary") or {}
        if not isinstance(verified, dict):
            return False
        for source, ru in verified.items():
            if source and ru:
                memory.glossary[str(source)] = str(ru)
        self.stats["cache_hit"] = True
        self.stats["applied"] = {str(k): str(v) for k, v in verified.items() if k and v}
        return True

    def _save_cache(self, memory: BookMemory, applied: dict[str, str]) -> None:
        try:
            data = json.loads(self.cache_path.read_text("utf-8")) if self.cache_path.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["glossary"] = dict(memory.glossary)
        data["deepseek_verified_glossary"] = dict(applied)
        data["deepseek_terminology_schema"] = _VERIFIER_CACHE_MARKER
        try:
            self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass

    def verify(self, segments: list[Segment], memory: BookMemory) -> dict[str, Any]:
        if self._load_cache(memory):
            return dict(self.stats)

        candidates = _verification_candidates(segments, memory)
        self.stats["candidates"] = len(candidates)
        if not candidates:
            self._save_cache(memory, {})
            return dict(self.stats)

        style = memory.style
        system = """You are the final SOURCE-ONLY terminology verifier for an EN→RU book translation pipeline.
You are NOT translating sentences. You receive a small set of terms selected because the English source itself gives evidence that they are specialist/high-risk.
For EVERY candidate return one decision: keep, change, add, or omit.

Rules:
- Use the established professional Russian term appropriate to the supplied source context, not a word-for-word calque.
- Preserve semantic category breadth exactly. A broad family/class in English must NOT become one subtype, implementation or example in Russian.
- For a term explicitly defined with an acronym, choose the conventional Russian term used with that acronym while preserving every semantic component.
- For an archaic/specialist object defined by its properties in literary prose, identify the actual object term rather than transliterating an ordinary-looking English word.
- `keep` means current_ru is already professionally acceptable and semantically exact.
- `change` means current_ru exists but should be replaced. `add` means no usable current_ru exists. `omit` means the candidate is ordinary prose, a proper name, or too ambiguous to canonicalize safely.
- `ru` must contain only the concise Russian canonical term, without explanations, alternatives, parentheses or translator notes.
- Never use outside plot facts and never infer a book-specific spelling not supported by the supplied source contexts.

Return ONLY JSON {"items":[{"source":"exact supplied source","decision":"keep|change|add|omit","ru":"...","confidence":0.0,"reason":"brief"}]} with exactly one row per supplied candidate."""
        payload = {
            "book_profile": {
                "voice": str(style.narrative_voice or ""),
                "rhythm": str(style.rhythm or ""),
                "dialogue": str(style.dialogue or ""),
                "humor": str(style.humor or ""),
            },
            "candidates": candidates,
        }
        try:
            raw = self.provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0)
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            print(f"[v10-term-verifier] error={type(exc).__name__}: {exc}", flush=True)
            return dict(self.stats)

        allowed = {str(row.get("source") or "").casefold(): row for row in candidates}
        applied: dict[str, str] = {}
        rows = [row for row in obj.get("items") or [] if isinstance(row, dict)]
        self.stats["returned"] = len(rows)
        for item in rows:
            source = str(item.get("source") or "").strip()
            key = source.casefold()
            if key not in allowed:
                continue
            decision = str(item.get("decision") or "").strip().casefold()
            ru = _norm(item.get("ru") or "")
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if decision == "keep":
                self.stats["kept"] += 1
                continue
            if decision == "omit":
                self.stats["omitted"] += 1
                continue
            if decision not in {"change", "add"} or confidence < 0.76:
                continue
            if not ru or not _has_clean_russian(ru) or len(ru.split()) > 9:
                continue
            if any(ch in ru for ch in "()[]{}"):
                continue

            old = str(memory.glossary.get(key) or "").strip()
            if old:
                if old.casefold() == ru.casefold():
                    self.stats["kept"] += 1
                    continue
                memory.glossary[key] = ru
                self.stats["refined"] += 1
            else:
                memory.glossary[key] = ru
                self.stats["added"] += 1
            applied[key] = ru

        self.stats["applied"] = dict(applied)
        self._save_cache(memory, applied)
        print("[v10-term-verifier] " + json.dumps(self.stats, ensure_ascii=False), flush=True)
        return dict(self.stats)


__all__ = ["DeepSeekPublicationTerminologyVerifier"]