from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import _giga_json, _json_from_text, _norm
from .v10_publication_release import _priority_term_candidates, _technical_style
from .v10_source_bible import _has_clean_russian


_VERIFIER_CACHE_MARKER = "v10-deepseek-terminology-3"
_LOCAL_DEFINITION_RE = re.compile(
    r"\b([a-z][a-z'-]{2,16})\b\s+(?:had|has|have|is|was|were)\s+"
    r"(?:no|not|only)\b",
    re.I,
)
_COMMON_DEFINITION_WORDS = {
    "thing", "person", "people", "time", "place", "part", "side", "way", "problem", "result",
    "model", "system", "method", "work", "research", "network", "algorithm",
}


def _is_academic_domain(memory: BookMemory) -> bool:
    domain = str(getattr(memory, "domain", "") or "").strip().casefold()
    if domain:
        return domain == "academic_technical"
    # Legacy caches may not yet carry the explicit domain field.
    return _technical_style(memory)


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

    # Literary/nontechnical books only need terms explicitly defined by source
    # context. This prevents proper names from being mistaken for terminology merely
    # because the style description contains words such as “technical precision”.
    if not _is_academic_domain(memory):
        rows = [
            row for row in rows
            if str(row.get("evidence") or "").startswith("explicit_acronym_definition")
            or row.get("evidence") == "local_source_definition"
        ]
    return rows[:32]


def _valid_ru_term(value: str) -> bool:
    ru = _norm(value)
    return bool(
        ru
        and _has_clean_russian(ru)
        and len(ru.split()) <= 9
        and not any(ch in ru for ch in "()[]{}")
    )


class DeepSeekPublicationTerminologyVerifier:
    """Two tiny source-only semantic passes over a compact set of high-risk terms.

    DeepSeek creates a draft canon. An optional stronger GigaChat critic challenges
    only those few proposals; if that critic is unavailable, the same DeepSeek
    provider performs the critique. The verified canon is cached per book.
    """

    def __init__(self, provider: Any, cache_path: Path, *, critic_backend: Any | None = None):
        self.provider = provider
        self.cache_path = Path(cache_path)
        self.critic_backend = critic_backend
        self.stats: dict[str, Any] = {
            "cache_hit": False,
            "calls": 0,
            "candidates": 0,
            "returned": 0,
            "draft_proposals": 0,
            "critic_backend": "",
            "critic_fallback": False,
            "critic_returned": 0,
            "critic_revised": 0,
            "critic_omitted": 0,
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

    def _draft(self, candidates: list[dict[str, Any]], memory: BookMemory) -> list[dict[str, Any]]:
        style = memory.style
        system = """You are a SOURCE-ONLY terminology verifier for an EN→RU book translation pipeline.
You are NOT translating sentences. Every candidate was selected because the English source itself gives evidence that it may be specialist or high-risk.
For EVERY candidate return one decision: keep, change, add, or omit.

Rules:
- Choose the established professional Russian term used in the relevant field, not a compositional word-for-word calque.
- Preserve semantic category breadth exactly. A broad family/class in English must NOT become one subtype, implementation or example in Russian.
- English head nouns such as machine, family, approach, method, model or memory are NOT required to map to their most literal Russian dictionary noun. Use the conventional Russian field label for the concept as a whole.
- For a term explicitly defined with an acronym, choose the conventional Russian term used with that acronym while preserving every semantic component.
- For an archaic/specialist object defined by its properties in literary prose, prefer the precise historical/technical Russian object name over transliteration or a broader neighboring object class.
- `keep` means current_ru is already professionally acceptable and semantically exact.
- `change` means current_ru exists but should be replaced. `add` means no usable current_ru exists. `omit` means ordinary prose, a proper name, or too ambiguous to canonicalize safely.
- `ru` must be only the concise Russian canonical term, without explanations, alternatives, parentheses or translator notes.

Return ONLY JSON {"items":[{"source":"exact supplied source","decision":"keep|change|add|omit","ru":"...","confidence":0.0,"reason":"brief"}]} with exactly one row per supplied candidate."""
        payload = {
            "book_profile": {
                "domain": str(getattr(memory, "domain", "") or ""),
                "voice": str(style.narrative_voice or ""),
                "rhythm": str(style.rhythm or ""),
                "dialogue": str(style.dialogue or ""),
                "humor": str(style.humor or ""),
            },
            "candidates": candidates,
        }
        raw = self.provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0)
        self.stats["calls"] += 1
        obj = _json_from_text(raw)
        rows = [row for row in obj.get("items") or [] if isinstance(row, dict)]
        self.stats["returned"] = len(rows)
        allowed = {str(row.get("source") or "").casefold(): row for row in candidates}
        proposals: list[dict[str, Any]] = []
        for item in rows:
            source = str(item.get("source") or "").strip()
            key = source.casefold()
            candidate = allowed.get(key)
            if not candidate:
                continue
            decision = str(item.get("decision") or "").strip().casefold()
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            current = str(memory.glossary.get(key) or candidate.get("current_ru") or "").strip()
            if decision == "omit":
                self.stats["omitted"] += 1
                continue
            if decision == "keep":
                if not current or not _valid_ru_term(current):
                    continue
                proposed = current
            elif decision in {"change", "add"}:
                proposed = _norm(item.get("ru") or "")
                if confidence < 0.76 or not _valid_ru_term(proposed):
                    continue
            else:
                continue
            proposals.append({
                "source": source,
                "current_ru": current,
                "proposed_ru": proposed,
                "draft_decision": decision,
                "draft_confidence": confidence,
                "evidence": candidate.get("evidence"),
                "contexts": candidate.get("contexts") or [],
            })
        self.stats["draft_proposals"] = len(proposals)
        return proposals

    @staticmethod
    def _critic_system() -> str:
        return """Act as an adversarial senior Russian terminology editor. You are reviewing another translator's proposed EN→RU term canon, not translating prose.
For EVERY row decide keep, revise, or omit.

Audit aggressively for three failure modes:
1) TRANSLATIONESE: a Russian phrase is merely a grammatical word-for-word rendering but is not the conventional term Russian specialists/readers actually use.
2) CATEGORY ERROR: the proposal narrows a broad English class to one subtype/example, broadens a precise object to a neighboring class, or changes the technical/historical denotation.
3) HEAD-NOUN CALQUE: the English head noun was translated literally even though established Russian nomenclature conventionally names the concept with a different head noun. Professional terminology takes precedence over lexical symmetry.

For historical weapons/tools/objects, demand the precise established Russian historical/technical name if context supports one; reject a merely related broader object.
For scientific/ML concepts, demand the conventional Russian textbook/research term. Do not accept an awkward phrase just because every English component is represented. Prefer the label a professional Russian textbook index or specialist glossary would actually use.
`keep` only when proposed_ru is both semantically exact AND idiomatic as a recognized Russian term. `revise` supplies a better concise canonical term. `omit` when the source evidence is insufficient to impose a hard canon.
Do not use current_ru/proposed_ru as authority; they are hypotheses to challenge.
Return ONLY JSON {"items":[{"source":"exact supplied source","decision":"keep|revise|omit","ru":"canonical Russian term or empty","confidence":0.0,"reason":"brief"}]} with exactly one row per input."""

    def _critic(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not proposals:
            return []
        payload = {"proposals": proposals}
        obj: dict[str, Any] | None = None
        backend = self.critic_backend
        if backend is not None and getattr(backend, "available", lambda: False)():
            self.stats["critic_backend"] = str(getattr(backend, "model", "GigaChat"))
            try:
                obj = _giga_json(backend, self._critic_system(), payload, max_tokens=3200)
                self.stats["calls"] += 1
            except Exception as exc:
                self.stats["critic_fallback"] = True
                print(f"[v10-term-critic] backend={self.stats['critic_backend']} error={type(exc).__name__}: {exc}; fallback=deepseek", flush=True)

        if obj is None:
            self.stats["critic_backend"] = "deepseek-fallback"
            raw = self.provider.complete(
                self._critic_system(),
                json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
            )
            self.stats["calls"] += 1
            obj = _json_from_text(raw)

        rows = [row for row in obj.get("items") or [] if isinstance(row, dict)]
        self.stats["critic_returned"] = len(rows)
        return rows

    def verify(self, segments: list[Segment], memory: BookMemory) -> dict[str, Any]:
        if self._load_cache(memory):
            return dict(self.stats)

        candidates = _verification_candidates(segments, memory)
        self.stats["candidates"] = len(candidates)
        if not candidates:
            self._save_cache(memory, {})
            return dict(self.stats)

        try:
            proposals = self._draft(candidates, memory)
        except Exception as exc:
            print(f"[v10-term-verifier-draft] error={type(exc).__name__}: {exc}", flush=True)
            return dict(self.stats)

        proposal_map = {str(row.get("source") or "").casefold(): row for row in proposals}
        final_terms: dict[str, str] = {}
        try:
            critic_rows = self._critic(proposals)
        except Exception as exc:
            print(f"[v10-term-verifier-critic] error={type(exc).__name__}: {exc}", flush=True)
            critic_rows = []

        critic_map = {
            str(row.get("source") or "").casefold(): row
            for row in critic_rows
            if isinstance(row, dict)
        }
        for key, proposal in proposal_map.items():
            review = critic_map.get(key)
            proposed = str(proposal.get("proposed_ru") or "").strip()
            final_ru = ""
            if review is None:
                if float(proposal.get("draft_confidence") or 0) >= 0.90:
                    final_ru = proposed
            else:
                decision = str(review.get("decision") or "").strip().casefold()
                try:
                    confidence = float(review.get("confidence") or 0)
                except Exception:
                    confidence = 0.0
                if decision == "omit":
                    self.stats["critic_omitted"] += 1
                    continue
                if decision == "keep" and confidence >= 0.72:
                    final_ru = proposed
                elif decision == "revise" and confidence >= 0.80:
                    revised = _norm(review.get("ru") or "")
                    if _valid_ru_term(revised):
                        final_ru = revised
                        if revised.casefold() != proposed.casefold():
                            self.stats["critic_revised"] += 1
            if final_ru and _valid_ru_term(final_ru):
                final_terms[key] = final_ru

        applied: dict[str, str] = {}
        for key, ru in final_terms.items():
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