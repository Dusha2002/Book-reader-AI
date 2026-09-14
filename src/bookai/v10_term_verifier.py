from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .gigachat_v3 import GigaChatLightningV3Backend
from .models import BookMemory, Segment
from .v10 import _giga_json, _json_from_text, _norm
from .v10_publication_release import _priority_term_candidates, _technical_style
from .v10_semantic_profile import SourceSemanticProfileVerifier
from .v10_source_bible import _has_clean_russian


_VERIFIER_CACHE_MARKER = "v10-deepseek-terminology-7-focus-consensus"
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


def _focus_profile_candidates(
    technical_terms: list[dict[str, Any]] | None,
    focus_segments: list[Segment],
    memory: BookMemory,
) -> list[dict[str, Any]]:
    """Convert English-only focus discoveries into terminology-review candidates.

    This layer still supplies no Russian answer. It only promotes exact source phrases
    discovered by the semantic profiler into the existing draft -> critic -> consensus
    pipeline. Thus one-off textbook terms gain recall without becoming hard canon from
    a single model call.
    """
    if not _is_academic_domain(memory) or not technical_terms:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in technical_terms:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        key = source.casefold()
        if not source or key in seen:
            continue
        contexts: list[str] = []
        for segment in focus_segments:
            text = str(segment.text or "")
            if source.casefold() in text.casefold():
                contexts.append(_norm(text)[:800])
                if len(contexts) >= 2:
                    break
        if not contexts:
            continue
        rows.append({
            "source": source,
            "current_ru": str(memory.glossary.get(key) or ""),
            "evidence": "focus_semantic_term",
            "source_meaning": _norm(item.get("meaning") or "")[:180],
            "contexts": contexts,
        })
        seen.add(key)
    return rows


def _verification_candidates(
    segments: list[Segment],
    memory: BookMemory,
    *,
    focus_terms: list[dict[str, Any]] | None = None,
    focus_segments: list[Segment] | None = None,
) -> list[dict[str, Any]]:
    # Focus-window discoveries are intentionally first: the book-wide glossary may
    # already be excellent while a rare but important term occurs only in this window.
    rows = _focus_profile_candidates(focus_terms, list(focus_segments or []), memory)
    seen = {str(row.get("source") or "").casefold() for row in rows}

    for row in _priority_term_candidates(segments, memory):
        candidate = dict(row)
        key = str(candidate.get("source") or "").casefold()
        if not key or key in seen:
            continue
        rows.append(candidate)
        seen.add(key)

    for row in _local_definition_candidates(segments):
        key = str(row.get("source") or "").casefold()
        if key in seen:
            continue
        current = str(memory.glossary.get(key) or "")
        row["current_ru"] = current
        rows.append(row)
        seen.add(key)

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
    """Source-only semantic preflight plus consensus-gated terminology canon.

    Book-wide source supplies terminology evidence, while an optional focus window
    supplies local semantic hints and rare technical terms for the text being translated.
    A term becomes a hard canon only after a draft and an adversarial review agree;
    critic revisions require one compact confirmation. If consensus is unavailable,
    the risky term is deliberately de-canonicalized rather than imposing a plausible
    but wrong calque. No Russian reference translation is used.
    """

    def __init__(self, provider: Any, cache_path: Path, *, critic_backend: Any | None = None):
        self.provider = provider
        self.cache_path = Path(cache_path)
        self.critic_backend = critic_backend
        self.stats: dict[str, Any] = {
            "cache_hit": False,
            "domain": "",
            "semantic_profile": {},
            "focus_term_candidates": 0,
            "calls": 0,
            "candidates": 0,
            "returned": 0,
            "draft_proposals": 0,
            "critic_backend": "",
            "critic_fallback": False,
            "critic_usage": {},
            "critic_returned": 0,
            "critic_revised": 0,
            "critic_omitted": 0,
            "consensus_confirm_calls": 0,
            "consensus_confirmed": 0,
            "consensus_rejected": 0,
            "decanonicalized": 0,
            "added": 0,
            "refined": 0,
            "kept": 0,
            "omitted": 0,
            "applied": {},
            "uncertain": [],
        }

    def _cache_data(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _hydrate_domain(self, memory: BookMemory) -> None:
        if str(getattr(memory, "domain", "") or "").strip():
            self.stats["domain"] = str(memory.domain)
            return
        data = self._cache_data()
        domain = str(data.get("domain") or "").strip()
        if domain:
            memory.domain = domain
        self.stats["domain"] = domain

    def _load_cache(self, memory: BookMemory) -> bool:
        data = self._cache_data()
        if not data or data.get("deepseek_terminology_schema") != _VERIFIER_CACHE_MARKER:
            return False
        verified = data.get("deepseek_verified_glossary") or {}
        uncertain = data.get("deepseek_uncertain_terms") or []
        if not isinstance(verified, dict) or not isinstance(uncertain, list):
            return False
        for key in uncertain:
            memory.glossary.pop(str(key).casefold(), None)
        for source, ru in verified.items():
            if source and ru:
                memory.glossary[str(source).casefold()] = str(ru)
        self.stats["cache_hit"] = True
        self.stats["applied"] = {str(k): str(v) for k, v in verified.items() if k and v}
        self.stats["uncertain"] = [str(k) for k in uncertain]
        return True

    def _save_cache(self, memory: BookMemory, applied: dict[str, str], uncertain: set[str]) -> None:
        data = self._cache_data()
        data["glossary"] = dict(memory.glossary)
        data["deepseek_verified_glossary"] = dict(applied)
        data["deepseek_uncertain_terms"] = sorted(uncertain)
        data["deepseek_terminology_schema"] = _VERIFIER_CACHE_MARKER
        if getattr(memory, "domain", ""):
            data["domain"] = str(memory.domain)
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
- `source_meaning`, when supplied, is an ENGLISH source-only denotation hint; use it to disambiguate, never as target wording.
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
            "semantic_hints": dict(memory.semantic_hints),
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
                "source_meaning": candidate.get("source_meaning") or "",
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
For scientific/technical concepts, demand the conventional Russian textbook/research term. Do not accept an awkward phrase just because every English component is represented. Prefer the label a professional Russian textbook index or specialist glossary would actually use.
`keep` only when proposed_ru is both semantically exact AND idiomatic as a recognized Russian term. `revise` supplies a better concise canonical term. `omit` when the source evidence is insufficient to impose a hard canon.
Do not use current_ru/proposed_ru as authority; they are hypotheses to challenge.
Return ONLY JSON {"items":[{"source":"exact supplied source","decision":"keep|revise|omit","ru":"canonical Russian term or empty","confidence":0.0,"reason":"brief"}]} with exactly one row per input."""

    def _ultra_backend(self) -> Any | None:
        if self.critic_backend is not None:
            return self.critic_backend
        backend = GigaChatLightningV3Backend()
        if not backend.available():
            return None
        backend.model = (os.getenv("BOOKAI_GIGACHAT_TERM_MODEL") or "GigaChat-3-Ultra").strip()
        backend.max_tokens = max(1800, min(4200, int(os.getenv("BOOKAI_GIGACHAT_TERM_MAX_TOKENS") or "3200")))
        backend.timeout_seconds = max(8, min(20, int(os.getenv("BOOKAI_GIGACHAT_TERM_TIMEOUT") or "12")))
        backend.max_retries = 0
        backend.retry_backoff = 0.2
        backend.oauth_attempts = 1
        self.critic_backend = backend
        return backend

    def _critic(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not proposals:
            return []
        payload = {"proposals": proposals}
        obj: dict[str, Any] | None = None
        backend = self._ultra_backend()
        if backend is not None:
            self.stats["critic_backend"] = str(getattr(backend, "model", "GigaChat"))
            try:
                obj = _giga_json(backend, self._critic_system(), payload, max_tokens=3200)
                self.stats["calls"] += 1
                usage = getattr(backend, "usage", None)
                if usage is not None and hasattr(usage, "as_dict"):
                    self.stats["critic_usage"] = usage.as_dict()
            except Exception as exc:
                self.stats["critic_fallback"] = True
                print(f"[v10-term-critic] backend={self.stats['critic_backend']} error={type(exc).__name__}: {exc}; fallback=deepseek", flush=True)

        if obj is None:
            self.stats["critic_backend"] = "deepseek-fallback"
            raw = self.provider.complete(self._critic_system(), json.dumps(payload, ensure_ascii=False), temperature=0.0)
            self.stats["calls"] += 1
            obj = _json_from_text(raw)
        rows = [row for row in obj.get("items") or [] if isinstance(row, dict)]
        self.stats["critic_returned"] = len(rows)
        return rows

    def _confirm_revisions(self, rows: list[dict[str, Any]], memory: BookMemory) -> dict[str, str]:
        if not rows:
            return {}
        system = """You are the final SOURCE-ONLY terminology consensus judge. Another editor revised proposed EN→RU canonical terms.
For each row decide accept or reject. Accept only if the revised Russian term is a conventional professional label in the stated domain, preserves the exact breadth/denotation of the English source, and is preferable to both the prior canon and draft proposal. Reject literal-but-nonstandard calques, subtype narrowing, broadened neighboring concepts, and uncertain guesses. Return ONLY JSON {"items":[{"source":"exact source","decision":"accept|reject","confidence":0.0,"reason":"brief"}]} with one row per input."""
        payload = {"domain": str(getattr(memory, "domain", "") or ""), "rows": rows}
        try:
            raw = self.provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0)
            self.stats["calls"] += 1
            self.stats["consensus_confirm_calls"] += 1
            obj = _json_from_text(raw)
        except Exception as exc:
            print(f"[v10-term-consensus] error={type(exc).__name__}: {exc}", flush=True)
            return {}
        out: dict[str, str] = {}
        expected = {str(row.get("source") or "").casefold(): row for row in rows}
        for item in obj.get("items") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("source") or "").strip().casefold()
            row = expected.get(key)
            if not row or str(item.get("decision") or "").strip().casefold() != "accept":
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            threshold = 0.90 if self.stats.get("critic_backend") == "deepseek-fallback" else 0.82
            if confidence >= threshold:
                out[key] = str(row.get("revised_ru") or "").strip()
        self.stats["consensus_confirmed"] += len(out)
        return out

    def verify(
        self,
        segments: list[Segment],
        memory: BookMemory,
        *,
        focus_segments: list[Segment] | None = None,
    ) -> dict[str, Any]:
        self._hydrate_domain(memory)

        semantic_profile = SourceSemanticProfileVerifier(self.provider, self.cache_path)
        semantic_focus = list(focus_segments or segments)
        profile_stats = semantic_profile.analyze(semantic_focus, memory)
        self.stats["semantic_profile"] = profile_stats
        self.stats["domain"] = str(memory.domain or "")

        if self._load_cache(memory):
            return dict(self.stats)

        focus_terms = [
            dict(row) for row in profile_stats.get("technical_terms") or []
            if isinstance(row, dict)
        ]
        candidates = _verification_candidates(
            segments,
            memory,
            focus_terms=focus_terms,
            focus_segments=semantic_focus,
        )
        self.stats["focus_term_candidates"] = sum(
            str(row.get("evidence") or "") == "focus_semantic_term" for row in candidates
        )
        self.stats["candidates"] = len(candidates)
        if not candidates:
            self._save_cache(memory, {}, set())
            return dict(self.stats)

        try:
            proposals = self._draft(candidates, memory)
        except Exception as exc:
            print(f"[v10-term-verifier-draft] error={type(exc).__name__}: {exc}", flush=True)
            return dict(self.stats)

        proposal_map = {str(row.get("source") or "").casefold(): row for row in proposals}
        final_terms: dict[str, str] = {}
        uncertain = {str(row.get("source") or "").casefold() for row in candidates if row.get("source")}
        try:
            critic_rows = self._critic(proposals)
        except Exception as exc:
            print(f"[v10-term-verifier-critic] error={type(exc).__name__}: {exc}", flush=True)
            critic_rows = []

        critic_map = {str(row.get("source") or "").casefold(): row for row in critic_rows if isinstance(row, dict)}
        revisions_to_confirm: list[dict[str, Any]] = []
        for key, proposal in proposal_map.items():
            review = critic_map.get(key)
            proposed = str(proposal.get("proposed_ru") or "").strip()
            draft_conf = float(proposal.get("draft_confidence") or 0)
            if review is None:
                self.stats["consensus_rejected"] += 1
                continue
            decision = str(review.get("decision") or "").strip().casefold()
            try:
                confidence = float(review.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            if decision == "omit":
                self.stats["critic_omitted"] += 1
                self.stats["consensus_rejected"] += 1
                continue
            keep_threshold = 0.90 if self.stats.get("critic_backend") == "deepseek-fallback" else 0.80
            if decision == "keep" and draft_conf >= 0.82 and confidence >= keep_threshold and _valid_ru_term(proposed):
                final_terms[key] = proposed
                uncertain.discard(key)
                continue
            if decision == "revise" and confidence >= 0.82:
                revised = _norm(review.get("ru") or "")
                if _valid_ru_term(revised):
                    self.stats["critic_revised"] += int(revised.casefold() != proposed.casefold())
                    revisions_to_confirm.append({
                        "source": proposal.get("source"),
                        "contexts": proposal.get("contexts") or [],
                        "current_ru": proposal.get("current_ru") or "",
                        "draft_ru": proposed,
                        "revised_ru": revised,
                        "draft_confidence": draft_conf,
                        "critic_confidence": confidence,
                    })
                    continue
            self.stats["consensus_rejected"] += 1

        for key, ru in self._confirm_revisions(revisions_to_confirm, memory).items():
            if _valid_ru_term(ru):
                final_terms[key] = ru
                uncertain.discard(key)
        self.stats["consensus_rejected"] += sum(
            1 for row in revisions_to_confirm if str(row.get("source") or "").casefold() not in final_terms
        )

        for key in sorted(uncertain):
            if key in memory.glossary:
                memory.glossary.pop(key, None)
                self.stats["decanonicalized"] += 1

        applied: dict[str, str] = {}
        for key, ru in final_terms.items():
            old = str(memory.glossary.get(key) or "").strip()
            if old:
                if old.casefold() == ru.casefold():
                    self.stats["kept"] += 1
                    applied[key] = ru
                    continue
                memory.glossary[key] = ru
                self.stats["refined"] += 1
            else:
                memory.glossary[key] = ru
                self.stats["added"] += 1
            applied[key] = ru

        self.stats["applied"] = dict(applied)
        self.stats["uncertain"] = sorted(uncertain)
        self._save_cache(memory, applied, uncertain)
        print("[v10-term-verifier] " + json.dumps(self.stats, ensure_ascii=False), flush=True)
        return dict(self.stats)


__all__ = ["DeepSeekPublicationTerminologyVerifier"]
