from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import _json_from_text, _norm


_SCHEMA = "v10-semantic-profile-2-focus-terms"
_ALLOWED_DOMAINS = {
    "literary_fiction",
    "narrative_nonfiction",
    "academic_technical",
    "general_nonfiction",
    "other",
}
_ALLOWED_KINDS = {"idiom", "archaic_sense", "polysemy", "attachment", "register", "metaphor"}


def _profile_key(segments: list[Segment]) -> str:
    digest = hashlib.sha1()
    for segment in segments:
        digest.update(str(segment.id).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(segment.text or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:20]


def _distributed_excerpts(segments: list[Segment], limit: int = 10) -> list[dict[str, str]]:
    if not segments:
        return []
    count = min(limit, len(segments))
    rows: list[dict[str, str]] = []
    for n in range(count):
        index = round(n * (len(segments) - 1) / max(1, count - 1))
        segment = segments[index]
        text = _norm(segment.text)
        if text:
            rows.append({"id": segment.id, "chapter": str(segment.chapter or ""), "text": text[:1400]})
    return rows


def _exact_source_phrase(joined: str, requested: str) -> str:
    phrase = _norm(requested)
    if not phrase or len(phrase) > 120:
        return ""
    index = joined.casefold().find(phrase.casefold())
    if index < 0:
        return ""
    return joined[index:index + len(phrase)]


def _valid_focus_term(source: str) -> bool:
    phrase = str(source or "").strip()
    if not phrase or len(phrase) > 100:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9.+/-]{1,15}", phrase):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", phrase)
    if not 1 <= len(words) <= 7:
        return False
    if len(words) == 1 and len(words[0]) < 7 and "-" not in words[0]:
        return False
    return True


class SourceSemanticProfileVerifier:
    """One cached source-only pass per translated span.

    It independently verifies domain, records semantic meaning hints for risky source
    phrases, and — for academic/technical prose — discovers high-value terminology in
    the *actual translation window*. All discovery remains English-only. Russian canon
    is decided later by the independent terminology consensus stage, so this profiler
    cannot silently inject a target translation.
    """

    def __init__(self, provider: Any, cache_path: Path):
        self.provider = provider
        self.cache_path = Path(cache_path)
        self.stats: dict[str, Any] = {
            "cache_hit": False,
            "calls": 0,
            "prior_domain": "",
            "verified_domain": "",
            "domain_confidence": 0.0,
            "domain_changed": False,
            "hints": 0,
            "hint_kinds": {},
            "technical_term_count": 0,
            "technical_terms": [],
        }

    def _cache_data(self) -> dict[str, Any]:
        try:
            data = json.loads(self.cache_path.read_text("utf-8")) if self.cache_path.exists() else {}
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def _save_cache(
        self,
        key: str,
        memory: BookMemory,
        hints: dict[str, str],
        raw_rows: list[dict[str, Any]],
        technical_terms: list[dict[str, Any]],
    ) -> None:
        data = self._cache_data()
        profiles = data.get("semantic_profiles")
        if not isinstance(profiles, dict):
            profiles = {}
        profiles[key] = {
            "schema": _SCHEMA,
            "domain": str(memory.domain or ""),
            "semantic_hints": dict(hints),
            "rows": raw_rows,
            "technical_terms": technical_terms,
        }
        data["semantic_profiles"] = profiles
        if memory.domain:
            data["domain"] = str(memory.domain)
        try:
            self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except OSError:
            pass

    def analyze(self, segments: list[Segment], memory: BookMemory) -> dict[str, Any]:
        self.stats["prior_domain"] = str(memory.domain or "")
        key = _profile_key(segments)
        data = self._cache_data()
        profiles = data.get("semantic_profiles") or {}
        cached = profiles.get(key) if isinstance(profiles, dict) else None
        if isinstance(cached, dict) and cached.get("schema") == _SCHEMA:
            domain = str(cached.get("domain") or "").strip()
            if domain in _ALLOWED_DOMAINS:
                memory.domain = domain
            hints = cached.get("semantic_hints") or {}
            if isinstance(hints, dict):
                memory.semantic_hints = {str(k): str(v) for k, v in hints.items() if k and v}
            cached_terms = cached.get("technical_terms") or []
            technical_terms = [dict(row) for row in cached_terms if isinstance(row, dict)]
            self.stats.update({
                "cache_hit": True,
                "verified_domain": str(memory.domain or ""),
                "hints": len(memory.semantic_hints),
                "technical_term_count": len(technical_terms),
                "technical_terms": technical_terms,
            })
            return dict(self.stats)

        excerpts = _distributed_excerpts(segments)
        if not excerpts:
            return dict(self.stats)
        joined = "\n\n".join(str(segment.text or "") for segment in segments)
        system = """You are a SOURCE-ONLY editorial profiler for an EN→RU book translation engine. Do NOT translate anything into Russian and do not use any target/reference text.

Task A — independently classify the source span by what it IS, not merely by prose style:
- literary_fiction: an invented narrative presented as fiction;
- narrative_nonfiction: autobiography, memoir, biography, history, travelogue, reportage, or another factual narrative about claimed real people/events;
- academic_technical: scientific, mathematical, engineering, textbook, research or technical exposition;
- general_nonfiction: factual argument/exposition that is neither narrative nonfiction nor academic/technical;
- other: only when none fits.
A first-person narrative is NOT automatically fiction. Autobiographical claims, real-life recollection, descendants/ancestors, historical testimony and memoir framing are evidence for narrative_nonfiction.

Task B — find at most 12 exact English source phrases that are unusually risky under literal EN→RU translation because of an idiom, archaic/historical sense, polysemy, non-compositional metaphor, deceptive syntactic attachment, or register-specific meaning. Do NOT list ordinary clear phrases, names, or stylistic preferences with no semantic risk. For each risk, copy `source` exactly from an excerpt and give `meaning` as a concise plain-ENGLISH contextual sense, not a Russian translation.

Task C — ONLY when the source is academic_technical, find at most 12 exact English specialist terms or compact noun phrases in these excerpts for which established Russian professional terminology matters. Include important one-off terms: do not require whole-book recurrence. Prefer terms whose accepted field label may differ from a literal word-by-word rendering, whose head noun is easy to calque incorrectly, or whose precise conventional wording is important for a textbook. Do NOT include people, organizations, brands, citations, standalone acronyms, mathematical variable names, generic words such as model/system/method alone, or whole sentences. Copy `source` EXACTLY and give only a short ENGLISH denotation in `meaning`; do not propose Russian wording.

Return ONLY JSON:
{"domain":"literary_fiction|narrative_nonfiction|academic_technical|general_nonfiction|other","domain_confidence":0.0,"domain_reason":"brief source evidence","risks":[{"source":"exact source substring","meaning":"concise English contextual meaning","kind":"idiom|archaic_sense|polysemy|attachment|register|metaphor","confidence":0.0}],"technical_terms":[{"source":"exact technical term","meaning":"concise English denotation","confidence":0.0}]}.
Only include risks/terms with confidence >= 0.75. For non-academic text return technical_terms: []."""
        payload = {
            "current_domain_guess": str(memory.domain or ""),
            "current_style": {
                "voice": str(memory.style.narrative_voice or ""),
                "rhythm": str(memory.style.rhythm or ""),
                "dialogue": str(memory.style.dialogue or ""),
                "humor": str(memory.style.humor or ""),
            },
            "existing_glossary_sources": list(memory.glossary.keys())[:120],
            "excerpts": excerpts,
        }
        try:
            raw = self.provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0)
            self.stats["calls"] = 1
            obj = _json_from_text(raw)
        except Exception as exc:
            print(f"[v10-semantic-profile] error={type(exc).__name__}: {exc}", flush=True)
            return dict(self.stats)

        domain = str(obj.get("domain") or "").strip().casefold()
        try:
            domain_confidence = float(obj.get("domain_confidence") or 0.0)
        except Exception:
            domain_confidence = 0.0
        self.stats["domain_confidence"] = domain_confidence
        prior = str(memory.domain or "")
        if domain in _ALLOWED_DOMAINS and domain_confidence >= 0.80:
            memory.domain = domain
        self.stats["verified_domain"] = str(memory.domain or "")
        self.stats["domain_changed"] = bool(prior and memory.domain and prior != memory.domain)

        hints: dict[str, str] = {}
        rows_for_cache: list[dict[str, Any]] = []
        kind_counts: dict[str, int] = {}
        for row in obj.get("risks") or []:
            if not isinstance(row, dict):
                continue
            try:
                confidence = float(row.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            if confidence < 0.78:
                continue
            kind = str(row.get("kind") or "").strip().casefold()
            if kind not in _ALLOWED_KINDS:
                continue
            source = _exact_source_phrase(joined, str(row.get("source") or ""))
            meaning = _norm(row.get("meaning") or "")
            if not source or not meaning or len(meaning) > 180:
                continue
            if any("А" <= ch <= "я" or ch in "Ёё" for ch in meaning):
                continue
            hints[source] = meaning
            rows_for_cache.append({"source": source, "meaning": meaning, "kind": kind, "confidence": confidence})
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            if len(hints) >= 12:
                break

        technical_terms: list[dict[str, Any]] = []
        seen_terms: set[str] = set()
        if memory.domain == "academic_technical":
            for row in obj.get("technical_terms") or []:
                if not isinstance(row, dict):
                    continue
                try:
                    confidence = float(row.get("confidence") or 0.0)
                except Exception:
                    confidence = 0.0
                if confidence < 0.78:
                    continue
                source = _exact_source_phrase(joined, str(row.get("source") or ""))
                meaning = _norm(row.get("meaning") or "")
                key = source.casefold()
                if not source or key in seen_terms or not _valid_focus_term(source):
                    continue
                if not meaning or len(meaning) > 180 or any("А" <= ch <= "я" or ch in "Ёё" for ch in meaning):
                    continue
                technical_terms.append({"source": source, "meaning": meaning, "confidence": confidence})
                seen_terms.add(key)
                if len(technical_terms) >= 12:
                    break

        memory.semantic_hints = hints
        self.stats["hints"] = len(hints)
        self.stats["hint_kinds"] = kind_counts
        self.stats["technical_term_count"] = len(technical_terms)
        self.stats["technical_terms"] = technical_terms
        self._save_cache(key, memory, hints, rows_for_cache, technical_terms)
        print("[v10-semantic-profile] " + json.dumps(self.stats, ensure_ascii=False), flush=True)
        return dict(self.stats)


__all__ = ["SourceSemanticProfileVerifier"]
