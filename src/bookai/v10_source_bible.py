from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import _giga_json, _norm


_CACHE_VERSION = "v10-source-only-3"
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_EN_VOWELS = set("aeiouy")
_RU_VOWELS = set("аеёиоуыэюя")
_COMMON_CAPITALIZED = {
    "The", "A", "An", "And", "But", "Or", "If", "When", "While", "Then", "There", "This", "That",
    "He", "She", "It", "They", "We", "I", "You", "His", "Her", "Their", "Chapter", "One", "Two",
    "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
    "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty", "In", "No", "What",
    "So", "As", "Not", "All", "Well", "At", "For", "Now", "Yes", "Of", "How", "After", "On", "Fine",
    "Just", "Oh", "To", "Do", "Maybe", "My", "Why", "By", "Instead", "Once", "Since", "Before", "Even",
    "Anyway", "Actually", "Apparently", "Perhaps", "Surely", "Really", "Meanwhile", "Next", "Later",
}
_TITLE_WORDS = {
    "duke", "duchess", "king", "queen", "prince", "princess", "count", "countess", "lord", "lady",
    "elector", "general", "captain", "commissioner", "master", "abbot", "bishop", "emperor", "empress",
    "chancellor", "doctor", "marshal", "colonel", "governor",
}
_RARE_TECH = {
    "cuiss", "cuisse", "cuisses", "gorget", "brigandine", "tinplate", "treadle", "scorpion",
    "halberd", "arquebus", "greave", "greaves", "pauldron", "pauldrons", "vambrace", "vambraces",
}
_TECH_HEADS = {
    "saw", "file", "plate", "lock", "locks", "gear", "gears", "blade", "blades", "tool", "tools",
    "machine", "screw", "screws", "spring", "springs", "stock", "armour", "armor", "lathe", "pulley",
    "rivet", "rivets", "anvil", "vise", "vice", "rasp", "caliper", "calipers", "drill", "crossbow",
}
_TECH_MODIFIERS = {
    "treadle", "lead", "set", "recoiling", "coil", "leaf", "steel", "iron", "brass", "bronze", "tinplate",
    "rectangular", "square", "sliding", "slider", "outside", "inside", "tapping", "fish-scale", "mezentine",
    "tempered", "hardened", "forged", "ground", "cutting", "parting", "boring", "threaded",
}


def _has_clean_russian(value: str) -> bool:
    value = str(value or "").strip()
    return bool(value and _CYRILLIC.search(value) and not _LATIN.search(value))


def _spelling_preserving(source: str, ru: str) -> bool:
    """Reject obvious source-letter collapse without pretending to know pronunciation.

    v9ad empirically worked best when fictional names preserved visible spelling.
    This is deliberately a weak guard: it catches Miel->Мель / Valens->Вальс style
    losses while leaving the actual transliteration decision to GigaChat.
    """
    src_letters = [c.casefold() for c in str(source or "") if c.isalpha() and c.isascii()]
    ru_letters = [c.casefold() for c in str(ru or "") if "а" <= c.casefold() <= "я" or c.casefold() == "ё"]
    if not src_letters or not ru_letters:
        return False
    src_vowels = sum(c in _EN_VOWELS for c in src_letters)
    ru_vowels = sum(c in _RU_VOWELS for c in ru_letters)
    if len(src_letters) >= 4 and src_vowels >= 2 and ru_vowels < src_vowels - 1:
        return False
    # A large overall collapse is suspicious for invented literary names.
    if len(src_letters) >= 6 and len(ru_letters) < len(src_letters) - 3:
        return False
    return True


class SourceOnlyBookBibleBuilder:
    """High-precision whole-book source intelligence for clean v10.

    No Russian reference, gold translation, old v9 cache, or hand-seeded target spelling
    is read. Names and technical terms are analyzed separately so an ordinary phrase can
    never silently become an authoritative glossary entry.
    """

    def __init__(self, backend, cache_path: Path):
        self.backend = backend
        self.cache_path = Path(cache_path)
        self.stats: dict[str, Any] = {
            "cache_hit": False,
            "analysis_calls": 0,
            "source_only": True,
            "reference_seed": False,
            "rejected_names": 0,
            "name_recovery_calls": 0,
            "name_recovered": 0,
            "rejected_terms": 0,
        }

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        proper_counts: Counter[str] = Counter()
        mid_counts: Counter[str] = Counter()
        title_evidence: Counter[str] = Counter()
        tech_counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}
        proper_re = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?){0,2}\b")
        token_re = re.compile(r"[A-Za-z][A-Za-z'-]*")

        def remember(key: str, segment: Segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 3:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:520]})

        for segment in segments:
            text = str(segment.text or "")
            for match in proper_re.finditer(text):
                value = _norm(match.group(0))
                if not value or value.split()[0] in _COMMON_CAPITALIZED or "'" in value:
                    continue
                proper_counts[value] += 1
                before = text[:match.start()].rstrip()
                if before and before[-1] not in ".!?…\"'“‘":
                    mid_counts[value] += 1
                prefix = text[max(0, match.start() - 36):match.start()].casefold()
                if any(re.search(rf"\b{re.escape(title)}\s+$", prefix) for title in _TITLE_WORDS):
                    title_evidence[value] += 1
                remember(value, segment)

            tokens = token_re.findall(text)
            low_tokens = [t.casefold() for t in tokens]
            for i, token in enumerate(low_tokens):
                if token in _RARE_TECH:
                    tech_counts[token] += 1
                    remember(token, segment)
                if token not in _TECH_HEADS or i == 0:
                    continue
                prev = low_tokens[i - 1]
                if prev in _TECH_MODIFIERS or prev in _RARE_TECH or "-" in prev or prev.endswith(("ing", "ed", "al", "ic", "ous", "ive")):
                    value = f"{prev} {token}"
                    tech_counts[value] += 1
                    remember(value, segment)

        records: list[dict[str, Any]] = []
        name_limit = max(40, min(90, int(os.getenv("BOOKAI_V10_NAME_MAX") or "64")))
        ranked_names = sorted(
            proper_counts.items(),
            key=lambda item: (title_evidence[item[0]] > 0, mid_counts[item[0]] > 0, item[1], len(item[0])),
            reverse=True,
        )
        for value, count in ranked_names:
            if count < 2:
                continue
            # Sentence-start-only English words are the dominant false positive.
            if mid_counts[value] == 0 and title_evidence[value] == 0 and count < 5:
                continue
            records.append({
                "candidate": value,
                "kind_hint": "proper",
                "frequency": count,
                "mid_frequency": mid_counts[value],
                "title_evidence": title_evidence[value],
                "contexts": contexts.get(value, []),
            })
            if sum(row.get("kind_hint") == "proper" for row in records) >= name_limit:
                break

        for value, count in tech_counts.most_common(80):
            if value not in _RARE_TECH and count < 2:
                continue
            records.append({
                "candidate": value,
                "kind_hint": "technical_term",
                "frequency": count,
                "contexts": contexts.get(value, []),
            })
        return records

    @staticmethod
    def _memory_from_data(data: dict[str, Any]) -> BookMemory:
        memory = BookMemory()
        style = dict(data.get("style") or {})
        for attr in ("narrative_voice", "rhythm", "dialogue", "humor"):
            value = _norm(style.get(attr) or "")
            if value:
                setattr(memory.style, attr, value)
        canon = {str(k): _norm(v) for k, v in dict(data.get("canonicals") or {}).items() if _norm(v)}
        glossary = {str(k): _norm(v) for k, v in dict(data.get("glossary") or {}).items() if _norm(v)}
        memory.glossary.update(glossary)
        memory.glossary.update(canon)
        for name, desc in dict(data.get("characters") or {}).items():
            clean = _norm(desc)
            ru = canon.get(str(name), "")
            memory.characters[str(name)] = (f"ru={ru};" if ru else "") + clean
        # Every accepted name gets a canon memory even when its role is unknown.
        for name, ru in canon.items():
            memory.characters.setdefault(name, f"ru={ru};gender=unknown;role=proper_name")
        memory.rolling_summary = _norm(data.get("summary") or "")[:7000]
        return memory

    @staticmethod
    def _accept_name_item(item: dict[str, Any], allowed: dict[str, dict[str, Any]], *, threshold: float) -> tuple[str, str, str, str, str] | None:
        source = str(item.get("source") or "").strip()
        ru = _norm(item.get("ru") or "")
        try:
            confidence = float(item.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        if source not in allowed or confidence < threshold or not _has_clean_russian(ru) or not _spelling_preserving(source, ru):
            return None
        kind = str(item.get("kind") or "other").casefold()
        if kind not in {"person", "place", "institution", "other"}:
            kind = "other"
        gender = str(item.get("gender") or "unknown").casefold()
        if gender not in {"male", "female", "unknown"}:
            gender = "unknown"
        role = _norm(item.get("role") or "")[:220]
        voice = _norm(item.get("voice") or "")[:220]
        return source, ru, kind, gender, f"role={role};voice={voice}"

    def _store_name(self, aggregate: dict[str, Any], accepted: tuple[str, str, str, str, str]) -> None:
        source, ru, kind, gender, notes = accepted
        aggregate["canonicals"][source] = ru
        if kind == "person":
            aggregate["characters"][source] = f"gender={gender};{notes}"

    def _name_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        batch_size = max(12, min(28, int(os.getenv("BOOKAI_V10_NAME_BATCH") or "20")))
        system = """Create a SOURCE-ONLY book-wide Russian spelling canon for fictional proper names.
You receive English candidate names, frequency, and English contexts. No Russian reference exists.
Return EVERY candidate that is genuinely a person/place/institution proper name; omit ordinary English words/titles.
For invented names prefer SPELLING-PRESERVING transliteration over guessed pronunciation. Preserve visible source information:
do not collapse a written final -ea to one vowel; preserve internal/final written vowel sequences; preserve written consonant
clusters rather than silently deleting letters; do not add й/я unless source spelling supports it. Use normal Russian orthography,
but never simplify away visible source letters just because a pronunciation is plausible. Keep ONE stable dictionary form.
ONLY JSON {"names":[{"source":"...","ru":"...","kind":"person|place|institution|other","gender":"male|female|unknown","role":"...","voice":"...","confidence":0.0}]}.
"""
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            allowed = {str(row["candidate"]): row for row in batch}
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=3800)
                self.stats["analysis_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible-names] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                continue
            for item in obj.get("names") or []:
                if not isinstance(item, dict):
                    continue
                accepted = self._accept_name_item(item, allowed, threshold=0.72)
                if accepted:
                    self._store_name(aggregate, accepted)
                else:
                    self.stats["rejected_names"] += 1

        # v9ad's key strength was coverage: important recurring names must not be
        # left free for the chapter translator to improvise. Recover only frequent
        # candidates missed/rejected above, still using Giga rather than DeepSeek.
        residual = [row for row in rows if row["candidate"] not in aggregate["canonicals"] and int(row.get("frequency") or 0) >= 4]
        recovery_system = """STRICT SOURCE-SPELLING Russian transliteration recovery for recurring fictional names.
For each supplied candidate decide is_name. If true, return a stable Russian spelling that preserves visible source letters
and vowel sequences; do not guess a pronunciation that deletes written material. Examples of the RULE, not target answers:
a two-vowel written sequence must not collapse to a one-vowel Russian form; a multi-letter consonant cluster must not silently
lose a consonant. Use only English source/context. Return every row. ONLY JSON
{"items":[{"source":"...","is_name":true,"ru":"...","kind":"person|place|institution|other","gender":"male|female|unknown","confidence":0.0}]}.
"""
        for start in range(0, len(residual), 16):
            batch = residual[start:start + 16]
            allowed = {str(row["candidate"]): row for row in batch}
            try:
                obj = _giga_json(self.backend, recovery_system, {"candidates": batch}, max_tokens=2600)
                self.stats["analysis_calls"] += 1
                self.stats["name_recovery_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible-name-recovery] batch={start // 16 + 1} error={type(exc).__name__}", flush=True)
                continue
            for item in obj.get("items") or []:
                if not isinstance(item, dict) or item.get("is_name") is not True:
                    continue
                accepted = self._accept_name_item(item, allowed, threshold=0.58)
                if accepted:
                    before = len(aggregate["canonicals"])
                    self._store_name(aggregate, accepted)
                    if len(aggregate["canonicals"]) > before:
                        self.stats["name_recovered"] += 1

    def _term_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        batch_size = max(10, min(24, int(os.getenv("BOOKAI_V10_TERM_BATCH") or "16")))
        system = """Build a SOURCE-ONLY HIGH-PRECISION EN→RU technical glossary for a literary novel.
Candidates are prefiltered technical phrases or rare specialist nouns with English contexts. Keep an item ONLY if its denotation
is stable enough across those contexts to deserve a book-wide rule. Omit ambiguous ordinary words and sentence fragments.
Return ONLY the Russian TERM itself, no explanation, parentheses or definition. No Russian reference exists.
ONLY JSON {"terms":[{"source":"...","ru":"...","confidence":0.0}]}.
"""
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            allowed = {str(row["candidate"]).casefold(): row for row in batch}
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=2600)
                self.stats["analysis_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible-terms] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                continue
            for item in obj.get("terms") or []:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source") or "").strip().casefold()
                ru = _norm(item.get("ru") or "")
                try:
                    confidence = float(item.get("confidence") or 0)
                except Exception:
                    confidence = 0.0
                if source not in allowed or confidence < 0.92 or not _has_clean_russian(ru) or any(ch in ru for ch in "()[]{}") or len(ru.split()) > 4:
                    self.stats["rejected_terms"] += 1
                    continue
                aggregate["glossary"][source] = ru

    def _style(self, segments: list[Segment], aggregate: dict[str, Any]) -> None:
        sample_count = min(26, len(segments))
        excerpts: list[dict[str, str]] = []
        if sample_count:
            for n in range(sample_count):
                index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
                seg = segments[index]
                excerpts.append({"chapter": str(seg.chapter or ""), "text": _norm(seg.text)[:650]})
        system = """Infer SOURCE-ONLY translation guidance from English excerpts distributed across an entire literary novel.
Do not invent a Russian reference style. Describe observable source properties to preserve: narrative register, sentence rhythm,
dialogue register, humor/irony, and a short continuity summary. ONLY JSON
{"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},"summary":"..."}."""
        try:
            obj = _giga_json(self.backend, system, {"excerpts": excerpts}, max_tokens=2200)
            self.stats["analysis_calls"] += 1
            if isinstance(obj.get("style"), dict):
                aggregate["style"] = {k: _norm(v) for k, v in obj["style"].items() if _norm(v)}
            aggregate["summary"] = _norm(obj.get("summary") or "")
        except Exception as exc:
            print(f"[v10-bible-style] error={type(exc).__name__}", flush=True)

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text("utf-8"))
                if isinstance(data, dict) and data.get("version") == _CACHE_VERSION:
                    self.stats.update({
                        "cache_hit": True,
                        "candidate_records": int(data.get("candidate_records") or 0),
                        "glossary": len(data.get("glossary") or {}),
                        "canon": len(data.get("canonicals") or {}),
                        "characters": len(data.get("characters") or {}),
                        "whole_book_segments_scanned": len(segments),
                    })
                    return self._memory_from_data(data), dict(self.stats)
            except Exception:
                pass

        records = self._candidate_records(segments)
        names = [row for row in records if row.get("kind_hint") == "proper"]
        terms = [row for row in records if row.get("kind_hint") == "technical_term"]
        aggregate: dict[str, Any] = {
            "version": _CACHE_VERSION,
            "source_only": True,
            "reference_seed": False,
            "candidate_records": len(records),
            "glossary": {},
            "canonicals": {},
            "characters": {},
            "style": {},
            "summary": "",
        }
        self._name_batches(names, aggregate)
        self._term_batches(terms, aggregate)
        self._style(segments, aggregate)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), "utf-8")
        self.stats.update({
            "candidate_records": len(records),
            "name_candidates": len(names),
            "term_candidates": len(terms),
            "glossary": len(aggregate["glossary"]),
            "canon": len(aggregate["canonicals"]),
            "characters": len(aggregate["characters"]),
            "whole_book_segments_scanned": len(segments),
        })
        print("[v10-source-bible] " + json.dumps(self.stats, ensure_ascii=False), flush=True)
        return self._memory_from_data(aggregate), dict(self.stats)
