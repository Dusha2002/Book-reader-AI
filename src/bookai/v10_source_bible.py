from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import _giga_json, _norm


_CACHE_VERSION = "v10-source-only-2"
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_COMMON_CAPITALIZED = {
    "The", "A", "An", "And", "But", "Or", "If", "When", "While", "Then", "There", "This", "That",
    "He", "She", "It", "They", "We", "I", "You", "His", "Her", "Their", "Chapter", "One", "Two",
    "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
    "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
}
_TITLE_WORDS = {
    "duke", "duchess", "king", "queen", "prince", "princess", "count", "countess", "lord", "lady",
    "elector", "general", "captain", "commissioner", "master", "abbot", "bishop", "emperor", "empress",
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
_STOP_MODIFIERS = {
    "the", "a", "an", "to", "in", "on", "at", "of", "for", "from", "with", "by", "into", "onto",
    "his", "her", "their", "our", "your", "my", "its", "this", "that", "these", "those", "and", "or",
}


def _has_clean_russian(value: str) -> bool:
    value = str(value or "").strip()
    return bool(value and _CYRILLIC.search(value) and not _LATIN.search(value))


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
            "rejected_terms": 0,
        }

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        proper_counts: Counter[str] = Counter()
        title_evidence: Counter[str] = Counter()
        tech_counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}
        proper_re = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){0,2}\b")
        token_re = re.compile(r"[A-Za-z][A-Za-z'-]*")

        def remember(key: str, segment: Segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 3:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:520]})

        for segment in segments:
            text = str(segment.text or "")
            for match in proper_re.finditer(text):
                value = _norm(match.group(0))
                if not value or value.split()[0] in _COMMON_CAPITALIZED:
                    continue
                proper_counts[value] += 1
                prefix = text[max(0, match.start() - 30):match.start()].casefold()
                if any(re.search(rf"\b{re.escape(title)}\s+$", prefix) for title in _TITLE_WORDS):
                    title_evidence[value] += 1
                remember(value, segment)

            tokens = token_re.findall(text)
            low_tokens = [t.casefold() for t in tokens]
            for i, token in enumerate(low_tokens):
                if token in _RARE_TECH:
                    tech_counts[token] += 1
                    remember(token, segment)
                if token not in _TECH_HEADS:
                    continue
                # Ambiguous heads such as file/saw/plate are glossary candidates only
                # when they have a content modifier. This blocks `in the file`,
                # `to the file`, `the saw`, etc.
                modifiers: list[str] = []
                for j in range(max(0, i - 2), i):
                    part = low_tokens[j]
                    if part in _STOP_MODIFIERS or len(part) < 3:
                        modifiers = []
                        continue
                    modifiers.append(part)
                if modifiers:
                    value = " ".join(modifiers[-2:] + [token])
                    tech_counts[value] += 1
                    remember(value, segment)

        records: list[dict[str, Any]] = []
        for value, count in proper_counts.most_common(150):
            if count < 2 and title_evidence[value] < 1:
                continue
            records.append({
                "candidate": value,
                "kind_hint": "proper",
                "frequency": count,
                "contexts": contexts.get(value, []),
            })
        for value, count in tech_counts.most_common(120):
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
        memory.rolling_summary = _norm(data.get("summary") or "")[:7000]
        return memory

    def _name_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        batch_size = max(16, min(40, int(os.getenv("BOOKAI_V10_NAME_BATCH") or "24")))
        system = """Build a SOURCE-ONLY Russian canon for fictional proper names in an English literary novel.
No Russian reference translation exists. For every returned item, use only the supplied English spelling and English contexts.
Return only genuine person/place/institution proper names; omit ordinary capitalized words and titles used without a name.
Prefer spelling-preserving Russian transliteration over guessed pronunciation: preserve visible consonants and vowel sequences,
do not silently collapse written vowels, do not invent letters unsupported by the source, and keep one stable dictionary form.
For a person, infer gender/role only when the English contexts actually support it; otherwise gender=unknown.
ONLY JSON {"names":[{"source":"...","ru":"...","kind":"person|place|institution|other","gender":"male|female|unknown","role":"...","voice":"...","confidence":0.0}]}.
"""
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            allowed = {str(row["candidate"]): row for row in batch}
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=4200)
                self.stats["analysis_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible-names] batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                continue
            for item in obj.get("names") or []:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source") or "").strip()
                ru = _norm(item.get("ru") or "")
                try:
                    confidence = float(item.get("confidence") or 0)
                except Exception:
                    confidence = 0.0
                if source not in allowed or confidence < 0.78 or not _has_clean_russian(ru):
                    self.stats["rejected_names"] += 1
                    continue
                kind = str(item.get("kind") or "other").casefold()
                if kind not in {"person", "place", "institution", "other"}:
                    kind = "other"
                aggregate["canonicals"][source] = ru
                if kind == "person":
                    gender = str(item.get("gender") or "unknown").casefold()
                    if gender not in {"male", "female", "unknown"}:
                        gender = "unknown"
                    role = _norm(item.get("role") or "")[:220]
                    voice = _norm(item.get("voice") or "")[:220]
                    aggregate["characters"][source] = f"gender={gender};role={role};voice={voice}"

    def _term_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        batch_size = max(16, min(40, int(os.getenv("BOOKAI_V10_TERM_BATCH") or "24")))
        system = """Build a SOURCE-ONLY high-precision EN→RU technical glossary for a literary novel.
Candidates are prefiltered technical phrases or rare specialist nouns with English contexts. Keep an item ONLY if its denotation
is stable enough across those contexts to deserve a book-wide glossary rule. Omit ambiguous ordinary words and any phrase whose
Russian rendering should vary by sentence. Choose the precise object/material/armour/tool term, not a generic paraphrase.
No Russian reference exists. ONLY JSON {"terms":[{"source":"...","ru":"...","confidence":0.0,"reason":"..."}]}.
"""
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            allowed = {str(row["candidate"]): row for row in batch}
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=3800)
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
                if source not in allowed or confidence < 0.84 or not _has_clean_russian(ru):
                    self.stats["rejected_terms"] += 1
                    continue
                aggregate["glossary"][source] = ru

    def _style(self, segments: list[Segment], aggregate: dict[str, Any]) -> None:
        sample_count = min(30, len(segments))
        excerpts: list[dict[str, str]] = []
        if sample_count:
            for n in range(sample_count):
                index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
                seg = segments[index]
                excerpts.append({"chapter": str(seg.chapter or ""), "text": _norm(seg.text)[:700]})
        system = """Infer SOURCE-ONLY translation guidance from English excerpts distributed across an entire literary novel.
Do not invent a Russian reference style. Describe observable source properties that a Russian translator should preserve:
narrative distance/register, sentence rhythm, dialogue register, humor/irony, and a short continuity summary.
ONLY JSON {"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},"summary":"..."}."""
        try:
            obj = _giga_json(self.backend, system, {"excerpts": excerpts}, max_tokens=2600)
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
