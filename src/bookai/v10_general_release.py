from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _json_from_text, _norm
from .v10_deepseek import PROVEN_DEEPSEEK_CODES
from .v10_dialogue import DialogueDiscourseGuard, source_has_dialogue
from .v10_release import HardenedDeepSeekSemanticSpecialist, HardenedV10QualityQA
from .v10_source_bible import SourceOnlyBookBibleBuilder


_GENERAL_CACHE_MARKER = "v10-general-release-1"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "onto", "over", "under", "about", "after", "before",
    "through", "between", "without", "within", "while", "where", "when", "which", "what", "who", "whom", "whose",
    "there", "their", "they", "them", "then", "than", "have", "has", "had", "having", "would", "could", "should",
    "will", "shall", "might", "must", "been", "being", "were", "was", "are", "is", "am", "not", "but", "for",
    "you", "your", "yours", "his", "her", "hers", "our", "ours", "its", "him", "she", "he", "we", "i", "a",
    "an", "of", "to", "in", "on", "at", "by", "as", "or", "if", "so", "do", "did", "does", "done", "said",
    "say", "says", "one", "two", "three", "first", "second", "other", "another", "some", "any", "all", "each",
}
_MATERIAL_STEMS = {
    "steel": ("сталь", "стальн"),
    "iron": ("желез", "чугун"),
    "copper": ("мед", "медн"),
    "brass": ("латун",),
    "bronze": ("бронз",),
    "aluminium": ("алюмин",),
    "aluminum": ("алюмин",),
}


def _ru_has_stem(text: str, stems: tuple[str, ...]) -> bool:
    words = re.findall(r"[а-яё]+", str(text or "").casefold())
    return any(any(word.startswith(stem) for stem in stems) for word in words)


def _canon_stem(value: str) -> str:
    words = re.findall(r"[А-Яа-яЁё]+", str(value or ""))
    if not words:
        return ""
    word = words[0].casefold().replace("ё", "е")
    return word[: max(3, min(6, len(word) - 1 if len(word) > 4 else len(word)))]


class FinalBookBibleBuilder(SourceOnlyBookBibleBuilder):
    """Book-adaptive source-only memory with no title-specific vocabulary in code.

    Proper entities are discovered by the source-bible model. Technical/rare term
    candidates are generated from recurring lexical n-grams across the whole book,
    then filtered by the existing high-confidence term classifier. This makes the
    same release pipeline usable for fantasy, SF, history, romance or technical prose.
    """

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        records = [dict(row) for row in SourceOnlyBookBibleBuilder._candidate_records(segments)]
        existing = {str(row.get("candidate") or "").casefold() for row in records}
        proper_lower = {
            str(row.get("candidate") or "").casefold()
            for row in records
            if row.get("kind_hint") == "proper"
        }

        counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}

        def remember(key: str, segment: Segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 3:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:520]})

        for segment in segments:
            tokens = [token.casefold() for token in _TOKEN_RE.findall(str(segment.text or ""))]
            for n in (1, 2, 3):
                for i in range(0, max(0, len(tokens) - n + 1)):
                    window = tokens[i:i + n]
                    if n == 1:
                        token = window[0]
                        if len(token) < 7 or token in _STOPWORDS:
                            continue
                    else:
                        content = [token for token in window if token not in _STOPWORDS and len(token) >= 4]
                        if len(content) < 2:
                            continue
                        if window[0] in _STOPWORDS and window[-1] in _STOPWORDS:
                            continue
                    phrase = " ".join(window)
                    if phrase in proper_lower:
                        continue
                    counts[phrase] += 1
                    remember(phrase, segment)

        ranked = sorted(
            (
                (phrase, count)
                for phrase, count in counts.items()
                if 2 <= count <= 40 and phrase not in existing
            ),
            key=lambda item: (
                item[0].count(" ") >= 1,
                item[0].count(" "),
                min(item[1], 12),
                len(item[0]),
            ),
            reverse=True,
        )
        added = 0
        for phrase, count in ranked:
            if added >= 96:
                break
            records.append({
                "candidate": phrase,
                "kind_hint": "technical_term",
                "frequency": count,
                "contexts": contexts.get(phrase, []),
            })
            existing.add(phrase)
            added += 1
        return records

    def _store_name(self, aggregate: dict[str, Any], accepted: tuple[str, str, str, str, str]) -> None:
        source, ru, kind, gender, notes = accepted
        aggregate["canonicals"][source] = ru
        aggregate.setdefault("canonical_kinds", {})[source] = kind
        if kind == "person":
            aggregate["characters"][source] = f"gender={gender};kind=person;{notes}"

    @staticmethod
    def _memory_from_data(data: dict[str, Any]) -> BookMemory:
        memory = SourceOnlyBookBibleBuilder._memory_from_data(data)
        canon = dict(data.get("canonicals") or {})
        kinds = dict(data.get("canonical_kinds") or {})
        for name, ru in canon.items():
            kind = str(kinds.get(name) or "other").casefold()
            if kind == "person":
                desc = str(memory.characters.get(name) or "")
                if "kind=" not in desc:
                    memory.characters[name] = desc + ";kind=person"
            elif kind in {"place", "institution"}:
                memory.characters[name] = f"ru={_norm(ru)};gender=unknown;kind={kind};role=entity_canon"
            else:
                # Keep low-confidence/other proper tokens available to the translator
                # through glossary, but do not make them a hard release invariant.
                memory.characters[name] = f"ru={_norm(ru)};gender=unknown;kind=other;role=proper_name"
        return memory

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text("utf-8"))
            except Exception:
                data = {}
            if data.get("general_release_schema") != _GENERAL_CACHE_MARKER:
                try:
                    self.cache_path.unlink()
                except OSError:
                    pass

        memory, stats = super().build(segments)
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
            if isinstance(data, dict):
                data["general_release_schema"] = _GENERAL_CACHE_MARKER
                self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass
        stats = dict(stats)
        stats["general_release_schema"] = _GENERAL_CACHE_MARKER
        return memory, stats


class FinalDialogueDiscourseGuard(DialogueDiscourseGuard):
    """Book-agnostic Russian dialogue normalizer preserving nested guillemets."""

    @staticmethod
    def _drop_unmatched_closing_guillemets(value: str) -> str:
        depth = 0
        out: list[str] = []
        for ch in value:
            if ch == "«":
                depth += 1
                out.append(ch)
            elif ch == "»":
                if depth > 0:
                    depth -= 1
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _normalize(segment: Segment, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        source = str(segment.text or "")
        value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r",\s*,+", ",", value)

        if source_has_dialogue(source):
            if re.match(r"^\s*[\"'“‘]", source):
                if re.match(r"^\s*[\"'«]", value):
                    value = re.sub(r"^\s*[\"'«]\s*", "— ", value, count=1)
                elif not re.match(r"^\s*—", value):
                    value = "— " + value.lstrip()
            value = re.sub(r"—\s*[\"']\s*(?=[А-Яа-яЁё])", "— ", value)
            value = re.sub(r"(?<=[.!?…])\s*[\"']\s*(?=[А-ЯЁ])", " — ", value)
            value = re.sub(r"(?<=[А-Яа-яЁё0-9])['\"](?=[,!?….])", "", value)
            value = re.sub(r"([,!?….])\s*['\"](?=\s*(?:—|-|$))", r"\1", value)
            value = re.sub(r"['\"]\s*$", "", value)
            value = FinalDialogueDiscourseGuard._drop_unmatched_closing_guillemets(value)
        else:
            value = re.sub(r"['\"]([^'\"\n]{1,240})['\"]", r"«\1»", value)

        value = re.sub(r"\s+([,.!?…])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        return re.sub(r",\s*,+", ",", value)


class FinalV10QualityQA(HardenedV10QualityQA):
    """General release QA: language invariants + dynamic per-book memory only."""

    def __init__(self, *, demote_clause_order: bool = False):
        super().__init__()
        self.demote_clause_order = bool(demote_clause_order)

    @staticmethod
    def _mixed_script_tokens(target: str) -> list[str]:
        tokens = HardenedV10QualityQA._mixed_script_tokens(target)
        return [token for token in tokens if not re.fullmatch(r"[A-Za-z]-[А-Яа-яЁё-]+", token)]

    @classmethod
    def _clean_quantity(cls, source: str, target: str) -> dict[str, Any]:
        quantity = dict(super()._clean_quantity(source, target))
        low = str(target or "").casefold().replace("ё", "е")
        suppress: set[int] = set()

        hundred_patterns = {
            100: (r"\bсто\b", r"\bсотн\w*\b", r"\bстопроцент\w*\b"),
            200: (r"\bдвест\w*\b",),
            300: (r"\bтрист\w*\b",),
            400: (r"\bчетырест\w*\b",),
            500: (r"\bпятьсот\b", r"\bпятисот\b"),
            600: (r"\bшестьсот\b", r"\bшестисот\b"),
            700: (r"\bсемьсот\b", r"\bсемисот\b"),
            800: (r"\bвосемьсот\b", r"\bвосьмисот\b"),
            900: (r"\bдевятьсот\b", r"\bдевятисот\b"),
        }
        for value, patterns in hundred_patterns.items():
            if any(re.search(pattern, low) for pattern in patterns):
                suppress.add(value)

        # Generic English fractional constructions can be misread as sums by the
        # legacy numeric parser. Suppress only when Russian visibly preserves the fraction.
        fractions = {
            "half": (2, r"половин|втор"),
            "third": (3, r"трет"),
            "quarter": (4, r"четверт"),
            "fifth": (5, r"пят"),
            "sixth": (6, r"шест"),
            "seventh": (7, r"седьм"),
            "eighth": (8, r"восьм"),
            "ninth": (9, r"девят"),
            "tenth": (10, r"десят"),
            "sixteenth": (16, r"шестнадцат"),
            "thirty-second": (32, r"тридцат\w*втор"),
            "sixty-fourth": (64, r"шестьдесят\w*четверт"),
        }
        for word, (denominator, ru_stem) in fractions.items():
            if re.search(rf"\bone\s+{re.escape(word)}\b", source, re.I):
                if re.search(rf"\bодн\w*\s+{ru_stem}\w*\b", low) or re.search(rf"\b1\s*/\s*{denominator}\b", low):
                    suppress.add(1 + denominator)

        # Generic approximate range N, M hundred -> N00-M00.
        small = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
        match = re.search(r"\b(" + "|".join(small) + r")\s*,\s*(" + "|".join(small) + r")\s+hundred\b", source, re.I)
        if match:
            left = small[match.group(1).casefold()] * 100
            right = small[match.group(2).casefold()] * 100
            if left in hundred_patterns and right in hundred_patterns:
                if any(re.search(p, low) for p in hundred_patterns[left]) and any(re.search(p, low) for p in hundred_patterns[right]):
                    suppress.update({small[match.group(1).casefold()], small[match.group(2).casefold()], left, right})

        quantity["base_missing"] = [value for value in quantity.get("base_missing") or [] if value not in suppress]
        quantity["missing_mentions"] = [
            row for row in quantity.get("missing_mentions") or [] if row.get("value") not in suppress
        ]
        quantity["ok"] = not quantity["base_missing"] and not quantity["missing_mentions"] and not quantity.get("numbered_choice_missing")
        return quantity

    @classmethod
    def _name_canon_issues(cls, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        low_target = cls._strip_marks(target).casefold().replace("ё", "е")
        out: list[V10Issue] = []
        for name, desc in memory.characters.items():
            if not re.search(rf"\b{re.escape(str(name))}\b", source, re.I):
                continue
            kind_match = re.search(r"kind=(person|place|institution|other)", str(desc or ""), re.I)
            kind = kind_match.group(1).casefold() if kind_match else "person" if re.search(r"gender=(male|female)", str(desc), re.I) else "other"
            if kind not in {"person", "place", "institution"}:
                continue
            ru_match = re.search(r"(?:^|;)ru=([^;]+)", str(desc or ""), re.I)
            canon = _norm(ru_match.group(1) if ru_match else memory.glossary.get(name, ""))
            stem = _canon_stem(canon)
            if stem and stem not in low_target:
                out.append(V10Issue(
                    segment.id,
                    "name_canon",
                    "local",
                    "hard",
                    f"source entity {name!r} must preserve the book-wide Russian canon {canon!r} (inflection allowed)",
                ))
        return out

    @staticmethod
    def _book_term_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        low_target = str(target or "").casefold().replace("ё", "е")
        out: list[V10Issue] = []
        for en, ru in memory.glossary.items():
            en_s = str(en or "").strip()
            if not en_s or en_s[:1].isupper() or (" " not in en_s and "-" not in en_s):
                continue
            if not re.search(rf"\b{re.escape(en_s)}\b", source, re.I):
                continue
            stem = _canon_stem(str(ru or ""))
            if stem and stem not in low_target:
                out.append(V10Issue(
                    segment.id,
                    "book_term_canon",
                    "local",
                    "hard",
                    f"source term {en_s!r} must preserve high-confidence book glossary term {ru!r}",
                ))
        return out

    @staticmethod
    def _material_contrast_issues(segment: Segment, target: str) -> list[V10Issue]:
        source = str(segment.text or "")
        out: list[V10Issue] = []
        names = "|".join(_MATERIAL_STEMS)
        for match in re.finditer(rf"\bnot\s+({names})\s+but\s+({names})\b", source, re.I):
            left = match.group(1).casefold()
            right = match.group(2).casefold()
            if not _ru_has_stem(target, _MATERIAL_STEMS[left]) or not _ru_has_stem(target, _MATERIAL_STEMS[right]):
                out.append(V10Issue(
                    segment.id,
                    "material",
                    "local",
                    "hard",
                    f"source material contrast 'not {left} but {right}' must preserve both materials",
                ))
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        issues = list(super().scan_segment(segment, target, memory))

        # Remove legacy title-specific lexical patches from the old hardened layer.
        # General invariants are reintroduced below from dynamic book memory.
        filtered: list[V10Issue] = []
        for issue in issues:
            if issue.code in {"ethnonym_canon", "hunting_collocation", "technical_denotation"}:
                continue
            if issue.code == "material" and not str(issue.reason).lstrip().startswith("{"):
                continue
            filtered.append(issue)
        issues = filtered

        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")
        if re.search(r"\b(?:half[- ]inch|half an inch)\b", source, re.I) and re.search(r"\bполдюйм\w*\b", low):
            issues = [issue for issue in issues if issue.code != "half_inch"]

        issues.extend(self._book_term_issues(segment, target, memory))
        issues.extend(self._material_contrast_issues(segment, target))

        if self.demote_clause_order:
            issues = [
                V10Issue(row.id, row.code, row.mode, "soft", row.reason)
                if row.code == "clause_order" and row.severity == "hard"
                else row
                for row in issues
            ]
        unique = {(row.code, row.mode, row.severity, row.reason): row for row in issues}
        return list(unique.values())


class FinalDeepSeekSemanticSpecialist(HardenedDeepSeekSemanticSpecialist):
    """Route only general invariants and dynamic book-memory violations."""

    _GENERAL_PROVEN = frozenset({
        "latin_leak",
        "question",
        "direction_relation",
        "half_inch",
        "material",
        "name_canon",
        "book_term_canon",
        "dialogue_typography",
        "idiom_last_but_one",
        "character_gender",
    })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        allowed = PROVEN_DEEPSEEK_CODES | FinalDeepSeekSemanticSpecialist._GENERAL_PROVEN
        out: dict[str, set[str]] = {}
        for issue in issues:
            if issue.severity == "hard" and issue.code in allowed:
                out.setdefault(issue.id, set()).add(issue.code)
        return out


class ResidualHardDeepSeekRepair:
    """One fail-closed rescue call over remaining hard rows, independent of book title."""

    def __init__(self, provider: Any, qa: FinalV10QualityQA, max_segments: int = 12):
        self.provider = provider
        self.qa = qa
        self.max_segments = max(1, int(max_segments))
        self.stats: dict[str, Any] = {
            "calls": 0,
            "selected": 0,
            "selected_ids": [],
            "accepted": 0,
            "changed_ids": [],
            "rejected": [],
        }

    def repair(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[str]:
        hard_by_id: dict[str, list[V10Issue]] = {}
        for issue in issues:
            if issue.severity == "hard":
                hard_by_id.setdefault(issue.id, []).append(issue)
        if not hard_by_id:
            return []
        selected = [segment for segment in targets if segment.id in hard_by_id][: self.max_segments]
        if not selected:
            return []
        self.stats["selected"] = len(selected)
        self.stats["selected_ids"] = [segment.id for segment in selected]

        items = [
            {
                "id": segment.id,
                "source_en": segment.text,
                "current_ru": translated.get(segment.id, ""),
                "hard_defects": [f"{row.code}: {row.reason}" for row in hard_by_id[segment.id]],
            }
            for segment in selected
        ]
        system = """Final fail-closed EN→RU literary repair. Every input row still fails a deterministic publication gate.
Return a COMPLETE corrected Russian translation for every id. Fix every listed hard_defect while preserving every source fact,
quantity, speaker attribution, book-wide name canon and high-confidence glossary term. If current_ru is mostly English, translate
source_en from scratch. Keep Russian dialogue typography and nested quotes balanced. Standard one-letter engineering notation
such as V-образный is allowed. Do not add commentary. ONLY JSON
{"items":[{"id":"...","corrected_ru":"..."}]}"""
        try:
            raw = self.provider.complete(system, json.dumps({"items": items}, ensure_ascii=False), temperature=0.0)
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            self.stats["rejected"].append({"reason": f"call_error:{type(exc).__name__}"})
            return []

        parsed = {str(row.get("id") or ""): row for row in (obj.get("items") or []) if isinstance(row, dict)}
        changed: list[str] = []
        for segment in selected:
            sid = segment.id
            candidate = _norm((parsed.get(sid) or {}).get("corrected_ru") or "")
            if not candidate:
                self.stats["rejected"].append({"id": sid, "reason": "empty_or_missing"})
                continue
            before = self.qa.scan_segment(segment, str(translated.get(sid) or ""), memory)
            after = self.qa.scan_segment(segment, candidate, memory)
            before_hard = sum(row.severity == "hard" for row in before)
            after_hard = sum(row.severity == "hard" for row in after)
            if after_hard >= before_hard:
                self.stats["rejected"].append({"id": sid, "reason": "hard_not_reduced", "before": before_hard, "after": after_hard})
                continue
            translated[sid] = candidate
            changed.append(sid)

        self.stats["accepted"] = len(changed)
        self.stats["changed_ids"] = changed
        return changed
