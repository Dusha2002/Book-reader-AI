from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .models import BookMemory, Segment
from .pipeline import _chapter_groups, _should_translate
from .v10 import V10Issue, _norm
from .v10_deepseek import DeepSeekSemanticSpecialist, PROVEN_DEEPSEEK_CODES
from .v10_dialogue import DialogueDiscourseGuard, source_has_dialogue
from .v10_local_repair import GigaLocalRewriter
from .v10_name_canon import V9ADSourceOnlyBookBibleBuilder
from .v10_quality import V10QualityQA
from .v10_quantity import compare_quantity_fidelity_v2
from .v10_transport import RobustTaggedPrimaryTransport


_HARDENING_CACHE_MARKER = "v10-release-hardening-1"
_NUMBERED_CHAPTER_RE = re.compile(
    r"^chapter\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
    re.I,
)
_CHAPTER_ORDINAL_RU = {
    "one": "первая",
    "two": "вторая",
    "three": "третья",
    "four": "четвёртая",
    "five": "пятая",
    "six": "шестая",
    "seven": "седьмая",
    "eight": "восьмая",
    "nine": "девятая",
    "ten": "десятая",
    "eleven": "одиннадцатая",
    "twelve": "двенадцатая",
    "thirteen": "тринадцатая",
    "fourteen": "четырнадцатая",
    "fifteen": "пятнадцатая",
    "sixteen": "шестнадцатая",
    "seventeen": "семнадцатая",
    "eighteen": "восемнадцатая",
    "nineteen": "девятнадцатая",
    "twenty": "двадцатая",
}
_RU_CARDINAL_EVIDENCE: dict[int, tuple[str, ...]] = {
    2: ("два", "две", "двух", "двум", "двумя"),
    3: ("три", "трех", "трёх", "трем", "трём", "тремя"),
    4: ("четыре", "четырех", "четырёх", "четырем", "четырём", "четырьмя"),
    5: ("пять", "пяти", "пятью"),
    6: ("шесть", "шести", "шестью"),
    7: ("семь", "семи", "семью"),
    8: ("восемь", "восьми", "восемью"),
    9: ("девять", "девяти", "девятью"),
    10: ("десять", "десяти"),
    11: ("одиннадцать", "одиннадцати"),
    12: ("двенадцать", "двенадцати"),
    13: ("тринадцать", "тринадцати"),
    14: ("четырнадцать", "четырнадцати"),
    15: ("пятнадцать", "пятнадцати"),
    16: ("шестнадцать", "шестнадцати"),
    17: ("семнадцать", "семнадцати"),
    18: ("восемнадцать", "восемнадцати"),
    19: ("девятнадцать", "девятнадцати"),
    20: ("двадцать", "двадцати"),
}
_EN_SIMPLE_NUMBER = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _norm_label(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _is_numbered_chapter(value: str) -> bool:
    return bool(_NUMBERED_CHAPTER_RE.match(_norm_label(value)))


def select_numbered_chapter(document: Any, chapter_name: str) -> tuple[list[Segment], str, list[Segment], dict[str, Any]]:
    """Select a complete numbered chapter, keeping internal subheadings inside it."""
    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    wanted = _norm_label(chapter_name)
    exact_indexes = [i for i, (name, _rows) in enumerate(groups) if _norm_label(name) == wanted]
    if not exact_indexes:
        exact_indexes = [i for i, (name, _rows) in enumerate(groups) if wanted in _norm_label(name)]
    if len(exact_indexes) != 1:
        raise RuntimeError(
            f"Expected one chapter matching {chapter_name!r}; found {len(exact_indexes)}; "
            f"available={[name for name, _ in groups]}"
        )

    start = exact_indexes[0]
    if not _is_numbered_chapter(groups[start][0]):
        raise RuntimeError(f"Requested chapter {groups[start][0]!r} is not a numbered Chapter N boundary")
    end = len(groups)
    next_chapter = ""
    for i in range(start + 1, len(groups)):
        if _is_numbered_chapter(groups[i][0]):
            end = i
            next_chapter = groups[i][0]
            break

    selected_groups = groups[start:end]
    targets = [segment for _name, rows in selected_groups for segment in rows]
    if not targets:
        raise RuntimeError(f"Chapter {chapter_name!r} selected no translatable segments")

    position = {segment.id: i for i, segment in enumerate(all_targets)}
    boundary_complete = True
    if end < len(groups):
        next_rows = groups[end][1]
        if not next_rows:
            boundary_complete = False
        else:
            boundary_complete = position[targets[-1].id] + 1 == position[next_rows[0].id]
    if not boundary_complete:
        raise RuntimeError(f"Chapter {chapter_name!r} is not contiguous with the next numbered chapter boundary")

    meta = {
        "requested": chapter_name,
        "matched": groups[start][0],
        "included_groups": [name for name, _rows in selected_groups],
        "next_numbered_chapter": next_chapter or None,
        "boundary_complete": boundary_complete,
        "segments": len(targets),
    }
    return all_targets, groups[start][0], targets, meta


def russian_numbered_chapter_heading(source_text: str) -> str:
    match = re.fullmatch(r"\s*chapter\s+([A-Za-z]+)\s*", str(source_text or ""), re.I)
    if not match:
        return ""
    ordinal = _CHAPTER_ORDINAL_RU.get(match.group(1).casefold())
    return f"Глава {ordinal}" if ordinal else ""


class HardenedRobustTaggedPrimaryTransport(RobustTaggedPrimaryTransport):
    name = "gigachat-3-lightning-v10-tagged-release-hardened"

    def _deterministic_heading(self, segment: Segment) -> str:
        heading = russian_numbered_chapter_heading(segment.text)
        if heading:
            return heading
        return super()._deterministic_heading(segment)


class HardenedBookBibleBuilder(V9ADSourceOnlyBookBibleBuilder):
    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        records = [dict(row) for row in V9ADSourceOnlyBookBibleBuilder._candidate_records(segments)]
        existing = {str(row.get("candidate") or "").casefold() for row in records}
        wanted = (
            "tuck",
            "brass bushing",
            "bronze bushing",
            "steel cable",
            "steel rod",
            "steel rods",
        )
        for term in wanted:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
            matches: list[Segment] = [s for s in segments if pattern.search(str(s.text or ""))]
            if not matches or term.casefold() in existing:
                continue
            records.append({
                "candidate": term,
                "kind_hint": "technical_term",
                "frequency": len(matches),
                "contexts": [
                    {"chapter": str(s.chapter or ""), "text": _norm(s.text)[:520]}
                    for s in matches[:3]
                ],
            })
            existing.add(term.casefold())
        return records

    def _name_batches(self, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
        boosted = []
        for row in rows:
            copy = dict(row)
            copy["frequency"] = max(4, int(copy.get("frequency") or 0))
            boosted.append(copy)
        super()._name_batches(boosted, aggregate)

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text("utf-8"))
            except Exception:
                data = {}
            if data.get("hardening") != _HARDENING_CACHE_MARKER:
                try:
                    self.cache_path.unlink()
                except OSError:
                    pass
        memory, stats = super().build(segments)
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
            if isinstance(data, dict) and data.get("hardening") != _HARDENING_CACHE_MARKER:
                data["hardening"] = _HARDENING_CACHE_MARKER
                self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass
        stats = dict(stats)
        stats["hardening"] = _HARDENING_CACHE_MARKER
        return memory, stats


class HardenedDialogueDiscourseGuard(DialogueDiscourseGuard):
    @staticmethod
    def _normalize(segment: Segment, text: str) -> str:
        value = DialogueDiscourseGuard._normalize(segment, text)
        if not value or not source_has_dialogue(segment.text):
            return value
        value = re.sub(r"\s+[–-]\s+", " — ", value)
        value = re.sub(r"([—–-])\s*[«\"'](?=[А-Яа-яЁё])", "— ", value)
        value = re.sub(r"([,!?….])\s*['\"»](?=\s*[А-Яа-яЁё])", r"\1", value)
        value = re.sub(r"([,!?….])\s*['\"»](?=\s*—)", r"\1", value)
        value = re.sub(r"\s+([,.!?…])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        return value


class HardenedV10QualityQA(V10QualityQA):
    @staticmethod
    def _visible_ru_value(value: int | float, target: str) -> bool:
        if not isinstance(value, int):
            return False
        words = _RU_CARDINAL_EVIDENCE.get(value)
        if not words:
            return False
        low = str(target or "").casefold().replace("ё", "е")
        return any(re.search(rf"\b{re.escape(word.replace('ё', 'е'))}\b", low) for word in words)

    @staticmethod
    def _spurious_and_sums(source: str) -> set[int]:
        names = "|".join(sorted(_EN_SIMPLE_NUMBER, key=len, reverse=True))
        out: set[int] = set()
        text = str(source or "")
        for match in re.finditer(rf"\b({names})\s+and\s+({names})\b", text, re.I):
            left = _EN_SIMPLE_NUMBER[match.group(1).casefold()]
            right = _EN_SIMPLE_NUMBER[match.group(2).casefold()]
            total = left + right
            explicit_word = next((word for word, value in _EN_SIMPLE_NUMBER.items() if value == total), None)
            if re.search(rf"\b{total}\b", text) or (explicit_word and re.search(rf"\b{explicit_word}\b", text, re.I)):
                continue
            out.add(total)
        return out

    @classmethod
    def _clean_quantity(cls, source: str, target: str) -> dict[str, Any]:
        quantity = dict(compare_quantity_fidelity_v2(source, target))
        base_missing = list(quantity.get("base_missing") or [])
        spurious = cls._spurious_and_sums(source)
        cleaned_missing = [
            value for value in base_missing
            if value not in spurious and not cls._visible_ru_value(value, target)
        ]
        obligations = list(quantity.get("obligations") or [])
        obligation_values = {int(row.get("value")) for row in obligations if row.get("value") is not None}
        missing_mentions = []
        for row in quantity.get("missing_mentions") or []:
            value = row.get("value")
            if value in spurious:
                continue
            if isinstance(value, int) and cls._visible_ru_value(value, target) and value not in obligation_values:
                continue
            if value not in cleaned_missing and value not in obligation_values:
                continue
            missing_mentions.append(row)
        quantity["base_missing"] = cleaned_missing
        quantity["missing_mentions"] = missing_mentions
        quantity["ok"] = not cleaned_missing and not missing_mentions and not quantity.get("numbered_choice_missing")
        return quantity

    @staticmethod
    def _strip_marks(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", str(value or ""))
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

    @classmethod
    def _name_canon_issues(cls, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        low_target = cls._strip_marks(target).casefold().replace("ё", "е")
        out: list[V10Issue] = []
        for name, desc in memory.characters.items():
            if not re.search(rf"\b{re.escape(str(name))}\b", source, re.I):
                continue
            match = re.search(r"(?:^|;)ru=([^;]+)", str(desc or ""), re.I)
            canon = _norm(match.group(1) if match else memory.glossary.get(name, ""))
            if not canon:
                continue
            words = re.findall(r"[А-Яа-яЁё]+", cls._strip_marks(canon))
            if not words:
                continue
            first = words[0].casefold().replace("ё", "е")
            stem_len = max(3, min(5, len(first) - 1 if len(first) > 3 else len(first)))
            stem = first[:stem_len]
            if stem and stem not in low_target:
                out.append(V10Issue(
                    segment.id,
                    "name_canon",
                    "local",
                    "hard",
                    f"source name {name!r} must use book-wide Russian canon {canon!r} (inflection allowed)",
                ))
        return out

    @staticmethod
    def _dialogue_typography_issues(segment: Segment, target: str) -> list[V10Issue]:
        if not source_has_dialogue(segment.text):
            return []
        value = str(target or "")
        reasons: list[str] = []
        if re.search(r"['\"]", value):
            reasons.append("ASCII quote/apostrophe remains in Russian direct speech")
        if value.count("«") != value.count("»"):
            reasons.append("unbalanced Russian guillemets")
        if not reasons:
            return []
        return [V10Issue(segment.id, "dialogue_typography", "local", "hard", "; ".join(reasons))]

    @staticmethod
    def _has_up(text: str) -> bool:
        return bool(re.search(r"\b(?:up\s+the\s+(?:stairs|steps|slope|hill|mountain)|(?:trudged|walked|went|ran|climbed|came)\s+up)\b", text, re.I))

    @staticmethod
    def _has_down(text: str) -> bool:
        return bool(re.search(r"\b(?:down\s+the\s+(?:stairs|steps|slope|hill|mountain)|(?:trudged|walked|went|ran|climbed|came)\s+down)\b", text, re.I))

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        ru = str(target or "")
        low = ru.casefold().replace("ё", "е")
        out = list(super().scan_segment(segment, target, memory))

        out = [row for row in out if row.code not in {"numeric", "quantity_obligation", "numbered_choice"}]
        quantity = self._clean_quantity(source, ru)
        if quantity.get("base_missing"):
            out.append(V10Issue(
                segment.id, "numeric", "local", "hard",
                f"missing source numeric values {quantity.get('base_missing')}",
            ))
        if quantity.get("missing_mentions"):
            out.append(V10Issue(
                segment.id, "quantity_obligation", "local", "hard",
                f"quantity mentions lost/substituted: {quantity.get('missing_mentions')}",
            ))
        if quantity.get("numbered_choice_missing"):
            out.append(V10Issue(
                segment.id, "numbered_choice", "local", "hard",
                f"numbered choice/label lost: {quantity.get('numbered_choice_missing')}",
            ))

        if re.search(r"\bat the back of (?:his|her|their|my|your|the) mind\b", source, re.I):
            out = [row for row in out if row.code != "order"]

        if self._has_up(source) and self._has_down(source):
            ru_up = bool(re.search(r"\b(?:поднима\w*|взош[её]л|вверх)\b", low))
            ru_down = bool(re.search(r"\b(?:спуска\w*|сош[её]л|вниз)\b", low))
            if ru_up and ru_down:
                out = [row for row in out if row.code != "direction_relation"]

        if re.search(r"\b(?:half[- ]inch|half an inch)\b", source, re.I):
            half_ok = bool(re.search(
                r"(?:\bполудюйм\w*\b|\bполовин\w*\s+дюйм\w*\b|\b(?:0[,.]5|1/2|½)\s*дюйм\w*\b|\b12[,.]7\s*мм\b)",
                low,
                re.I,
            ))
            if not half_ok:
                out.append(V10Issue(
                    segment.id, "half_inch", "local", "hard",
                    "source requires half-inch (1/2 inch, 12.7 mm), but Russian does not preserve the half-inch dimension",
                ))

        if re.search(r"\bbrass\s+bushings?\b", source, re.I) and "латун" not in low:
            out.append(V10Issue(segment.id, "material", "local", "hard", "brass bushing must be латунная втулка, not bronze/copper"))
        if re.search(r"\bbronze\s+bushings?\b", source, re.I) and "бронз" not in low:
            out.append(V10Issue(segment.id, "material", "local", "hard", "bronze bushing must preserve bronze (бронза), not brass/copper"))
        if re.search(r"\bnot\s+brass\s+but\s+bronze\b", source, re.I):
            if "латун" not in low or "бронз" not in low:
                out.append(V10Issue(
                    segment.id, "material", "local", "hard",
                    "source contrast is explicitly not brass but bronze; Russian must preserve both materials and the contrast",
                ))

        if re.search(r"\bsteel\s+cable\b", source, re.I) and not re.search(r"\b(?:стальн\w*\s+)?(?:трос|канат|кабел)\w*\b", low):
            out.append(V10Issue(
                segment.id, "technical_denotation", "semantic", "hard",
                "steel cable in a mechanical bow is a cable/трос, not a thread/нить",
            ))

        if re.search(r"\bMezentines?\b", source, re.I) and not re.search(r"\bмезент[а-я]*\b", low):
            out.append(V10Issue(
                segment.id, "ethnonym_canon", "semantic", "hard",
                "Mezentine demonym must stay tied to Mezentia (Мезент-), not become another nationality",
            ))

        if re.search(r"\blast\s+(?:lesson|class|appointment|meeting|item|thing)\s+but\s+one\b", source, re.I):
            contradictory = bool(re.search(r"\bпоследн[а-я]*\b.{0,12}\b(?:но|и)\b.{0,12}\bпредпослед", low))
            if "предпослед" not in low or contradictory:
                out.append(V10Issue(
                    segment.id, "idiom_last_but_one", "semantic", "hard",
                    "English 'last ... but one' means 'предпоследний', not 'последний, но предпоследний'",
                ))

        if re.search(r"\bround wood\b", source, re.I) and re.search(r"\b(?:boar|hunt|hunting)\b", source, re.I):
            if re.search(r"\bкругл\w*\s+дерев\w*\b", low):
                out.append(V10Issue(
                    segment.id, "hunting_collocation", "semantic", "hard",
                    "hunting-place 'round wood' was calqued as a single round tree; translate it as woodland/grove in context",
                ))

        out.extend(self._name_canon_issues(segment, ru, memory))
        out.extend(self._dialogue_typography_issues(segment, ru))

        unique = {(row.code, row.mode, row.reason): row for row in out}
        return list(unique.values())


class HardenedLocalRewriter(GigaLocalRewriter):
    _CODES = GigaLocalRewriter._CODES | {"half_inch", "name_canon", "dialogue_typography"}
    _SINGLE_FALLBACK_CODES = GigaLocalRewriter._SINGLE_FALLBACK_CODES | {"half_inch", "name_canon", "dialogue_typography"}
    _PRIORITY = {
        **GigaLocalRewriter._PRIORITY,
        "name_canon": 92,
        "half_inch": 91,
        "dialogue_typography": 85,
    }


class HardenedDeepSeekSemanticSpecialist(DeepSeekSemanticSpecialist):
    _EXTRA_PROVEN = frozenset({
        "half_inch",
        "material",
        "name_canon",
        "dialogue_typography",
        "technical_denotation",
        "ethnonym_canon",
        "idiom_last_but_one",
        "hunting_collocation",
    })

    @staticmethod
    def _proven_codes_by_id(issues: list[V10Issue]) -> dict[str, set[str]]:
        allowed = PROVEN_DEEPSEEK_CODES | HardenedDeepSeekSemanticSpecialist._EXTRA_PROVEN
        out: dict[str, set[str]] = {}
        for issue in issues:
            if issue.severity == "hard" and issue.code in allowed:
                out.setdefault(issue.id, set()).add(issue.code)
        return out
