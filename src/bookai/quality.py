from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .models import BookMemory, Segment


_LATIN_WORD = re.compile(r"\b[A-Za-z][A-Za-z'’-]{2,}\b")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?")
_FORBIDDEN_SCRIPTS = {
    "cjk": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "hangul": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
}
_SERVICE_LEAK = re.compile(
    r"(?:так\s+и\s+оставь|оставь\s+без\s+перевода|глоссари|служебн(?:ая|ые)\s+инструкц|"
    r"переведи\s+только|не\s+переводи\s+контекст|context[_ ]only|\btargets?\b|translation\s+note)",
    re.I,
)
_FEMALE_VERBS = (
    "сказала", "ответила", "повторила", "спросила", "настояла", "подумала", "решила",
    "повернулась", "посмотрела", "усмехнулась", "улыбнулась", "кивнула", "вздохнула",
    "пожала", "заметила", "продолжила", "остановилась", "встала", "села",
)
_MALE_VERBS = tuple(
    word for word in (
        "сказал", "ответил", "повторил", "спросил", "настоял", "подумал", "решил",
        "повернулся", "посмотрел", "усмехнулся", "улыбнулся", "кивнул", "вздохнул",
        "пожал", "заметил", "продолжил", "остановился", "встал", "сел",
    )
)


@dataclass(slots=True, frozen=True)
class QualityIssue:
    id: str
    severity: str
    code: str
    reason: str


def is_heading(segment: Segment) -> bool:
    loc = segment.locator.lower()
    return "/title" in loc or "/subtitle" in loc


def _numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", ".") for m in _NUMBER.finditer(text)]


def _unexpected_scripts(original: str, candidate: str) -> list[str]:
    bad: list[str] = []
    for name, pattern in _FORBIDDEN_SCRIPTS.items():
        if pattern.search(candidate) and not pattern.search(original):
            bad.append(name)
    return bad


def _balanced(text: str, left: str, right: str) -> bool:
    return text.count(left) == text.count(right)


def _has_repetition_loop(text: str) -> bool:
    sentences = [
        " ".join(row.casefold().split())
        for row in re.split(r"(?<=[.!?…])\s+|\n+", text)
        if len(" ".join(row.split())) >= 28
    ]
    counts = Counter(sentences)
    return any(count >= 3 for count in counts.values())


def _character_gender_mismatch(original: str, candidate: str, memory: BookMemory) -> str | None:
    original_lower = original.casefold()
    for source_name, description in memory.characters.items():
        if not source_name or source_name.casefold() not in original_lower:
            continue
        desc = str(description or "")
        gender_match = re.search(r"\bgender=(male|female)\b", desc, flags=re.I)
        ru_match = re.search(r"\bru=([^;]+)", desc, flags=re.I)
        if not gender_match or not ru_match:
            continue
        gender = gender_match.group(1).casefold()
        ru_name = ru_match.group(1).strip()
        wrong = _FEMALE_VERBS if gender == "male" else _MALE_VERBS
        verbs = "|".join(map(re.escape, wrong))
        name = re.escape(ru_name)
        # Restrict to direct name/verb adjacency to avoid blaming another character.
        if re.search(rf"\b{name}\b\s+(?:\w+\s+){{0,2}}(?:{verbs})\b", candidate, flags=re.I):
            return f"gender drift near {ru_name}: expected {gender}"
        if re.search(rf"\b(?:{verbs})\b\s+\b{name}\b", candidate, flags=re.I):
            return f"gender drift near {ru_name}: expected {gender}"
    return None


def candidate_issues(segment: Segment, candidate: str, memory: BookMemory | None = None) -> list[QualityIssue]:
    original = segment.text.strip()
    candidate = (candidate or "").strip()
    out: list[QualityIssue] = []

    def add(severity: str, code: str, reason: str) -> None:
        out.append(QualityIssue(segment.id, severity, code, reason))

    if not candidate:
        add("hard", "empty", "empty translation")
        return out
    if candidate == original and _LATIN_WORD.search(original):
        add("hard", "unchanged", "translation is identical to English source")

    scripts = _unexpected_scripts(original, candidate)
    if scripts:
        add("hard", "unexpected_script", "unexpected writing system: " + ", ".join(scripts))

    latin_words = _LATIN_WORD.findall(candidate)
    cyr = len(_CYRILLIC.findall(candidate))
    if len(latin_words) >= 5 and cyr < max(8, sum(map(len, latin_words)) // 2):
        add("hard", "english_leftover", f"too much English remains ({len(latin_words)} words)")

    if _numbers(original) != _numbers(candidate):
        add("hard", "numbers", "numbers changed, disappeared, or were added")

    source_len = len(original)
    ratio = len(candidate) / max(1, source_len)
    if source_len <= 60:
        if len(candidate) > max(120, source_len * 6):
            add("hard", "short_expansion", f"short source expanded from {source_len} to {len(candidate)} chars")
        elif len(candidate) > max(80, source_len * 4):
            add("medium", "short_expansion", f"short source unusually expanded from {source_len} to {len(candidate)} chars")
    elif source_len < 80 and ratio > 1.85:
        add("hard", "too_long", f"translation/source length ratio is {ratio:.2f}")
    elif source_len >= 80:
        if ratio < 0.48:
            add("hard", "too_short", f"translation/source length ratio is {ratio:.2f}")
        elif ratio > 1.65:
            add("hard", "too_long", f"translation/source length ratio is {ratio:.2f}")
        elif ratio < 0.58 or ratio > 1.35:
            add("medium", "length", f"unusual translation/source length ratio is {ratio:.2f}")

    if _SERVICE_LEAK.search(candidate) and not _SERVICE_LEAK.search(original):
        add("hard", "service_leak", "translator instruction or service text leaked into literary output")

    if len(candidate) >= 180 and _has_repetition_loop(candidate):
        add("hard", "repetition_loop", "translation contains a repeated sentence loop")

    for left, right, label in (("(", ")", "parentheses"), ("[", "]", "brackets"), ("«", "»", "Russian quotes")):
        if not _balanced(candidate, left, right):
            add("medium", "unbalanced_punctuation", f"unbalanced {label}")
            break

    if is_heading(segment):
        if "\n" in candidate:
            add("hard", "heading_multiline", "chapter/title translation became multiline")
        if len(candidate) > max(80, len(original) * 3):
            add("hard", "heading_expanded", "heading expanded into body-like prose")
        if re.match(r"^\s*Chapter\b", original, flags=re.I) and not re.match(r"^\s*Глава\b", candidate, flags=re.I):
            add("hard", "chapter_heading", "English Chapter heading was not translated as a Russian chapter heading")

    if memory is not None:
        lower_original = original.casefold()
        lower_candidate = candidate.casefold()
        for src, preferred in memory.glossary.items():
            if not src or not preferred:
                continue
            if src.casefold() in lower_original and preferred.casefold() not in lower_candidate:
                add("medium", "glossary", f"preferred term missing: {src} → {preferred}")
                break
        mismatch = _character_gender_mismatch(original, candidate, memory)
        if mismatch:
            add("hard", "character_gender", mismatch)

    return out


def batch_issues(segments: list[Segment], translations: dict[str, str], memory: BookMemory | None = None) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for segment in segments:
        if segment.id not in translations:
            issues.append(QualityIssue(segment.id, "hard", "missing_id", "model omitted required segment id"))
            continue
        issues.extend(candidate_issues(segment, translations[segment.id], memory))
    return issues


def _single_segment_wrapper(obj: dict, sid: str) -> str | None:
    for key in ("translation", "translated_text", "text", "result", "output"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = obj.get("data")
    if isinstance(nested, dict):
        for key in ("translation", "translated_text", "text", "result", "output"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if len(obj) == 1:
        only = next(iter(obj.values()))
        if isinstance(only, str) and only.strip():
            return only.strip()
    return None


def _unwrap_exact_id_envelope(obj: dict, expected_set: set[str]) -> dict:
    current = obj
    wrapper_keys = {"type", "data", "result", "output", "json", "response", "content"}
    for _ in range(3):
        actual = {str(k) for k in current}
        if actual == expected_set:
            return current
        advanced = False
        for key in ("data", "result", "output", "json", "response", "content"):
            nested = current.get(key)
            if not isinstance(nested, dict):
                continue
            if not actual.issubset(wrapper_keys):
                continue
            nested_keys = {str(k) for k in nested}
            if nested_keys == expected_set or nested_keys.issubset(wrapper_keys):
                current = nested
                advanced = True
                break
        if not advanced:
            break
    return obj


def assert_exact_ids(segments: list[Segment], obj: object, role: str) -> dict[str, str]:
    if not isinstance(obj, dict):
        raise ValueError(f"{role} returned non-object JSON")
    expected = [s.id for s in segments]
    expected_set = set(expected)
    obj = _unwrap_exact_id_envelope(obj, expected_set)
    actual_set = {str(k) for k in obj}
    if len(expected) == 1 and expected[0] not in actual_set:
        recovered = _single_segment_wrapper(obj, expected[0])
        if recovered is not None:
            return {expected[0]: recovered}
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing or extra:
        raise ValueError(f"{role} id contract violation: missing={sorted(missing)[:12]} extra={sorted(extra)[:12]}")
    out: dict[str, str] = {}
    for sid in expected:
        value = obj[sid]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role} returned empty/non-string value for {sid}")
        out[sid] = value.strip()
    return out


def hard_ids(issues: list[QualityIssue]) -> set[str]:
    return {issue.id for issue in issues if issue.severity == "hard"}


def medium_or_hard_ids(issues: list[QualityIssue]) -> set[str]:
    return {issue.id for issue in issues if issue.severity in {"medium", "hard"}}
