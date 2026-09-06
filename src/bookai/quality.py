from __future__ import annotations

import re
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
    if len(original) >= 80:
        ratio = len(candidate) / max(1, len(original))
        if ratio < 0.42:
            add("hard", "too_short", f"translation/source length ratio is {ratio:.2f}")
        elif ratio > 2.20:
            add("hard", "too_long", f"translation/source length ratio is {ratio:.2f}")
        elif ratio < 0.58 or ratio > 1.75:
            add("medium", "length", f"unusual translation/source length ratio is {ratio:.2f}")
    for left, right, label in (("(", ")", "parentheses"), ("[", "]", "brackets"), ("«", "»", "Russian quotes")):
        if not _balanced(candidate, left, right):
            add("medium", "unbalanced_punctuation", f"unbalanced {label}")
            break
    if is_heading(segment):
        if "\n" in candidate:
            add("hard", "heading_multiline", "chapter/title translation became multiline")
        if len(candidate) > max(180, len(original) * 4):
            add("hard", "heading_expanded", "heading expanded into body-like prose")
    if memory is not None:
        lower_original = original.casefold()
        lower_candidate = candidate.casefold()
        for src, preferred in memory.glossary.items():
            if not src or not preferred:
                continue
            if src.casefold() in lower_original and preferred.casefold() not in lower_candidate:
                add("medium", "glossary", f"preferred term missing: {src} → {preferred}")
                break
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
    # Single-target recovery is safe because there is no possible cross-target mapping.
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
