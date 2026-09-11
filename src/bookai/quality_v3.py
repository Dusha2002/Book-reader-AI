from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BookMemory, Segment
from .quality import QualityIssue, candidate_issues


_CHAPTER = re.compile(r"^\s*Chapter\s+([A-Za-z0-9 -]+)\s*$", re.I)
_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|my|me)\b", re.I)
_LETTER_HEADER = re.compile(r"\b([A-Z][A-Za-z'’-]+)(?:\s+[A-Z][A-Za-z'’-]+){0,2}\s+to\s+([A-Z][A-Za-z'’-]+)", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])(?:[\"'’”)]*)\s+")

_FEMALE_WRONG = re.compile(
    r"\bя\b[^.!?…]{0,55}\b(?:был|уверен|рад|готов|должен|решил|подумал|понял|сказал|видел|знал|хотел|сделал)\b",
    re.I,
)
_MALE_WRONG = re.compile(
    r"\bя\b[^.!?…]{0,55}\b(?:была|уверена|рада|готова|должна|решила|подумала|поняла|сказала|видела|знала|хотела|сделала)\b",
    re.I,
)


def is_chapter_heading_text(segment: Segment) -> bool:
    return bool(_CHAPTER.fullmatch(segment.text.strip()))


def _sentence_count(text: str) -> int:
    rows = [row.strip() for row in _SENTENCE_BREAK.split((text or "").strip()) if row.strip()]
    return max(1, len(rows)) if text.strip() else 0


def infer_active_speaker(
    segment: Segment,
    source_segments: list[Segment] | None,
    memory: BookMemory | None,
    *,
    lookback: int = 18,
) -> tuple[str, str, str] | None:
    """Infer an epistolary first-person speaker from a recent `Name to Name` salutation."""
    if not source_segments or memory is None or not _FIRST_PERSON.search(segment.text):
        return None
    by_id = {row.id: i for i, row in enumerate(source_segments)}
    idx = by_id.get(segment.id)
    if idx is None:
        return None
    start = max(0, idx - lookback)
    for row in reversed(source_segments[start : idx + 1]):
        match = _LETTER_HEADER.search(row.text)
        if not match:
            continue
        header = row.text.casefold()
        for source_name, desc in memory.characters.items():
            if not source_name or source_name.casefold() not in header:
                continue
            gender = re.search(r"\bgender=(male|female)\b", str(desc), flags=re.I)
            ru = re.search(r"\bru=([^;]+)", str(desc), flags=re.I)
            if gender and ru:
                return source_name, gender.group(1).casefold(), ru.group(1).strip()
    return None


def speaker_metadata(
    segments: list[Segment],
    source_segments: list[Segment] | None,
    memory: BookMemory,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in segments:
        speaker = infer_active_speaker(segment, source_segments, memory)
        if not speaker:
            continue
        source_name, gender, ru_name = speaker
        form = "женский" if gender == "female" else "мужской"
        result[segment.id] = (
            f"first-person speaker={source_name}/{ru_name}; gender={gender}; "
            f"для форм прошедшего времени/прилагательных от первого лица используй {form} род"
        )
    return result


def enhanced_candidate_issues(
    segment: Segment,
    candidate: str,
    memory: BookMemory | None = None,
    *,
    source_segments: list[Segment] | None = None,
) -> list[QualityIssue]:
    out = list(candidate_issues(segment, candidate, memory))
    original = segment.text.strip()
    translated = (candidate or "").strip()
    if not translated:
        return out

    existing = {(issue.severity, issue.code) for issue in out}

    def add(severity: str, code: str, reason: str) -> None:
        if (severity, code) not in existing:
            out.append(QualityIssue(segment.id, severity, code, reason))
            existing.add((severity, code))

    if is_chapter_heading_text(segment):
        if not re.match(r"^\s*Глава\s+", translated, flags=re.I):
            add("hard", "chapter_heading_v3", "chapter heading must be translated deterministically")
        if re.search(r"\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen)\b", translated, flags=re.I):
            add("hard", "chapter_heading_english_number", "English chapter number leaked into Russian heading")

    source_len = len(original)
    ratio = len(translated) / max(1, source_len)
    src_sentences = _sentence_count(original)
    dst_sentences = _sentence_count(translated)

    # Russian literary prose is usually somewhat shorter than English. Ratios below
    # ~0.58 on a substantial paragraph are much more likely to be a dropped clause
    # than legitimate compression (Chapter Nine exposed exactly this failure mode).
    if source_len >= 140 and ratio < 0.58:
        add("hard", "semantic_omission", f"substantial paragraph shrank to ratio {ratio:.2f}")
    elif source_len >= 180 and src_sentences >= 3 and dst_sentences <= src_sentences - 2 and ratio < 0.70:
        add("hard", "sentence_loss", f"sentence structure collapsed {src_sentences}→{dst_sentences}")

    # Context leakage often appears as one or two extra translated sentences prepended
    # from the previous paragraph while total character length still looks plausible.
    if source_len >= 100 and src_sentences <= 3 and dst_sentences >= src_sentences + 2 and ratio > 1.02:
        add("hard", "context_leak", f"candidate has unexplained sentence inflation {src_sentences}→{dst_sentences}")

    speaker = infer_active_speaker(segment, source_segments, memory)
    if speaker:
        _, gender, ru_name = speaker
        wrong = _FEMALE_WRONG if gender == "female" else _MALE_WRONG
        if wrong.search(translated):
            add("hard", "speaker_gender", f"first-person morphology conflicts with {ru_name} gender={gender}")

    return out


def enhanced_batch_issues(
    segments: list[Segment],
    translations: dict[str, str],
    memory: BookMemory | None = None,
    *,
    source_segments: list[Segment] | None = None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for segment in segments:
        candidate = translations.get(segment.id)
        if candidate is None:
            issues.append(QualityIssue(segment.id, "hard", "missing_id", "model omitted required segment id"))
            continue
        issues.extend(
            enhanced_candidate_issues(
                segment,
                candidate,
                memory,
                source_segments=source_segments,
            )
        )
    return issues


def is_short_semantic_risk(segment: Segment) -> bool:
    text = segment.text.strip()
    if is_chapter_heading_text(segment) or len(text) > 95:
        return False
    return bool(
        "?" in text
        or re.search(r"['\"‘’“”]", text)
        or re.search(r"\b(?:not|never|no|yes|why|who|what|how|such as|I see)\b", text, flags=re.I)
    )
