from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BookMemory, Segment
from .quality import candidate_issues


_CHAPTER = re.compile(r"^\s*Chapter\s+([A-Za-z0-9 -]+)\s*$", re.I)
_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|my|me)\b", re.I)
_LETTER_HEADER = re.compile(r"\b([A-Z][A-Za-z'’-]+)(?:\s+[A-Z][A-Za-z'’-]+){0,2}\s+to\s+([A-Z][A-Za-z'’-]+)", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])(?:[\"'’”)]*)\s+")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_MIXED_SCRIPT_TOKEN = re.compile(
    r"(?<![\w'’-])(?=[\w'’-]*[A-Za-z])(?=[\w'’-]*[А-Яа-яЁё])[\w'’-]+(?![\w'’-])"
)
_LATIN_RESIDUE = re.compile(r"(?<![A-Za-zА-Яа-яЁё])([A-Za-z][A-Za-z'’-]{2,})(?![A-Za-zА-Яа-яЁё])")
_URL_EMAIL = re.compile(r"(?:https?://\S+|www\.\S+|\b\S+@\S+\.\S+\b)", re.I)

_FEMALE_WRONG = re.compile(
    r"\bя\b[^.!?…]{0,55}\b(?:был|уверен|рад|готов|должен|решил|подумал|понял|сказал|видел|знал|хотел|сделал)\b",
    re.I,
)
_MALE_WRONG = re.compile(
    r"\bя\b[^.!?…]{0,55}\b(?:была|уверена|рада|готова|должна|решила|подумала|поняла|сказала|видела|знала|хотела|сделала)\b",
    re.I,
)

_RU_SUFFIXES = (
    "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "ей", "ой", "ая", "яя",
    "ий", "ый", "ое", "ее", "ов", "ев", "ам", "ям", "ах", "ях", "ом", "ем", "ы", "и",
    "а", "я", "у", "ю", "е",
)


@dataclass(frozen=True)
class QualityIssueV3:
    id: str
    severity: str
    code: str
    reason: str


def is_chapter_heading_text(segment: Segment) -> bool:
    return bool(_CHAPTER.fullmatch(segment.text.strip()))


def _sentence_count(text: str) -> int:
    rows = [row.strip() for row in _SENTENCE_BREAK.split((text or "").strip()) if row.strip()]
    return max(1, len(rows)) if text.strip() else 0


def _source_term_present(term: str, text: str) -> bool:
    term = str(term or "").strip()
    if not term:
        return False
    pattern = r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])"
    return bool(re.search(pattern, text, flags=re.I))


def _ru_stem(word: str) -> str:
    token = word.casefold()
    for suffix in _RU_SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _target_term_present(target: str, candidate: str) -> bool:
    target_words = [_ru_stem(x) for x in re.findall(r"[А-Яа-яЁё-]+", str(target or ""))]
    candidate_words = [_ru_stem(x) for x in re.findall(r"[А-Яа-яЁё-]+", str(candidate or ""))]
    if not target_words:
        return str(target or "").casefold() in str(candidate or "").casefold()
    return all(any(c.startswith(t) or t.startswith(c) for c in candidate_words) for t in target_words)


def _named_source_term(term: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(term or ""))
    return bool(words) and any(word[:1].isupper() for word in words)


def _latin_residue_words(candidate: str) -> list[str]:
    scrubbed = _URL_EMAIL.sub(" ", candidate or "")
    words: list[str] = []
    for word in _LATIN_RESIDUE.findall(scrubbed):
        # Short all-caps abbreviations can be legitimate even in Russian prose.
        if word.isupper() and len(word) <= 8:
            continue
        words.append(word)
    return words


def infer_active_speaker(
    segment: Segment,
    source_segments: list[Segment] | None,
    memory: BookMemory | None,
    *,
    lookback: int = 18,
) -> tuple[str, str, str] | None:
    """Infer an epistolary first-person speaker from the sender before `to`."""
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
        sender = match.group(1).casefold()
        for source_name, desc in memory.characters.items():
            if not source_name or source_name.casefold() != sender:
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
) -> list[QualityIssueV3]:
    out = [
        QualityIssueV3(issue.id, issue.severity, issue.code, issue.reason)
        for issue in candidate_issues(segment, candidate, memory)
    ]
    original = segment.text.strip()
    translated = (candidate or "").strip()
    if not translated:
        return out

    existing = {(issue.severity, issue.code) for issue in out}

    def add(severity: str, code: str, reason: str) -> None:
        if (severity, code) not in existing:
            out.append(QualityIssueV3(segment.id, severity, code, reason))
            existing.add((severity, code))

    if is_chapter_heading_text(segment):
        if not re.match(r"^\s*Глава\s+", translated, flags=re.I):
            add("hard", "chapter_heading_v3", "chapter heading must be translated deterministically")
        if re.search(r"\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen)\b", translated, flags=re.I):
            add("hard", "chapter_heading_english_number", "English chapter number leaked into Russian heading")

    mixed = _MIXED_SCRIPT_TOKEN.findall(translated)
    if mixed:
        add("hard", "mixed_script_token", "mixed Latin/Cyrillic token leaked into Russian: " + ", ".join(mixed[:4]))

    latin_residue = _latin_residue_words(translated)
    if latin_residue and _CYRILLIC.search(translated):
        add("hard", "latin_residue", "untranslated Latin token(s) remain in Russian prose: " + ", ".join(latin_residue[:6]))

    src_questions = original.count("?")
    dst_questions = translated.count("?")
    if src_questions and dst_questions < src_questions:
        add("hard", "question_loss", f"interrogative structure collapsed {src_questions}→{dst_questions}")

    source_len = len(original)
    ratio = len(translated) / max(1, source_len)
    src_sentences = _sentence_count(original)
    dst_sentences = _sentence_count(translated)

    if source_len >= 140 and ratio < 0.58:
        add("hard", "semantic_omission", f"substantial paragraph shrank to ratio {ratio:.2f}")
    elif source_len >= 180 and src_sentences >= 3 and dst_sentences <= src_sentences - 2 and ratio < 0.70:
        add("hard", "sentence_loss", f"sentence structure collapsed {src_sentences}→{dst_sentences}")

    if source_len >= 100 and src_sentences <= 3 and dst_sentences >= src_sentences + 2 and ratio > 1.02:
        add("hard", "context_leak", f"candidate has unexplained sentence inflation {src_sentences}→{dst_sentences}")

    character_sources: set[str] = set()
    if memory is not None:
        for source_name, desc in memory.characters.items():
            source_name = str(source_name or "").strip()
            if not source_name or not _source_term_present(source_name, original):
                continue
            ru = re.search(r"\bru=([^;]+)", str(desc), flags=re.I)
            if not ru:
                continue
            character_sources.add(source_name.casefold())
            preferred = ru.group(1).strip()
            if preferred and not _target_term_present(preferred, translated):
                add("hard", "character_name", f"canonical character rendering missing: {source_name} → {preferred}")

        # Analyzer-derived named glossary entries form a dynamic entity ledger.
        # Treat ordinary lexical glossary misses as advisory (base QA), but proper
        # names/titles must remain stable across a book.
        for source_term, preferred in memory.glossary.items():
            source_term = str(source_term or "").strip()
            preferred = str(preferred or "").strip()
            if (
                not source_term
                or not preferred
                or source_term.casefold() in character_sources
                or not _named_source_term(source_term)
                or not _source_term_present(source_term, original)
            ):
                continue
            if not _target_term_present(preferred, translated):
                add("hard", "entity_consistency", f"canonical named term missing: {source_term} → {preferred}")

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
) -> list[QualityIssueV3]:
    issues: list[QualityIssueV3] = []
    for segment in segments:
        candidate = translations.get(segment.id)
        if candidate is None:
            issues.append(QualityIssueV3(segment.id, "hard", "missing_id", "model omitted required segment id"))
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
