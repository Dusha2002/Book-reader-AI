from __future__ import annotations

import re
from typing import Any

from .models import Segment


_NAME = r"[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2}"
_SPEECH_VERB = r"(?:said|asked|replied|answered|added|continued|murmured|whispered|shouted|called|cut\s+him\s+off|cut\s+her\s+off)"
_EXPLICIT_AFTER = re.compile(rf"['\"”’][,!?….]?\s*({_NAME})\s+{_SPEECH_VERB}\b", re.I)
_EXPLICIT_BEFORE = re.compile(rf"\b({_NAME})\s+{_SPEECH_VERB}\b", re.I)
_QUOTE_ONLY = re.compile(r"^\s*['\"“‘].*['\"”’]?\s*$", re.S)

_MALE_EVIDENCE = re.compile(
    r"\b(?:сказал|спросил|ответил|произнес|добавил|перебил|уверен|должен|готов|согласен)\b",
    re.I,
)
_FEMALE_EVIDENCE = re.compile(
    r"\b(?:сказала|спросила|ответила|произнесла|добавила|перебила|уверена|должна|готова|согласна)\b",
    re.I,
)

_MALE_TO_FEMALE = {
    "сказал": "сказала",
    "ответил": "ответила",
    "спросил": "спросила",
    "решил": "решила",
    "подумал": "подумала",
    "согласился": "согласилась",
    "уверен": "уверена",
    "готов": "готова",
    "должен": "должна",
}
_FEMALE_TO_MALE = {v: k for k, v in _MALE_TO_FEMALE.items()}


def _explicit_speaker(source: str) -> str:
    text = str(source or "")
    # Attribution before or after a quote is much stronger than a nearby capitalized
    # token. We intentionally do not guess speakers from narration alone.
    for pattern in (_EXPLICIT_AFTER, _EXPLICIT_BEFORE):
        match = pattern.search(text)
        if match:
            name = str(match.group(1) or "").strip()
            if name:
                return name
    return ""


def _is_short_quote_only(source: str) -> bool:
    text = str(source or "").strip()
    if not text or len(text) > 180:
        return False
    # No narrative attribution in the segment itself.
    if _explicit_speaker(text):
        return False
    return bool(_QUOTE_ONLY.match(text))


def _gender_evidence(target: str) -> str:
    text = str(target or "")
    male = bool(_MALE_EVIDENCE.search(text))
    female = bool(_FEMALE_EVIDENCE.search(text))
    if male and not female:
        return "male"
    if female and not male:
        return "female"
    return ""


def _replace_first_person_gender(text: str, gender: str) -> str:
    value = str(text or "")
    pairs = _FEMALE_TO_MALE if gender == "male" else _MALE_TO_FEMALE
    # Only touch a gendered form inside an explicit first-person phrase. This avoids
    # changing references to third parties elsewhere in the reply.
    for old, new in pairs.items():
        pattern = re.compile(rf"(\b[Яя](?:\s+[А-Яа-яЁё]+){{0,3}}\s+)\b{re.escape(old)}\b", re.I)
        match = pattern.search(value)
        if match:
            start, end = match.span()
            prefix = match.group(1)
            replacement = prefix + new
            value = value[:start] + replacement + value[end:]
    return value


class DialogueSpeakerContinuityGuard:
    """Conservative two-speaker continuity for short unattributed replies.

    v9d's structural insight is extended one step: when a dialogue has already
    established exactly two named speakers, and an unattributed short quote is
    sandwiched between turns by the same speaker, infer that the middle reply is
    the other participant. The layer never invents a name in the translation; it
    only uses that inference to fix demonstrably contradictory first-person gender.
    """

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {
            "explicit_speakers": 0,
            "inferred_speakers": 0,
            "gender_fixes": 0,
            "inferred_ids": [],
            "changed_ids": [],
        }

    def apply(self, segments: list[Segment], translated: dict[str, str]) -> list[str]:
        explicit = [_explicit_speaker(s.text) for s in segments]
        assigned = list(explicit)
        self.stats["explicit_speakers"] += sum(bool(x) for x in explicit)

        # Learn gender only from Russian turns already attributable to a named speaker.
        gender_by_speaker: dict[str, str] = {}
        for idx, speaker in enumerate(assigned):
            if not speaker:
                continue
            evidence = _gender_evidence(translated.get(segments[idx].id, ""))
            if evidence:
                gender_by_speaker[speaker] = evidence

        # Process sequentially so a confidently inferred alternating turn can help
        # establish the local two-speaker pair for the next turn.
        recent: list[str] = []
        for i, segment in enumerate(segments):
            speaker = assigned[i]
            if speaker:
                if speaker in recent:
                    recent.remove(speaker)
                recent.append(speaker)
                recent = recent[-2:]
                continue
            if not _is_short_quote_only(segment.text) or len(recent) != 2:
                continue

            previous = assigned[i - 1] if i > 0 else ""
            if not previous:
                continue
            # Strong look-ahead confirmation: the next explicit attribution repeats
            # the previous speaker, so this middle unattributed reply is the other.
            next_explicit = ""
            for j in range(i + 1, min(len(segments), i + 3)):
                if explicit[j]:
                    next_explicit = explicit[j]
                    break
            if not next_explicit or next_explicit.casefold() != previous.casefold():
                continue

            other = recent[0] if recent[-1].casefold() == previous.casefold() else recent[-1]
            if other.casefold() == previous.casefold():
                continue
            assigned[i] = other
            self.stats["inferred_speakers"] += 1
            self.stats["inferred_ids"].append(segment.id)

            target = str(translated.get(segment.id) or "")
            gender = gender_by_speaker.get(other, "")
            if gender:
                fixed = _replace_first_person_gender(target, gender)
                if fixed != target:
                    translated[segment.id] = fixed
                    self.stats["gender_fixes"] += 1
                    self.stats["changed_ids"].append(segment.id)
                    target = fixed
            evidence = _gender_evidence(target)
            if evidence:
                gender_by_speaker[other] = evidence
            if other in recent:
                recent.remove(other)
            recent.append(other)
            recent = recent[-2:]

        return list(self.stats["changed_ids"])
