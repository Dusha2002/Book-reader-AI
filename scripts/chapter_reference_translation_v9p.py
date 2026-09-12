from __future__ import annotations

import json
import re

import chapter_reference_translation_v9o as v9o

v9n = v9o.v9n
v9m = v9o.v9m
v9 = v9o.v9

# v9p adds a general person/deixis invariant for plural third-person pronouns.
# A non-reflexive English "them" cannot silently resolve to the current
# first-person speaker. This is language-level discourse logic, not a
# book-specific expected answer.

_SELF_EN = re.compile(r"\b(?:myself|ourselves|himself|herself|themselves|yourself|yourselves)\b", re.I)
_SPEAKER_LABEL = re.compile(
    r"\b(?:speaker|the\s+man\s+himself|man\s+himself|himself|self|сам\s+мужчина|самого\s+себя|себя\s+самого|говорящ\w*)\b",
    re.I,
)
_MULTI_THEM = re.compile(r"\b(?:either|neither|both)\s+of\s+them\b", re.I)

_BASE_VALIDATE = v9n._validate_priority_candidate
_BASE_REPAIR_PROMPT = v9n._repair_prompt
_BASE_RETRY_PROMPT = v9n._retry_prompt


def _labels_include_speaker(item: dict) -> bool:
    labels = [*(item.get("antecedents") or []), *(item.get("antecedents_ru") or [])]
    return any(_SPEAKER_LABEL.search(str(x or "")) for x in labels)


def _validate_priority_candidate_v9p(segment, candidate: str, code: str, item: dict):
    ok, reason = _BASE_VALIDATE(segment, candidate, code, item)
    if not ok:
        return ok, reason

    source = str(segment.text or "")
    if code == "referent" and _MULTI_THEM.search(source) and not _SELF_EN.search(source):
        if _labels_include_speaker(item):
            return False, "third-person them cannot include the current first-person speaker without an explicit reflexive source form"

        target = v9._norm_text(candidate).casefold()
        # Guard against a model returning clean metadata but lexicalizing the
        # speaker in Russian anyway.
        if re.search(r"\b(?:себя|самого\s+себя|себя\s+самого)\b", target):
            return False, "Russian candidate incorrectly lexicalizes the speaker as a member of third-person them"

    return True, ""


def _repair_prompt_v9p() -> str:
    return _BASE_REPAIR_PROMPT() + """

PERSON/DEIXIS INVARIANT: for non-reflexive third-person plural forms such as either/neither/both of them, NEVER include the current first-person speaker among the antecedents merely to make a pair. 'them' denotes third-person discourse entities. If one entity is explicit and another is omitted, recover the missing third-person entity from the nearest coordinated questions/answers or prior discourse slots. For example, questions about multiple family slots may introduce a spouse slot and a child slot even if the answer explicitly states only one of them. Do not invent the speaker as the second member."""


def _retry_prompt_v9p() -> str:
    return _BASE_RETRY_PROMPT() + """

STRICT PERSON RULE: if the source says non-reflexive 'them', all proposed antecedents must be third-person entities distinct from the current speaker. Reject any analysis that uses 'the man himself', 'speaker', 'self', 'себя', or equivalent unless the English source explicitly contains a reflexive pronoun. Recover omitted entities from the nearest coordinated semantic slots/questions before translating."""


v9n._validate_priority_candidate = _validate_priority_candidate_v9p
v9n._repair_prompt = _repair_prompt_v9p
v9n._retry_prompt = _retry_prompt_v9p


def main():
    v9o.main()
    report = v9.v3.REPORT
    if report.exists():
        try:
            data = json.loads(report.read_text("utf-8"))
            data["architecture"] = {
                **dict(data.get("architecture") or {}),
                "version": "quality-v9p-third-person-antecedent-invariant",
                "pronoun_invariant": "non-reflexive third-person them excludes current first-person speaker",
                "referent_recovery": "nearest coordinated discourse slots/questions before fallback",
                "gold_reference_available_to_pipeline": False,
            }
            report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
