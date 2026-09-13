from __future__ import annotations

import re
from typing import Any

from .models import Segment


_DIALOGUE_IN_SOURCE = re.compile(r"(^|[.!?…]\s+)[\"'“‘]", re.M)


def source_has_dialogue(text: str) -> bool:
    return bool(_DIALOGUE_IN_SOURCE.search(str(text or "")))


class DialogueDiscourseGuard:
    """Source-aware, lexically inert Russian dialogue/quotation normalizer.

    Ported as a clean v10 principle from v9d: direct speech and ordinary quoted
    narration are different structural modes. This layer never asks an LLM and
    never changes lexical content; it only normalizes punctuation when the source
    structure supports it.
    """

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {
            "scanned": 0,
            "source_dialogue_segments": 0,
            "changed": 0,
            "changed_ids": [],
        }

    @staticmethod
    def _normalize(segment: Segment, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r",\s*,+", ",", value)

        if source_has_dialogue(segment.text):
            if re.match(r"^\s*[\"'«]", value):
                value = re.sub(r"^\s*[\"'«]\s*", "— ", value, count=1)
            value = re.sub(r"—\s*[\"'«]\s*(?=[А-ЯЁ])", "— ", value)
            value = re.sub(r"(?<=[.!?…])\s*[\"'«]\s*(?=[А-ЯЁ])", " — ", value)
            # Russian model output often has «реплика», — сказал. Once the opening
            # quote became a dialogue dash, its paired closing quote must disappear.
            value = re.sub(r"(?<=[А-Яа-яЁё0-9])['\"»](?=[,!?….])", "", value)
            value = re.sub(r"([,!?….])\s*[\"'»](?=\s*(?:—|-|$))", r"\1", value)
            value = re.sub(r"»\s*(?=,\s*—)", "", value)
            value = re.sub(r"[\"'»]\s*$", "", value)
        else:
            value = re.sub(r"['\"]([^'\"\n]{1,240})['\"]", r"«\1»", value)

        value = re.sub(r"\s+([,.!?…])", r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        value = re.sub(r",\s*,+", ",", value)
        return value

    def apply(self, segments: list[Segment], translated: dict[str, str]) -> list[str]:
        changed: list[str] = []
        for segment in segments:
            if segment.id not in translated:
                continue
            self.stats["scanned"] += 1
            if source_has_dialogue(segment.text):
                self.stats["source_dialogue_segments"] += 1
            before = str(translated.get(segment.id) or "")
            after = self._normalize(segment, before)
            if after and after != before:
                translated[segment.id] = after
                changed.append(segment.id)
        self.stats["changed"] += len(changed)
        seen = list(self.stats.get("changed_ids") or [])
        for sid in changed:
            if sid not in seen:
                seen.append(sid)
        self.stats["changed_ids"] = seen
        return changed
