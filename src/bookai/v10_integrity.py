from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import BookMemory, Segment
from .v10 import _usage
from .v10_transport import RobustTaggedPrimaryTransport, _looks_like_prompt_leak


_PROTOCOL_RE = re.compile(
    r"(?:\b(?:characters?|glossary|context_only|targets?|source|translation)\s*:|"
    r"\b(?:персонажи|глоссарий|контекст|источник|перевод)\s*:)",
    re.I,
)


@dataclass(frozen=True)
class IntegrityIssue:
    id: str
    code: str
    reason: str


class SegmentIntegrityGate:
    """Reject cross-segment/protocol corruption before semantic QA."""

    def __init__(self, backend: RobustTaggedPrimaryTransport) -> None:
        self.backend = backend
        self.stats: dict[str, Any] = {
            "detected": 0,
            "recovery_calls": 0,
            "repaired": 0,
            "rejected_recoveries": 0,
            "residual": 0,
            "detected_ids": [],
            "repaired_ids": [],
            "residual_ids": [],
        }

    @staticmethod
    def scan_segment(segment: Segment, target: str) -> list[IntegrityIssue]:
        source = str(segment.text or "").strip()
        ru = str(target or "").strip()
        out: list[IntegrityIssue] = []
        if not ru:
            return [IntegrityIssue(segment.id, "missing", "segment has no translation")]
        if _looks_like_prompt_leak(ru) or _PROTOCOL_RE.search(ru):
            out.append(IntegrityIssue(segment.id, "protocol_residue", "model prompt/protocol residue leaked into translation"))
        if len(source) >= 55:
            ratio = len(ru) / max(1, len(source))
            low_limit = 0.48 if len(source) < 260 else 0.42
            high_limit = 1.75 if len(source) < 260 else 1.65
            if ratio < low_limit:
                out.append(IntegrityIssue(segment.id, "implausible_compression", f"source/target char ratio={ratio:.2f}"))
            elif ratio > high_limit:
                out.append(IntegrityIssue(segment.id, "implausible_expansion", f"source/target char ratio={ratio:.2f}"))
        return out

    def scan(self, segments: list[Segment], translated: dict[str, str]) -> list[IntegrityIssue]:
        out: list[IntegrityIssue] = []
        for segment in segments:
            out.extend(self.scan_segment(segment, translated.get(segment.id, "")))
        return out

    def _source_only_recover(self, segment: Segment) -> str:
        """Retranslate one exact source segment with zero neighboring prose.

        The previous fallback reused CONTEXT_ONLY, which is precisely dangerous after
        detecting a boundary shift: the model can copy the neighboring paragraph again.
        Structural recovery therefore receives only the suspect source itself.
        """
        client = self.backend._ensure_client()
        request = {
            "model": self.backend.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Переведи РОВНО один английский фрагмент литературной прозы на русский. "
                        "SOURCE ниже — единственный текст, который разрешено переводить. "
                        "Не продолжай сцену, не добавляй соседний контекст, служебные подписи или комментарии. "
                        "Сохрани все предложения, реплики, числа, имена, причинность и порядок. "
                        "Верни только полный готовый русский перевод этого SOURCE."
                    ),
                },
                {"role": "user", "content": str(segment.text or "")},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": max(1200, min(6000, int(self.backend.max_tokens))),
        }
        try:
            response = client.chat(request)
            self.backend.usage.add(_usage(response), calls=1)
        except Exception as exc:
            print(f"[v10-integrity] id={segment.id} error={type(exc).__name__}", flush=True)
            return ""
        value = str(response.choices[0].message.content or "").strip()
        value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
        value = re.sub(r"^\s*(?:перевод|translation)\s*:\s*", "", value, flags=re.I)
        return value.strip()

    def repair(
        self,
        segments: list[Segment],
        translated: dict[str, str],
        memory: BookMemory,
        *,
        source_segments: list[Segment] | None = None,
    ) -> list[str]:
        del memory, source_segments  # integrity recovery intentionally ignores both
        issues = self.scan(segments, translated)
        suspect_ids = list(dict.fromkeys(issue.id for issue in issues))
        self.stats["detected"] = len(suspect_ids)
        self.stats["detected_ids"] = suspect_ids
        by_id = {s.id: s for s in segments}
        changed: list[str] = []
        for sid in suspect_ids:
            segment = by_id[sid]
            self.stats["recovery_calls"] += 1
            candidate = self._source_only_recover(segment)
            if not candidate:
                self.stats["rejected_recoveries"] += 1
                continue
            before = self.scan_segment(segment, translated.get(sid, ""))
            after = self.scan_segment(segment, candidate)
            if before and not after:
                translated[sid] = candidate
                changed.append(sid)
                self.stats["repaired"] += 1
                self.stats["repaired_ids"].append(sid)
            else:
                self.stats["rejected_recoveries"] += 1

        residual = self.scan(segments, translated)
        residual_ids = list(dict.fromkeys(issue.id for issue in residual))
        self.stats["residual"] = len(residual_ids)
        self.stats["residual_ids"] = residual_ids
        return changed
