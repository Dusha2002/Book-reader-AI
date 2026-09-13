from __future__ import annotations

import re
from collections import Counter

from .models import Segment
from .v10 import BookBibleBuilder, _norm


class AtomicBookBibleBuilder(BookBibleBuilder):
    """Whole-book candidate builder that preserves technical collocations atomically."""

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict]:
        base = BookBibleBuilder._candidate_records(segments)
        by_key = {str(row.get("candidate") or "").casefold(): row for row in base}
        counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}
        noun = r"(?:saw|file|plate|lock|locks|gear|gears|blade|blades|tool|tools|machine|screw|screws|spring|springs|stock|armou?r)"
        collocation_re = re.compile(rf"\b(?:the\s+|a\s+|an\s+)?([a-z][a-z-]+(?:\s+[a-z][a-z-]+)?)\s+({noun})\b", re.I)
        rare_re = re.compile(r"\b(?:cuisses?|gorget|brigandine|tinplate)\b", re.I)

        def add(value: str, segment: Segment) -> None:
            value = _norm(re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)).casefold()
            if not value:
                return
            counts[value] += 1
            rows = contexts.setdefault(value, [])
            if len(rows) < 2:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:430]})

        for segment in segments:
            text = str(segment.text or "")
            for match in collocation_re.finditer(text):
                add(f"{match.group(1)} {match.group(2)}", segment)
            for match in rare_re.finditer(text):
                add(match.group(0), segment)

        for value, count in counts.most_common(100):
            key = value.casefold()
            if key in by_key:
                # Enrich the existing record with better phrase-level contexts.
                old = by_key[key]
                old["frequency"] = max(int(old.get("frequency") or 0), count)
                old["contexts"] = contexts[value]
                old["kind_hint"] = "technical_phrase"
                continue
            row = {
                "candidate": value,
                "kind_hint": "technical_phrase",
                "frequency": count,
                "contexts": contexts[value],
            }
            base.append(row)
            by_key[key] = row
        return base
