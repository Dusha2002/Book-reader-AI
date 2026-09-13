from __future__ import annotations

from typing import Any

from .v10_source_bible import SourceOnlyBookBibleBuilder, _has_clean_russian, _norm


_EN_VOWELS = set("aeiouy")
_RU_VOWELS = set("аеёиоуыэюя")


def spelling_preserving_v9ad(source: str, ru: str) -> bool:
    """Conservative source-spelling guard for invented literary proper names."""
    src = [c.casefold() for c in str(source or "") if c.isascii() and c.isalpha()]
    rus = [c.casefold() for c in str(ru or "") if ("а" <= c.casefold() <= "я") or c.casefold() == "ё"]
    if not src or not rus:
        return False
    src_vowels = sum(c in _EN_VOWELS for c in src)
    ru_vowels = sum(c in _RU_VOWELS for c in rus)
    # v9ad deliberately preferred spelling preservation over guessed
    # pronunciation for fictional names. Do not collapse visible vowels.
    if len(src) >= 4 and src_vowels >= 2 and ru_vowels < src_vowels:
        return False
    # Nor may a substantial visible source cluster simply disappear.
    if len(src) >= 6 and len(rus) < len(src) - 1:
        return False
    return True


class V9ADSourceOnlyBookBibleBuilder(SourceOnlyBookBibleBuilder):
    """Source-only Bible with v9ad's high-coverage spelling-preserving canon."""

    @staticmethod
    def _accept_name_item(item: dict[str, Any], allowed: dict[str, dict[str, Any]], *, threshold: float):
        source = str(item.get("source") or "").strip()
        ru = _norm(item.get("ru") or "")
        try:
            confidence = float(item.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        if source not in allowed or confidence < threshold or not _has_clean_russian(ru) or not spelling_preserving_v9ad(source, ru):
            return None
        kind = str(item.get("kind") or "other").casefold()
        if kind not in {"person", "place", "institution", "other"}:
            kind = "other"
        gender = str(item.get("gender") or "unknown").casefold()
        if gender not in {"male", "female", "unknown"}:
            gender = "unknown"
        role = _norm(item.get("role") or "")[:220]
        voice = _norm(item.get("voice") or "")[:220]
        return source, ru, kind, gender, f"role={role};voice={voice}"
