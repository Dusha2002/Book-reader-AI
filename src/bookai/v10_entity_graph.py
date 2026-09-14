from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import BookMemory
from .v10 import _norm


_RU_RE = re.compile(r"(?:^|;)ru=([^;]+)", re.I)
_KIND_RE = re.compile(r"(?:^|;)kind=(person|place|institution|other)(?:;|$)", re.I)
_RU_TOKEN_RE = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё'’.-]*")


def _canonical(memory: BookMemory, source: str) -> str:
    desc = str(memory.characters.get(source) or "")
    match = _RU_RE.search(desc)
    if match:
        return _norm(match.group(1))
    return _norm(memory.glossary.get(source) or "")


def _kind(memory: BookMemory, source: str) -> str:
    desc = str(memory.characters.get(source) or "")
    match = _KIND_RE.search(desc)
    if match:
        return match.group(1).casefold()
    if re.search(r"(?:^|;)gender=(?:male|female)(?:;|$)", desc, re.I):
        return "person"
    return "other"


def _replace_ru(desc: str, ru: str) -> str:
    value = str(desc or "")
    if _RU_RE.search(value):
        return _RU_RE.sub(lambda m: (";" if m.group(0).startswith(";") else "") + f"ru={ru}", value, count=1)
    return f"ru={ru};" + value.lstrip(";")


def _ru_tokens(value: str) -> list[str]:
    return _RU_TOKEN_RE.findall(_norm(value))


def harmonize_composite_entities(memory: BookMemory, cache_path: Path | None = None) -> dict[str, Any]:
    """Compose multi-token entity canonicals from independently accepted components.

    The graph never invents a spelling. Fully confirmed composites are assembled from
    all independently accepted components. When only some components are independently
    confirmed, a conservative partial harmonization is allowed *only* when the current
    composite canon has the same token count as the source: confirmed positions are
    replaced, while unconfirmed positions keep their already accepted composite form.
    This lets a recurring full name and a separately observed first/surname converge
    without requiring every component to occur alone in the source.
    """
    changed: dict[str, str] = {}
    skipped_partial: list[str] = []
    partially_harmonized: list[str] = []

    for source in sorted(memory.characters, key=lambda value: (value.count(" "), len(value))):
        parts = [part for part in str(source).split() if part]
        if not 2 <= len(parts) <= 4:
            continue
        if _kind(memory, source) not in {"person", "place", "institution"}:
            continue

        current = _canonical(memory, source)
        current_tokens = _ru_tokens(current)
        independent: dict[int, str] = {}
        for index, part in enumerate(parts):
            if part not in memory.characters:
                continue
            if _kind(memory, part) not in {"person", "place", "institution"}:
                continue
            canon = _canonical(memory, part)
            if canon and len(_ru_tokens(canon)) == 1:
                independent[index] = canon

        if not independent:
            skipped_partial.append(source)
            continue

        if len(independent) == len(parts):
            composed_tokens = [independent[index] for index in range(len(parts))]
        elif len(current_tokens) == len(parts):
            composed_tokens = list(current_tokens)
            for index, canon in independent.items():
                composed_tokens[index] = canon
            partially_harmonized.append(source)
        else:
            skipped_partial.append(source)
            continue

        composed = " ".join(composed_tokens)
        if not composed or current.casefold().replace("ё", "е") == composed.casefold().replace("ё", "е"):
            continue

        memory.glossary[source] = composed
        memory.characters[source] = _replace_ru(memory.characters.get(source, ""), composed)
        changed[source] = composed

    if changed and cache_path is not None:
        path = Path(cache_path)
        try:
            data = json.loads(path.read_text("utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            canonicals = dict(data.get("canonicals") or {})
            canonicals.update(changed)
            data["canonicals"] = canonicals
            characters = dict(data.get("characters") or {})
            for source, ru in changed.items():
                if source in characters:
                    characters[source] = _replace_ru(str(characters[source]), ru)
            data["characters"] = characters
            try:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            except OSError:
                pass

    stats = {
        "composite_candidates": sum(1 for source in memory.characters if 2 <= len(str(source).split()) <= 4),
        "changed": len(changed),
        "changed_entities": changed,
        "partial_harmonized": len(partially_harmonized),
        "partial_entities": partially_harmonized,
        "skipped_partial": len(skipped_partial),
    }
    if changed:
        print("[v10-entity-graph] " + json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


__all__ = ["harmonize_composite_entities"]
