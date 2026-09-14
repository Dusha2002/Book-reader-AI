from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import BookMemory
from .v10 import _norm


_RU_RE = re.compile(r"(?:^|;)ru=([^;]+)", re.I)
_KIND_RE = re.compile(r"(?:^|;)kind=(person|place|institution|other)(?:;|$)", re.I)


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


def harmonize_composite_entities(memory: BookMemory, cache_path: Path | None = None) -> dict[str, Any]:
    """Compose multi-token entity canonicals from independently accepted components.

    The graph is deliberately conservative: it never invents a spelling. A composite
    entity is rewritten only when every whitespace-separated component is itself an
    accepted proper entity with a Russian canon. This prevents full names from drifting
    away from stable first/surname canonicals while leaving genuinely indivisible names
    untouched.
    """
    changed: dict[str, str] = {}
    skipped_partial: list[str] = []

    for source in sorted(memory.characters, key=lambda value: (value.count(" "), len(value))):
        parts = [part for part in str(source).split() if part]
        if not 2 <= len(parts) <= 4:
            continue
        if _kind(memory, source) not in {"person", "place", "institution"}:
            continue

        component_ru: list[str] = []
        complete = True
        for part in parts:
            if part not in memory.characters:
                complete = False
                break
            if _kind(memory, part) not in {"person", "place", "institution"}:
                complete = False
                break
            canon = _canonical(memory, part)
            if not canon:
                complete = False
                break
            component_ru.append(canon)
        if not complete:
            skipped_partial.append(source)
            continue

        composed = " ".join(component_ru)
        current = _canonical(memory, source)
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
        "skipped_partial": len(skipped_partial),
    }
    if changed:
        print("[v10-entity-graph] " + json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


__all__ = ["harmonize_composite_entities"]
