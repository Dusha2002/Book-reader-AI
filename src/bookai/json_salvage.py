from __future__ import annotations

import json
import re
from typing import Any


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.I)
_FENCE_CLOSE_RE = re.compile(r"\s*```$", re.I)


def _strip_wrapping(text: str) -> str:
    raw = str(text or "").strip().lstrip("\ufeff")
    raw = _FENCE_OPEN_RE.sub("", raw)
    raw = _FENCE_CLOSE_RE.sub("", raw)
    return raw.strip()


def _clean_jsonish(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _balanced_objects(text: str, start: int = 0) -> list[str]:
    """Extract complete JSON object substrings, ignoring braces inside strings."""
    out: list[str] = []
    depth = 0
    obj_start: int | None = None
    in_string = False
    escaped = False
    for i in range(max(0, start), len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and obj_start is not None:
                out.append(text[obj_start : i + 1])
                obj_start = None
    return out


def parse_json_with_item_salvage(text: str) -> dict[str, Any]:
    """Parse GigaChat JSON and salvage complete `items` rows from partial output.

    GigaChat occasionally wraps JSON in prose/fences, emits trailing commas, or
    truncates the final array/object after already producing several valid items.
    Repair callers can safely use those complete rows and retry only the residual.
    """
    raw = _strip_wrapping(text)
    attempts = [raw]

    if "{" in raw and "}" in raw:
        attempts.append(raw[raw.find("{") : raw.rfind("}") + 1])
    attempts.extend(_clean_jsonish(x) for x in list(attempts))

    seen: set[str] = set()
    for candidate in attempts:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"items": obj}

    # Salvage complete objects specifically from an `items` array, even if the
    # response was truncated before the closing ]/}.
    marker = re.search(r'"items"\s*:\s*\[', raw)
    search_from = marker.end() if marker else 0
    items: list[dict[str, Any]] = []
    for chunk in _balanced_objects(raw, search_from):
        try:
            value = json.loads(_clean_jsonish(chunk))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and str(value.get("id") or ""):
            items.append(value)
    if items:
        return {"items": items, "_salvaged_partial": True}

    raise json.JSONDecodeError("Unable to parse or salvage JSON object", raw, 0)
