from __future__ import annotations

import json


def issue_rows(obj: object, role: str) -> list[dict]:
    """Normalize only unambiguous variants of the critic `issues` contract.

    Small models often encode an empty list as 0/false/null or emit one issue as
    an object instead of a one-element array. Those representations are
    semantically unambiguous and safe to normalize. Anything ambiguous is a
    ValueError so the caller can retry/split instead of silently skipping QA.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"{role} returned non-object JSON")

    raw = obj.get("issues", [])
    if raw is None or raw is False or raw == 0 or raw == "":
        return []

    if isinstance(raw, str):
        low = raw.strip().casefold()
        if low in {"none", "no issues", "no_issue", "no_issues", "ok", "[]"}:
            return []
        if raw.lstrip().startswith(("[", "{")):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{role} returned malformed stringified issues") from exc

    if isinstance(raw, dict):
        if not raw:
            return []
        return [raw]

    if not isinstance(raw, list):
        raise ValueError(f"{role} issues must be an array, got {type(raw).__name__}")

    rows: list[dict] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{role} issue #{index} is not an object")
        rows.append(item)
    return rows
