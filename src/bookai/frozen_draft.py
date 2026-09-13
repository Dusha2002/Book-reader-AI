from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from .models import BookMemory, Segment


T = TypeVar("T")


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text("utf-8")) if path.exists() else {}
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(value, dict):
        return {"version": 1, "entries": {}}
    entries = value.get("entries")
    if not isinstance(entries, dict):
        value["entries"] = {}
    return value


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def make_frozen_draft_backend(base_cls: type[T], *, path: str | Path, mode: str) -> type[T]:
    """Wrap a bulk translation backend with deterministic draft record/replay.

    record: call the real backend and persist every successful segment translation.
    replay: never call the upstream translation API; return the recorded translation
    only when both segment id and exact source text match. This makes post-processing
    A/B comparisons use byte-identical primary drafts.
    """
    draft_path = Path(path)
    normalized_mode = str(mode or "").strip().casefold()
    if normalized_mode not in {"record", "replay"}:
        raise ValueError(f"unsupported frozen draft mode: {mode!r}")

    class FrozenDraftBackend(base_cls):  # type: ignore[misc, valid-type]
        name = f"{getattr(base_cls, 'name', base_cls.__name__)}-frozen-{normalized_mode}"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._frozen_data = _load(draft_path)
            self._frozen_data.setdefault("version", 1)
            self._frozen_data.setdefault("entries", {})
            self._frozen_data["base_backend"] = getattr(base_cls, "name", base_cls.__name__)
            self._frozen_data["mode_last_used"] = normalized_mode

        def available(self) -> bool:
            if normalized_mode == "replay":
                return draft_path.exists() and bool(self._frozen_data.get("entries"))
            return bool(super().available())

        def translate_many(
            self,
            segments: list[Segment],
            memory: BookMemory,
            *,
            source_segments: list[Segment] | None = None,
        ) -> tuple[dict[str, str], dict[str, str]]:
            if normalized_mode == "record":
                rows, errors = super().translate_many(
                    segments,
                    memory,
                    source_segments=source_segments,
                )
                entries = self._frozen_data.setdefault("entries", {})
                for segment in segments:
                    value = rows.get(segment.id)
                    if not value:
                        continue
                    entries[str(segment.id)] = {
                        "source": str(segment.text or ""),
                        "translation": str(value),
                    }
                self._frozen_data["entry_count"] = len(entries)
                _save(draft_path, self._frozen_data)
                print(
                    f"[frozen-draft] mode=record call_segments={len(segments)} "
                    f"saved={len(rows)} total={len(entries)} path={draft_path}",
                    flush=True,
                )
                return rows, errors

            entries = dict(self._frozen_data.get("entries") or {})
            rows: dict[str, str] = {}
            errors: dict[str, str] = {}
            for segment in segments:
                sid = str(segment.id)
                entry = entries.get(sid)
                if not isinstance(entry, dict):
                    errors[sid] = "missing from frozen primary draft"
                    continue
                recorded_source = str(entry.get("source") or "")
                if recorded_source != str(segment.text or ""):
                    errors[sid] = "frozen draft source mismatch"
                    continue
                value = str(entry.get("translation") or "").strip()
                if not value:
                    errors[sid] = "empty frozen draft translation"
                    continue
                rows[sid] = value
            print(
                f"[frozen-draft] mode=replay requested={len(segments)} "
                f"replayed={len(rows)} errors={len(errors)} path={draft_path}",
                flush=True,
            )
            return rows, errors

    FrozenDraftBackend.__name__ = f"Frozen{base_cls.__name__}{normalized_mode.title()}"
    FrozenDraftBackend.__qualname__ = FrozenDraftBackend.__name__
    return FrozenDraftBackend
