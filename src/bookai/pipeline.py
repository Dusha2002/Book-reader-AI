from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .llm import analyze_memory, edit_batch, qa_batch, translate_batch, update_memory
from .models import BookMemory, LLMProvider, Segment, StyleGuide
from .parsers.base import load_book, save_book

ProgressCallback = Callable[[dict], None]


def _batches(segments: list[Segment], char_limit: int = 12000):
    batch: list[Segment] = []
    size = 0
    chapter = ""
    for segment in segments:
        chapter_changed = bool(batch and segment.chapter and chapter and segment.chapter != chapter)
        would_overflow = bool(batch and size + len(segment.text) > char_limit)
        if chapter_changed or would_overflow:
            yield batch
            batch, size = [], 0
        batch.append(segment)
        size += len(segment.text)
        chapter = segment.chapter or chapter
    if batch:
        yield batch


def _cache_path(source: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    return cache_dir / f"{digest}.json"


def _memory_from_dict(data: dict) -> BookMemory:
    style_data = data.get("style") or {}
    return BookMemory(
        title=str(data.get("title") or ""),
        author=str(data.get("author") or ""),
        style=StyleGuide(**{k: v for k, v in style_data.items() if k in StyleGuide.__dataclass_fields__}),
        glossary=dict(data.get("glossary") or {}),
        characters=dict(data.get("characters") or {}),
        rolling_summary=str(data.get("rolling_summary") or ""),
        last_chapter=str(data.get("last_chapter") or ""),
    )


def _notify(callback: ProgressCallback | None, **payload) -> None:
    if callback:
        callback(payload)


def translate_book(
    source: Path,
    output: Path,
    provider: LLMProvider,
    *,
    mode: str = "high",
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    memory_updates: bool = True,
) -> Path:
    if mode not in {"fast", "standard", "high"}:
        raise ValueError("mode must be fast, standard, or high")

    _notify(progress, phase="parsing", progress=1)
    document = load_book(source)
    if not document.segments:
        raise ValueError("No translatable text found in book")

    cache_dir = cache_dir or source.parent / ".bookai-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = _cache_path(source, cache_dir)
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

    memory_data = state.get("memory")
    if memory_data:
        memory = _memory_from_dict(memory_data)
    else:
        _notify(progress, phase="analyzing_style", progress=3)
        sample = "\n\n".join(s.text for s in document.segments[:150])[:50000]
        memory = analyze_memory(provider, sample)
        state["memory"] = asdict(memory)
        state["translations"] = {}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    translated: dict[str, str] = dict(state.get("translations") or {})
    batches = list(_batches(document.segments))
    total = len(document.segments)
    completed = sum(1 for s in document.segments if s.id in translated)
    _notify(progress, phase="translating", progress=max(5, int(completed / total * 90)), completed=completed, total=total)

    for batch_index, batch in enumerate(batches):
        pending = [s for s in batch if s.id not in translated]
        if not pending:
            continue
        final = translate_batch(provider, pending, memory)
        if mode in {"standard", "high"}:
            _notify(progress, phase="literary_edit", progress=max(5, int(completed / total * 90)), completed=completed, total=total)
            final = edit_batch(provider, pending, final, memory)
        if mode == "high":
            _notify(progress, phase="quality_check", progress=max(5, int(completed / total * 90)), completed=completed, total=total)
            final = qa_batch(provider, pending, final, memory)

        translated.update(final)
        completed += len(pending)

        if memory_updates and mode != "fast":
            _notify(progress, phase="updating_memory", progress=max(5, int(completed / total * 90)), completed=completed, total=total)
            memory = update_memory(provider, pending, final, memory)

        state["translations"] = translated
        state["memory"] = asdict(memory)
        state["last_batch"] = batch_index
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
        _notify(
            progress,
            phase="translating",
            progress=min(95, 5 + int(completed / total * 90)),
            completed=completed,
            total=total,
            chapter=pending[-1].chapter if pending else "",
        )

    _notify(progress, phase="building_book", progress=97, completed=completed, total=total)
    save_book(document, translated, output)
    _notify(progress, phase="done", progress=100, completed=total, total=total)
    return output
