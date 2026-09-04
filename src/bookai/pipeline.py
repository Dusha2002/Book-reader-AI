from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .llm import analyze_memory, edit_batch, qa_batch, translate_batch
from .models import BookMemory, LLMProvider, Segment, StyleGuide
from .parsers.base import load_book, save_book


def _batches(segments: list[Segment], char_limit: int = 12000):
    batch: list[Segment] = []
    size = 0
    for segment in segments:
        if batch and size + len(segment.text) > char_limit:
            yield batch
            batch, size = [], 0
        batch.append(segment)
        size += len(segment.text)
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
    )


def translate_book(source: Path, output: Path, provider: LLMProvider, *, mode: str = "high", cache_dir: Path | None = None) -> Path:
    document = load_book(source)
    cache_dir = cache_dir or source.parent / ".bookai-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = _cache_path(source, cache_dir)
    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text("utf-8"))

    memory_data = state.get("memory")
    if memory_data:
        memory = _memory_from_dict(memory_data)
    else:
        sample = "\n\n".join(s.text for s in document.segments[:80])[:30000]
        memory = analyze_memory(provider, sample)
        state["memory"] = asdict(memory)
        state["translations"] = {}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    translated: dict[str, str] = dict(state.get("translations") or {})
    for batch in _batches(document.segments):
        pending = [s for s in batch if s.id not in translated]
        if not pending:
            continue
        final = translate_batch(provider, pending, memory)
        if mode in {"standard", "high"}:
            final = edit_batch(provider, pending, final, memory)
        if mode == "high":
            final = qa_batch(provider, pending, final, memory)
        translated.update(final)
        state["translations"] = translated
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    save_book(document, translated, output)
    return output
