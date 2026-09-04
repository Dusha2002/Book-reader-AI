from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .harness import TranslationHarness
from .models import BookMemory, LLMProvider, Segment, StyleGuide
from .parsers.base import load_book, save_book

ProgressCallback = Callable[[dict], None]
MODE_ALIASES = {"standard": "optimal", "high": "literary"}
VALID_MODES = {"fast", "optimal", "literary", *MODE_ALIASES}


def _batches(segments: list[Segment], char_limit: int = 30000):
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


def _should_translate(text: str) -> bool:
    """Skip already-Russian/service-only blocks when translating English books."""
    latin = len(re.findall(r"[A-Za-z]", text))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    if latin < 2:
        return False
    if cyrillic and cyrillic > max(3, latin // 3):
        return False
    return True


def _cache_path(source: Path, cache_dir: Path, mode: str) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    return cache_dir / f"{digest}.{mode}.json"


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


def _context_for(segments: list[Segment], batch: list[Segment], radius: int = 2) -> tuple[list[Segment], list[Segment]]:
    by_id = {s.id: i for i, s in enumerate(segments)}
    start = by_id[batch[0].id]
    end = by_id[batch[-1].id]
    return segments[max(0, start - radius) : start], segments[end + 1 : end + 1 + radius]


def translate_book(
    source: Path,
    output: Path,
    provider_or_harness: LLMProvider | TranslationHarness,
    *,
    mode: str = "optimal",
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    memory_updates: bool = True,
) -> Path:
    if mode not in VALID_MODES:
        raise ValueError("mode must be fast, optimal/literary (legacy aliases: standard/high)")
    mode = MODE_ALIASES.get(mode, mode)
    harness = provider_or_harness if isinstance(provider_or_harness, TranslationHarness) else TranslationHarness.single_provider(provider_or_harness)

    _notify(progress, phase="parsing", progress=1)
    document = load_book(source)
    if not document.segments:
        raise ValueError("No translatable text found in book")

    cache_dir = cache_dir or source.parent / ".bookai-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = _cache_path(source, cache_dir, mode)
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
        source_segments = [s for s in document.segments if _should_translate(s.text)]
        sample = "\n\n".join(s.text for s in source_segments[:150])[:50000]
        memory = harness.analyze(sample)
        state["memory"] = asdict(memory)
        state["translations"] = {}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    translated: dict[str, str] = dict(state.get("translations") or {})
    source_segments = [s for s in document.segments if _should_translate(s.text)]
    batch_chars = max(4000, int(os.getenv("BOOKAI_BATCH_CHARS") or "30000"))
    batches = list(_batches(source_segments, batch_chars))
    total = len(source_segments)
    completed = sum(1 for s in source_segments if s.id in translated)
    updated_chapters = set(state.get("memory_updated_chapters") or [])
    _notify(progress, phase="translating", progress=max(5, int(completed / total * 90)), completed=completed, total=total)

    for batch_index, batch in enumerate(batches):
        pending = [s for s in batch if s.id not in translated]
        if not pending:
            continue
        before, after = _context_for(source_segments, pending)
        final = harness.translate(pending, memory, context_before=before, context_after=after)

        findings = []
        if mode in {"optimal", "literary"}:
            _notify(progress, phase="quality_gate", progress=max(5, int(completed / total * 90)), completed=completed, total=total)
            findings = harness.gate_findings(pending, final, memory)
            finding_by_id = {f.id: f for f in findings}
            medium = [s for s in pending if s.id in finding_by_id and finding_by_id[s.id].severity == "medium"]
            hard = [s for s in pending if s.id in finding_by_id and finding_by_id[s.id].severity == "hard"]

            # Qwen3.8 Flash (default editor) sees only flagged passages, not the whole book.
            to_edit = medium + hard
            if to_edit:
                _notify(progress, phase="selective_edit", progress=max(5, int(completed / total * 90)), completed=completed, total=total, flagged=len(to_edit))
                final.update(harness.edit(to_edit, final, memory))

            # Expensive senior model is reserved for hard cases and only in literary mode.
            if mode == "literary" and hard:
                _notify(progress, phase="hard_cases", progress=max(5, int(completed / total * 90)), completed=completed, total=total, hard=len(hard))
                final.update(harness.hard_edit(hard, final, memory))

        translated.update(final)
        completed += len(pending)

        # Update continuity once per completed chapter instead of once per translation batch.
        # This cuts a large novel from O(batches) memory calls to roughly O(chapters).
        current_chapter = batch[-1].chapter if batch else ""
        next_chapter = batches[batch_index + 1][0].chapter if batch_index + 1 < len(batches) else None
        chapter_finished = next_chapter != current_chapter
        if memory_updates and mode != "fast" and chapter_finished and current_chapter not in updated_chapters:
            chapter_segments = [s for s in source_segments if s.chapter == current_chapter]
            chapter_translations = {s.id: translated[s.id] for s in chapter_segments if s.id in translated}
            if chapter_segments and len(chapter_translations) == len(chapter_segments):
                _notify(progress, phase="updating_memory", progress=max(5, int(completed / total * 90)), completed=completed, total=total)
                memory = harness.update_memory(chapter_segments, chapter_translations, memory)
                updated_chapters.add(current_chapter)

        state["translations"] = translated
        state["memory"] = asdict(memory)
        state["last_batch"] = batch_index
        state["memory_updated_chapters"] = sorted(updated_chapters)
        state["last_findings"] = [asdict(f) for f in findings]
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
        _notify(
            progress,
            phase="translating",
            progress=min(95, 5 + int(completed / total * 90)),
            completed=completed,
            total=total,
            chapter=pending[-1].chapter if pending else "",
            flagged=len(findings),
        )

    _notify(progress, phase="building_book", progress=97, completed=completed, total=total)
    save_book(document, translated, output)
    _notify(progress, phase="done", progress=100, completed=total, total=total)
    return output
