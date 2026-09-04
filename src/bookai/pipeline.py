from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .harness import TranslationHarness
from .models import BookMemory, LLMProvider, Segment, StyleGuide
from .parsers.base import load_book, save_book

ProgressCallback = Callable[[dict], None]
MODE_ALIASES = {"standard": "optimal", "high": "literary"}
VALID_MODES = {"fast", "optimal", "literary", *MODE_ALIASES}


def _batches(segments: list[Segment], char_limit: int = 100000):
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


def _chapter_groups(segments: list[Segment]) -> list[tuple[str, list[Segment]]]:
    groups: list[tuple[str, list[Segment]]] = []
    current_name = ""
    current: list[Segment] = []
    for segment in segments:
        name = segment.chapter or current_name or "Book"
        if current and name != current_name:
            groups.append((current_name, current))
            current = []
        current_name = name
        current.append(segment)
    if current:
        groups.append((current_name, current))
    return groups


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


def _analysis_sample(chapters: list[tuple[str, list[Segment]]], char_limit: int) -> str:
    """Prefer whole-book analysis; for very large books, sample evenly across chapters."""
    full = "\n\n".join(s.text for _, chapter in chapters for s in chapter)
    if len(full) <= char_limit:
        return full
    per_chapter = max(4000, char_limit // max(1, len(chapters)))
    parts: list[str] = []
    for _name, chapter in chapters:
        text = "\n\n".join(s.text for s in chapter)
        if len(text) <= per_chapter:
            parts.append(text)
        else:
            half = per_chapter // 2
            parts.append(text[:half] + "\n...[middle omitted for analysis budget]...\n" + text[-half:])
    return "\n\n".join(parts)[:char_limit]


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
    source_segments = [s for s in document.segments if _should_translate(s.text)]
    if not source_segments:
        raise ValueError("No translatable text found in book")
    chapters = _chapter_groups(source_segments)

    cache_dir = cache_dir or source.parent / ".bookai-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = _cache_path(source, cache_dir, mode)
    try:
        state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        state = {}

    memory_data = state.get("memory")
    if memory_data:
        memory = _memory_from_dict(memory_data)
    else:
        _notify(progress, phase="analyzing_book", progress=3, chapters=len(chapters))
        analysis_chars = max(50000, int(os.getenv("BOOKAI_ANALYSIS_CHARS") or "1500000"))
        memory = harness.analyze(_analysis_sample(chapters, analysis_chars))
        state["memory"] = asdict(memory)
        state.setdefault("translations", {})
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    translated: dict[str, str] = dict(state.get("translations") or {})
    total = len(source_segments)
    completed = sum(s.id in translated for s in source_segments)
    batch_chars = max(4000, int(os.getenv("BOOKAI_BATCH_CHARS") or "100000"))
    concurrency = max(1, min(16, int(os.getenv("BOOKAI_CONCURRENCY") or "4")))
    _notify(progress, phase="translating", progress=max(5, int(completed / total * 90)), completed=completed, total=total, concurrency=concurrency)

    # The global book bible makes chapters independent enough to translate concurrently.
    # A chapter remains intact whenever it fits the model context.
    def process_chapter(name: str, chapter: list[Segment]):
        existing = {s.id: translated[s.id] for s in chapter if s.id in translated}
        new: dict[str, str] = {}
        target_for_gate: list[Segment] = []
        for batch in _batches(chapter, batch_chars):
            pending = [s for s in batch if s.id not in existing]
            if not pending:
                continue
            before, after = _context_for(source_segments, pending)
            draft = harness.translate(pending, memory, context_before=before, context_after=after)
            new.update(draft)
            target_for_gate.extend(pending)

        findings = []
        if mode in {"optimal", "literary"} and target_for_gate:
            chapter_draft = {**existing, **new}
            findings = harness.gate_findings(target_for_gate, chapter_draft, memory)
            finding_by_id = {f.id: f for f in findings}
            medium = [s for s in target_for_gate if s.id in finding_by_id and finding_by_id[s.id].severity == "medium"]
            hard = [s for s in target_for_gate if s.id in finding_by_id and finding_by_id[s.id].severity == "hard"]
            to_edit = medium + hard
            if to_edit:
                new.update(harness.edit(to_edit, {**existing, **new}, memory))
            if mode == "literary" and hard:
                new.update(harness.hard_edit(hard, {**existing, **new}, memory))
        return name, new, findings

    pending_chapters = [(name, chapter) for name, chapter in chapters if any(s.id not in translated for s in chapter)]
    if concurrency == 1 or len(pending_chapters) <= 1:
        results = [process_chapter(name, chapter) for name, chapter in pending_chapters]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="bookai") as pool:
            futures = {pool.submit(process_chapter, name, chapter): name for name, chapter in pending_chapters}
            for future in as_completed(futures):
                results.append(future.result())

    completed_chapters = set(state.get("completed_chapters") or [])
    for name, new, findings in results:
        translated.update(new)
        completed_chapters.add(name)
        completed = sum(s.id in translated for s in source_segments)
        state["translations"] = translated
        state["memory"] = asdict(memory)
        state["completed_chapters"] = sorted(completed_chapters)
        state["last_findings"] = [asdict(f) for f in findings]
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
        _notify(
            progress,
            phase="translating",
            progress=min(95, 5 + int(completed / total * 90)),
            completed=completed,
            total=total,
            chapter=name,
            flagged=len(findings),
            concurrency=concurrency,
        )

    # One final continuity update keeps useful cache metadata without serializing every chapter.
    if memory_updates and mode != "fast" and chapters:
        _, last_chapter_segments = chapters[-1]
        last_translations = {s.id: translated[s.id] for s in last_chapter_segments if s.id in translated}
        if len(last_translations) == len(last_chapter_segments):
            _notify(progress, phase="finalizing_memory", progress=96, completed=completed, total=total)
            memory = harness.update_memory(last_chapter_segments, last_translations, memory)
            state["memory"] = asdict(memory)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")

    _notify(progress, phase="building_book", progress=97, completed=completed, total=total)
    save_book(document, translated, output)
    _notify(progress, phase="done", progress=100, completed=total, total=total)
    return output
