from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from .harness import TranslationHarness
from .models import BookMemory, GateFinding, LLMProvider, Segment, StyleGuide
from .parsers.base import load_book, save_book
from .quality import batch_issues, candidate_issues, hard_ids

ProgressCallback = Callable[[dict], None]
MODE_ALIASES = {"standard": "optimal", "high": "literary"}
VALID_MODES = {"fast", "optimal", "literary", *MODE_ALIASES}
PIPELINE_VERSION = "literary-harness-v2"


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
    latin = len(re.findall(r"[A-Za-z]", text))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    if latin < 2:
        return False
    if cyrillic and cyrillic > max(3, latin // 3):
        return False
    return True


def _looks_untranslated(original: str, candidate: str) -> bool:
    """Backwards-compatible helper used by recovery scripts/tests."""
    dummy = Segment("compat", original, "/p")
    codes = {issue.code for issue in candidate_issues(dummy, candidate)}
    return bool(codes & {"empty", "unchanged", "english_leftover", "unexpected_script"})


def _cache_path(source: Path, cache_dir: Path, mode: str) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    return cache_dir / f"{digest}.{mode}.{PIPELINE_VERSION}.json"


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


def _context_for(segments: list[Segment], batch: list[Segment], radius: int = 3) -> tuple[list[Segment], list[Segment]]:
    by_id = {s.id: i for i, s in enumerate(segments)}
    start = by_id[batch[0].id]
    end = by_id[batch[-1].id]
    return segments[max(0, start - radius) : start], segments[end + 1 : end + 1 + radius]


def _translated_context(
    source_segments: list[Segment],
    target: list[Segment],
    translations: dict[str, str],
    radius: int = 3,
) -> list[dict]:
    before, after = _context_for(source_segments, target, radius=radius)
    return [
        {"id": s.id, "source": s.text, "translation": translations.get(s.id, "")}
        for s in before + after
    ]


def _analysis_sample(chapters: list[tuple[str, list[Segment]]], char_limit: int) -> str:
    """Even sample from beginning/middle/end of every chapter."""
    full = "\n\n".join(s.text for _, chapter in chapters for s in chapter)
    if len(full) <= char_limit:
        return full
    per_chapter = max(2500, char_limit // max(1, len(chapters)))
    parts: list[str] = []
    for name, chapter in chapters:
        text = "\n\n".join(s.text for s in chapter)
        if len(text) <= per_chapter:
            excerpt = text
        else:
            third = max(600, per_chapter // 3)
            middle = max(0, len(text) // 2 - third // 2)
            excerpt = (
                text[:third]
                + "\n...[omitted]...\n"
                + text[middle : middle + third]
                + "\n...[omitted]...\n"
                + text[-third:]
            )
        parts.append(f"\n### {name}\n{excerpt}")
    return "\n".join(parts)[:char_limit]


def _persist(path: Path, state: dict, translations: dict[str, str], memory: BookMemory) -> None:
    state["pipeline_version"] = PIPELINE_VERSION
    state["translations"] = translations
    state["memory"] = asdict(memory)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")


def _reasons(findings: list[GateFinding]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for finding in findings:
        out.setdefault(finding.id, []).append(finding.reason)
    return {sid: "; ".join(items) for sid, items in out.items()}


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
    harness = (
        provider_or_harness
        if isinstance(provider_or_harness, TranslationHarness)
        else TranslationHarness.single_provider(provider_or_harness)
    )

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
    if state and state.get("pipeline_version") != PIPELINE_VERSION:
        state = {}

    memory_data = state.get("memory")
    if memory_data:
        memory = _memory_from_dict(memory_data)
    else:
        _notify(progress, phase="analyzing_book", progress=2, chapters=len(chapters))
        analysis_chars = max(30000, int(os.getenv("BOOKAI_ANALYSIS_CHARS") or "90000"))
        memory = harness.analyze(_analysis_sample(chapters, analysis_chars))
        state = {
            "pipeline_version": PIPELINE_VERSION,
            "translations": {},
            "completed_chapters": [],
            "chapter_briefs": {},
            "qa_passed_chapters": [],
        }
        _persist(state_path, state, {}, memory)

    translated: dict[str, str] = dict(state.get("translations") or {})
    completed_chapters = set(state.get("completed_chapters") or [])
    qa_passed_chapters = set(state.get("qa_passed_chapters") or [])
    chapter_briefs: dict[str, str] = dict(state.get("chapter_briefs") or {})

    total = len(source_segments)
    completed = sum(s.id in translated for s in source_segments)
    batch_chars = max(3000, int(os.getenv("BOOKAI_BATCH_CHARS") or "12000"))
    gate_chars = max(3000, int(os.getenv("BOOKAI_GATE_BATCH_CHARS") or "10000"))
    edit_chars = max(2000, int(os.getenv("BOOKAI_EDIT_BATCH_CHARS") or "7000"))
    max_repair_rounds = max(1, min(4, int(os.getenv("BOOKAI_QUALITY_REPAIR_ROUNDS") or "2")))

    _notify(
        progress,
        phase="translating",
        progress=max(4, int(completed / total * 90)),
        completed=completed,
        total=total,
        restored=bool(completed),
        strategy="sequential-chapters-validator-first",
    )

    # Quality modes deliberately process chapters in book order. This lets the
    # glossary/character memory learn from chapter N before chapter N+1.
    for chapter_index, (name, chapter) in enumerate(chapters, 1):
        chapter_existing = {s.id: translated[s.id] for s in chapter if s.id in translated}

        if name in qa_passed_chapters and len(chapter_existing) == len(chapter):
            continue

        if mode != "fast":
            brief = chapter_briefs.get(name)
            if not brief:
                _notify(progress, phase="chapter_brief", chapter=name, chapter_index=chapter_index, chapters=len(chapters))
                brief = harness.chapter_brief(chapter, memory)
                chapter_briefs[name] = brief
                state["chapter_briefs"] = chapter_briefs
                _persist(state_path, state, translated, memory)
            local_memory = replace(
                memory,
                rolling_summary=(memory.rolling_summary + "\nCURRENT CHAPTER BRIEF: " + brief)[-12000:],
            )
        else:
            local_memory = memory

        # Pass 1: translate contiguous windows. Strict ID contract + deterministic
        # hard checks mean malformed output is never cached.
        for batch in _batches(chapter, batch_chars):
            pending = [s for s in batch if s.id not in translated]
            if not pending:
                continue
            before, after = _context_for(source_segments, pending)
            last_error: BaseException | None = None
            accepted: dict[str, str] | None = None
            for attempt in range(3):
                try:
                    candidate = harness.translate(
                        pending,
                        local_memory,
                        context_before=before,
                        context_after=after,
                    )
                    hard = hard_ids(batch_issues(pending, candidate, local_memory))
                    if hard:
                        raise ValueError("deterministic hard QA failed for: " + ", ".join(sorted(hard)))
                    accepted = candidate
                    break
                except BaseException as exc:
                    last_error = exc
                    _notify(
                        progress,
                        phase="batch_retry",
                        chapter=name,
                        attempt=attempt + 1,
                        ids=[s.id for s in pending],
                        error=type(exc).__name__,
                    )
            if accepted is None:
                raise RuntimeError(f"Translation batch failed strict acceptance in chapter {name}") from last_error
            translated.update(accepted)
            completed = sum(s.id in translated for s in source_segments)
            _persist(state_path, state, translated, memory)
            _notify(
                progress,
                phase="chapter_translate",
                chapter=name,
                chapter_index=chapter_index,
                chapters=len(chapters),
                completed=completed,
                total=total,
                progress=min(91, 4 + int(completed / total * 87)),
            )

        if any(s.id not in translated for s in chapter):
            raise RuntimeError(f"Chapter {name} has missing translated ids before QA")

        if mode != "fast":
            # Pass 2: bilingual QA over every segment. Deterministic checks are
            # merged with a Flash semantic/literary audit.
            findings: list[GateFinding] = []
            checked = 0
            for gate_batch in _batches(chapter, gate_chars):
                findings.extend(harness.gate_findings(gate_batch, translated, local_memory))
                checked += len(gate_batch)
                _notify(
                    progress,
                    phase="chapter_gate",
                    chapter=name,
                    checked=checked,
                    total=len(chapter),
                    flagged=len({f.id for f in findings}),
                )

            # Pass 3: targeted repair. Hard cases first get an independent second
            # Flash translation and blind Flash A/B choice. No expensive model.
            for repair_round in range(max_repair_rounds):
                if not findings:
                    break
                finding_map = {f.id: f for f in findings}
                hard_segments = [s for s in chapter if s.id in finding_map and finding_map[s.id].severity == "hard"]
                if hard_segments:
                    for hard_batch in _batches(hard_segments, edit_chars):
                        current = {s.id: translated[s.id] for s in hard_batch}
                        alternative = harness.alternative(hard_batch, local_memory)
                        alt_hard = hard_ids(batch_issues(hard_batch, alternative, local_memory))
                        if not alt_hard:
                            chosen = harness.choose(hard_batch, current, alternative, local_memory)
                            translated.update(chosen)

                targets = [s for s in chapter if s.id in finding_map]
                reasons = _reasons(findings)
                for target_batch in _batches(targets, edit_chars):
                    context = _translated_context(source_segments, target_batch, translated)
                    edited = harness.edit(
                        target_batch,
                        translated,
                        local_memory,
                        reasons=reasons,
                        context=context,
                    )
                    # Bad editor output is rejected; the known-good previous candidate remains.
                    if not hard_ids(batch_issues(target_batch, edited, local_memory)):
                        translated.update(edited)
                    else:
                        _notify(
                            progress,
                            phase="edit_rejected",
                            chapter=name,
                            ids=[s.id for s in target_batch],
                        )

                _persist(state_path, state, translated, memory)

                findings = []
                for gate_batch in _batches(targets, gate_chars):
                    findings.extend(harness.gate_findings(gate_batch, translated, local_memory))
                _notify(
                    progress,
                    phase="chapter_recheck",
                    chapter=name,
                    round=repair_round + 1,
                    remaining=len({f.id for f in findings}),
                )

            deterministic = batch_issues(chapter, translated, local_memory)
            det_hard = [i for i in deterministic if i.severity == "hard"]
            final_hard = [f for f in findings if f.severity == "hard"]
            if det_hard or final_hard:
                ids = sorted({x.id for x in det_hard} | {x.id for x in final_hard})
                raise RuntimeError(
                    f"Chapter {name} failed final literary QA; unresolved hard ids: {', '.join(ids[:20])}"
                )

        completed_chapters.add(name)
        qa_passed_chapters.add(name)
        state["completed_chapters"] = sorted(completed_chapters)
        state["qa_passed_chapters"] = sorted(qa_passed_chapters)

        # Pass 4: continuity memory updates after every completed chapter.
        if memory_updates and mode != "fast":
            chapter_translations = {s.id: translated[s.id] for s in chapter}
            _notify(progress, phase="updating_memory", chapter=name, chapter_index=chapter_index)
            memory = harness.update_memory(chapter, chapter_translations, memory)

        _persist(state_path, state, translated, memory)
        _notify(
            progress,
            phase="chapter_done",
            chapter=name,
            chapter_index=chapter_index,
            chapters=len(chapters),
            completed=sum(s.id in translated for s in source_segments),
            total=total,
        )

    # Final book-wide invariant. No artifact if a target is missing/corrupted.
    final_issues = batch_issues(source_segments, translated, memory)
    final_hard = [i for i in final_issues if i.severity == "hard"]
    missing = [s.id for s in source_segments if s.id not in translated]
    if missing or final_hard:
        ids = missing + [i.id for i in final_hard]
        unique = list(dict.fromkeys(ids))
        _persist(state_path, state, translated, memory)
        raise RuntimeError(f"Final book validation failed for {len(unique)} segments: {', '.join(unique[:20])}")

    _notify(progress, phase="building_book", progress=97, completed=total, total=total)
    save_book(document, translated, output)
    state["final_quality"] = {
        "hard_issues": 0,
        "medium_deterministic_issues": len([i for i in final_issues if i.severity == "medium"]),
        "segments": total,
    }
    _persist(state_path, state, translated, memory)
    _notify(progress, phase="done", progress=100, completed=total, total=total, usage=harness.usage)
    return output
