from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bookai.parsers.base import load_book, save_book
from bookai.pipeline import _chapter_groups, _should_translate

import evil_v10_production_chapter as production


SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Evil for evil.fb2")
OUTPUT = Path(os.getenv("BOOKAI_V10_FULL_OUTPUT") or "Evil_for_Evil_RU_V10_FULL.fb2")
REPORT = Path(os.getenv("BOOKAI_V10_FULL_REPORT") or "v10-evil-full-book-report.json")
PROVENANCE = Path(os.getenv("BOOKAI_V10_FULL_PROVENANCE") or "v10-evil-full-book-provenance.json")
PROGRESS = Path(os.getenv("BOOKAI_V10_FULL_PROGRESS") or "v10-evil-full-book-progress.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def _chapter_paths(root: Path, chapter: str) -> dict[str, Path]:
    safe = re.sub(r"[^0-9A-Za-z_-]+", "-", chapter).strip("-") or "chapter"
    directory = root / "chapters" / safe
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "dir": directory,
        "report": directory / "report.json",
        "map": directory / "map.json",
        "output": directory / "chapter.fb2",
        "source_txt": directory / "source.txt",
        "translated_txt": directory / "translated.txt",
    }


def _validate_checkpoint(chapter: str, targets, paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, Any]] | None:
    if not paths["report"].exists() or not paths["map"].exists():
        return None
    try:
        report = json.loads(paths["report"].read_text("utf-8"))
        rows = json.loads(paths["map"].read_text("utf-8"))
    except Exception:
        return None

    expected_ids = [segment.id for segment in targets]
    expected_source = {segment.id: segment.text for segment in targets}
    if report.get("completed") != len(targets) or report.get("segments") != len(targets):
        return None
    if report.get("missing_ids"):
        return None
    if (report.get("final_integrity") or {}).get("count") != 0:
        return None
    if (report.get("final_hard") or {}).get("count") != 0:
        return None
    selection = report.get("chapter_selection") or {}
    if str(selection.get("matched") or "").strip() != str(chapter):
        return None
    if int(selection.get("segments") or -1) != len(targets):
        return None
    if not isinstance(rows, list) or [str(row.get("id")) for row in rows] != expected_ids:
        return None

    translations: dict[str, str] = {}
    for row in rows:
        sid = str(row.get("id") or "")
        if str(row.get("source") or "") != expected_source.get(sid):
            return None
        target = str(row.get("translation") or "").strip()
        if not target or row.get("final_issues") or row.get("final_integrity"):
            return None
        translations[sid] = target
    if set(translations) != set(expected_ids):
        return None
    return translations, report


def _run_chapter(chapter: str, targets, root: Path, bible: Path) -> tuple[dict[str, str], dict[str, Any], bool]:
    paths = _chapter_paths(root, chapter)
    restored = _validate_checkpoint(chapter, targets, paths)
    if restored is not None:
        translations, report = restored
        print(
            f"[v10-full-resume] chapter={chapter} segments={len(targets)} reused=true",
            flush=True,
        )
        return translations, report, True

    for key in ("report", "map", "output", "source_txt", "translated_txt"):
        try:
            paths[key].unlink()
        except FileNotFoundError:
            pass

    env = dict(os.environ)
    env.update({
        "BOOKAI_SOURCE": str(SOURCE),
        "BOOKAI_MEMORY_SOURCE": str(SOURCE),
        "BOOKAI_CHAPTER_NAME": str(chapter),
        "BOOKAI_V10_BIBLE_CACHE": str(bible),
        "BOOKAI_V10_OUTPUT": str(paths["output"]),
        "BOOKAI_V10_REPORT": str(paths["report"]),
        "BOOKAI_V10_MAP": str(paths["map"]),
        "BOOKAI_V10_SOURCE_TXT": str(paths["source_txt"]),
        "BOOKAI_V10_TRANSLATED_TXT": str(paths["translated_txt"]),
    })

    print(
        f"[v10-full-chapter-start] chapter={chapter} segments={len(targets)} chars={sum(len(s.text) for s in targets)}",
        flush=True,
    )
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "scripts/evil_v10_production_chapter.py"],
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"v10 chapter {chapter} failed with exit={result.returncode} after {elapsed:.1f}s")

    validated = _validate_checkpoint(chapter, targets, paths)
    if validated is None:
        raise RuntimeError(f"v10 chapter {chapter} finished but strict checkpoint validation failed")
    translations, report = validated
    print(
        f"[v10-full-chapter-done] chapter={chapter} segments={len(targets)} seconds={elapsed:.1f}",
        flush=True,
    )

    # The merged book is built from map checkpoints. Large per-chapter FB2/text copies
    # are not needed for resume and only inflate the Actions cache.
    for key in ("output", "source_txt", "translated_txt"):
        try:
            paths[key].unlink()
        except FileNotFoundError:
            pass
    return translations, report, False


def main() -> None:
    run_started = time.perf_counter()
    if not SOURCE.exists() or SOURCE.stat().st_size < 10_000:
        raise RuntimeError(f"invalid/missing source: {SOURCE}")
    source_sha = _sha256(SOURCE)
    source_fp = source_sha[:16]
    root = Path(os.getenv("BOOKAI_V10_FULL_CACHE") or f".bookai-v10-full-h4-{source_fp}")
    root.mkdir(parents=True, exist_ok=True)
    bible = root / "memory" / "book-bible.json"
    bible.parent.mkdir(parents=True, exist_ok=True)

    # Use the exact same selector/hardening as the successful chapter-one bakeoff.
    production._install_production_hardening()
    document = load_book(SOURCE)
    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    if not all_targets:
        raise RuntimeError("source contains no translatable segments")
    groups = _chapter_groups(all_targets)

    chapter_labels: list[str] = []
    for name, _rows in groups:
        label = str(name or "").strip()
        if re.fullmatch(r"\d+", label) and label not in chapter_labels:
            chapter_labels.append(label)
    if not chapter_labels:
        raise RuntimeError(f"no numeric chapters found; groups={[name for name, _ in groups][:50]}")

    chapter_targets: dict[str, list] = {}
    coverage: list[str] = []
    coverage_seen: set[str] = set()
    for label in chapter_labels:
        _all, matched, targets, selection = production._select_numeric_or_named_chapter(document, label)
        if str(matched).strip() != label or not selection.get("boundary_complete"):
            raise RuntimeError(f"chapter selector failed for {label}: {selection}")
        ids = [segment.id for segment in targets]
        duplicated = [sid for sid in ids if sid in coverage_seen]
        if duplicated:
            raise RuntimeError(f"chapter overlap at {label}: {duplicated[:10]}")
        coverage.extend(ids)
        coverage_seen.update(ids)
        chapter_targets[label] = targets

    all_ids = [segment.id for segment in all_targets]
    if coverage != all_ids:
        covered = set(coverage)
        missing = [sid for sid in all_ids if sid not in covered]
        extra = [sid for sid in coverage if sid not in set(all_ids)]
        raise RuntimeError(
            f"numeric chapter coverage is not exact: chapters={len(chapter_labels)} "
            f"covered={len(coverage)}/{len(all_ids)} missing={missing[:20]} extra={extra[:20]}"
        )

    print(
        f"[v10-full-plan] chapters={len(chapter_labels)} segments={len(all_targets)} "
        f"source_sha256={source_sha} labels={chapter_labels}",
        flush=True,
    )

    translations: dict[str, str] = {}
    chapter_reports: list[dict[str, Any]] = []
    reused_chapters: list[str] = []
    completed_labels: list[str] = []

    for index, label in enumerate(chapter_labels, 1):
        chapter_map, chapter_report, reused = _run_chapter(label, chapter_targets[label], root, bible)
        for sid, target in chapter_map.items():
            previous = translations.get(sid)
            if previous is not None and previous != target:
                raise RuntimeError(f"conflicting translation for {sid} while merging chapter {label}")
            translations[sid] = target
        completed_labels.append(label)
        if reused:
            reused_chapters.append(label)
        chapter_reports.append({
            "chapter": label,
            "segments": len(chapter_targets[label]),
            "completed": chapter_report.get("completed"),
            "final_hard": (chapter_report.get("final_hard") or {}).get("count"),
            "final_integrity": (chapter_report.get("final_integrity") or {}).get("count"),
            "timing": chapter_report.get("timing") or {},
            "reused": reused,
        })
        _atomic_json(PROGRESS, {
            "source_sha256": source_sha,
            "hardening": "v10-production-hardening-4",
            "chapters_total": len(chapter_labels),
            "chapters_completed": len(completed_labels),
            "completed_labels": completed_labels,
            "reused_chapters": reused_chapters,
            "segments_total": len(all_targets),
            "segments_completed": len(translations),
            "current_chapter": label,
            "reading_build_complete": False,
        })
        print(
            f"[v10-full-progress] chapter={index}/{len(chapter_labels)} label={label} "
            f"segments={len(translations)}/{len(all_targets)} reused={reused}",
            flush=True,
        )

    if set(translations) != set(all_ids) or len(translations) != len(all_ids):
        missing = [sid for sid in all_ids if sid not in translations]
        extra = [sid for sid in translations if sid not in set(all_ids)]
        raise RuntimeError(f"strict merge failed: missing={missing[:20]} extra={extra[:20]}")

    save_book(document, translations, OUTPUT)
    if not OUTPUT.exists() or OUTPUT.stat().st_size < 10_000:
        raise RuntimeError("merged FB2 was not created or is implausibly small")

    elapsed = time.perf_counter() - run_started
    full_report = {
        "version": "v10-production-hardening-4-full-book",
        "source": str(SOURCE),
        "source_sha256": source_sha,
        "output": str(OUTPUT),
        "reading_build_complete": True,
        "chapters_total": len(chapter_labels),
        "chapters_completed": len(completed_labels),
        "chapter_labels": chapter_labels,
        "reused_chapters": reused_chapters,
        "segments_total": len(all_targets),
        "segments_merged": len(translations),
        "final_hard_total": sum(int(row.get("final_hard") or 0) for row in chapter_reports),
        "final_integrity_total": sum(int(row.get("final_integrity") or 0) for row in chapter_reports),
        "elapsed_seconds": round(elapsed, 2),
        "chapter_reports": chapter_reports,
        "checkpoint_root": str(root),
        "book_bible": str(bible),
    }
    if full_report["final_hard_total"] or full_report["final_integrity_total"]:
        raise RuntimeError(f"strict full-book release gate failed: {full_report}")
    _atomic_json(REPORT, full_report)
    _atomic_json(PROVENANCE, {
        "source_sha256": source_sha,
        "pipeline": "v10-production-hardening-4",
        "selector": "production-bare-numeric-v4",
        "chapters": chapter_reports,
        "segment_ids": all_ids,
        "book_bible": str(bible),
    })
    _atomic_json(PROGRESS, {
        "source_sha256": source_sha,
        "hardening": "v10-production-hardening-4",
        "chapters_total": len(chapter_labels),
        "chapters_completed": len(completed_labels),
        "completed_labels": completed_labels,
        "reused_chapters": reused_chapters,
        "segments_total": len(all_targets),
        "segments_completed": len(translations),
        "reading_build_complete": True,
        "output": str(OUTPUT),
    })
    print(
        f"[v10-full-done] chapters={len(completed_labels)} segments={len(translations)} "
        f"bytes={OUTPUT.stat().st_size} elapsed={elapsed:.1f}s hard=0 integrity=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
