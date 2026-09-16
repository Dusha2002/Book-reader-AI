from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import evil_v10_full_book as fullbook


_ORIGINAL_RUN_CHAPTER = fullbook._run_chapter
_REVIEW_CHAPTERS: dict[str, dict[str, Any]] = {}


def _bump_int(env: dict[str, str], key: str, minimum: int) -> None:
    try:
        current = int(env.get(key) or "0")
    except ValueError:
        current = 0
    env[key] = str(max(current, minimum))


def _validate_reviewable_checkpoint(
    chapter: str,
    targets,
    paths: dict[str, Path],
) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Accept a complete chapter even when only non-structural QA residue remains.

    HARD quality issues are valuable review signals, but they must not destroy a
    complete book translation. Structural integrity, missing segments, source/map
    mismatches, or empty translations remain fatal and are never accepted here.
    """
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
    if int((report.get("final_integrity") or {}).get("count") or 0) != 0:
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
        if not target or row.get("final_integrity"):
            return None
        translations[sid] = target
    if set(translations) != set(expected_ids):
        return None
    return translations, report


def _hard_payload(report: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(report.get("final_hard") or {})
    count = int(raw.get("count") or 0)
    return {
        "count": count,
        "ids": list(raw.get("ids") or []),
        "issues": list(raw.get("issues") or []),
    }


def _record_review(chapter: str, report: dict[str, Any], *, reason: str = "qa_residue") -> None:
    hard = _hard_payload(report)
    if hard["count"] <= 0 and reason == "qa_residue":
        return
    _REVIEW_CHAPTERS[str(chapter)] = {
        "chapter": str(chapter),
        "reason": reason,
        "hard": hard,
    }


def _sanitized_for_base_gate(chapter: str, report: dict[str, Any]) -> dict[str, Any]:
    """Hide QA residue only from the legacy fatal gate; preserve it for final report."""
    cloned = copy.deepcopy(report)
    hard = _hard_payload(report)
    if hard["count"] > 0:
        _record_review(chapter, report)
        cloned["nonfatal_final_hard"] = hard
        cloned["final_hard"] = {"count": 0, "ids": [], "issues": []}
    return cloned


def _cleanup_large_chapter_files(paths: dict[str, Path]) -> None:
    for key in ("output", "source_txt", "translated_txt"):
        try:
            paths[key].unlink()
        except FileNotFoundError:
            pass


def _run_chapter_tolerant(
    chapter: str,
    targets,
    root: Path,
    bible: Path,
    *,
    include_prelude: bool = False,
) -> tuple[dict[str, str], dict[str, Any], bool]:
    paths = fullbook._chapter_paths(root, chapter)

    strict = fullbook._validate_checkpoint(chapter, targets, paths)
    if strict is not None:
        translations, report = strict
        print(
            f"[v10-full-resume] chapter={chapter} segments={len(targets)} reused=true strict=true",
            flush=True,
        )
        return translations, report, True

    reviewable = _validate_reviewable_checkpoint(chapter, targets, paths)
    if reviewable is not None:
        translations, report = reviewable
        hard = _hard_payload(report)
        _record_review(chapter, report)
        print(
            f"[v10-full-resume-review] chapter={chapter} segments={len(targets)} "
            f"reused=true hard={hard['count']} integrity=0",
            flush=True,
        )
        return translations, _sanitized_for_base_gate(chapter, report), True

    try:
        return _ORIGINAL_RUN_CHAPTER(
            chapter,
            targets,
            root,
            bible,
            include_prelude=include_prelude,
        )
    except RuntimeError as exc:
        # The chapter process intentionally exits non-zero when its strict release QA
        # leaves HARD residue. If the produced checkpoint is otherwise complete and
        # structurally clean, keep it, record the residue, and continue the book.
        reviewable = _validate_reviewable_checkpoint(chapter, targets, paths)
        if reviewable is None:
            raise
        translations, report = reviewable
        hard = _hard_payload(report)
        reason = "qa_residue" if hard["count"] else "complete_checkpoint_after_nonzero_exit"
        _record_review(chapter, report, reason=reason)
        _cleanup_large_chapter_files(paths)
        print(
            f"[v10-full-nonfatal] chapter={chapter} accepted_complete_checkpoint=true "
            f"hard={hard['count']} integrity=0 cause={type(exc).__name__}",
            flush=True,
        )
        return translations, _sanitized_for_base_gate(chapter, report), False


def _finalize_review_metadata() -> None:
    if not fullbook.REPORT.exists():
        return
    report = json.loads(fullbook.REPORT.read_text("utf-8"))
    review_rows = [_REVIEW_CHAPTERS[key] for key in sorted(_REVIEW_CHAPTERS, key=lambda x: int(x) if x.isdigit() else x)]
    hard_total = sum(int((row.get("hard") or {}).get("count") or 0) for row in review_rows)

    by_chapter = {str(row.get("chapter")): row for row in review_rows}
    for chapter_row in report.get("chapter_reports") or []:
        review = by_chapter.get(str(chapter_row.get("chapter")))
        if review:
            count = int((review.get("hard") or {}).get("count") or 0)
            chapter_row["final_hard"] = count
            chapter_row["needs_review"] = count > 0

    report["final_hard_total"] = hard_total
    report["needs_review"] = hard_total > 0
    report["release_ready"] = hard_total == 0
    report["build_status"] = "complete_needs_review" if hard_total else "complete_clean"
    report["qa_policy"] = "nonfatal_quality_residue; structural_integrity_and_missing_segments_remain_fatal"
    report["review_chapters"] = review_rows
    fullbook._atomic_json(fullbook.REPORT, report)

    if fullbook.PROGRESS.exists():
        progress = json.loads(fullbook.PROGRESS.read_text("utf-8"))
        progress["needs_review"] = hard_total > 0
        progress["final_hard_total"] = hard_total
        progress["build_status"] = report["build_status"]
        fullbook._atomic_json(fullbook.PROGRESS, progress)

    if fullbook.PROVENANCE.exists():
        provenance = json.loads(fullbook.PROVENANCE.read_text("utf-8"))
        provenance["qa_policy"] = report["qa_policy"]
        provenance["review_chapters"] = review_rows
        fullbook._atomic_json(fullbook.PROVENANCE, provenance)

    print(
        f"[v10-full-final-status] build={report['build_status']} "
        f"segments={report.get('segments_merged')}/{report.get('segments_total')} "
        f"hard_review={hard_total} integrity={report.get('final_integrity_total', 0)}",
        flush=True,
    )


def main() -> None:
    max_attempts = max(1, min(5, int(os.getenv("BOOKAI_V10_FULL_ATTEMPTS") or "3")))
    fullbook._run_chapter = _run_chapter_tolerant
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        env = dict(os.environ)
        env["BOOKAI_V10_FULL_ATTEMPT"] = str(attempt)

        # Retries are now reserved for genuinely incomplete/technical failures.
        # Residual quality flags with a complete, structurally valid checkpoint are
        # accepted immediately and never cause a wasteful whole-chapter rerun.
        if attempt >= 2:
            _bump_int(env, "BOOKAI_V10_RESIDUAL_DEEP_MAX", 48)
            _bump_int(env, "BOOKAI_V10_RELEASE_DEEP_MAX", 36)
            _bump_int(env, "BOOKAI_V10_EDITORIAL_MAX", 14)
            _bump_int(env, "BOOKAI_GIGACHAT_RETRIES", 4)
        if attempt >= 3:
            _bump_int(env, "BOOKAI_V10_RESIDUAL_DEEP_MAX", 64)
            _bump_int(env, "BOOKAI_V10_RELEASE_DEEP_MAX", 48)
            _bump_int(env, "BOOKAI_V10_EDITORIAL_MAX", 18)
            _bump_int(env, "BOOKAI_GIGACHAT_RETRIES", 5)
        os.environ.update(env)

        print(
            f"[v10-full-self-heal] attempt={attempt}/{max_attempts} "
            "reuse_valid_checkpoints=true qa_residue_nonfatal=true structural_gate=strict",
            flush=True,
        )
        try:
            fullbook.main()
            _finalize_review_metadata()
            print(
                f"[v10-full-self-heal] completed attempt={attempt}/{max_attempts}",
                flush=True,
            )
            return
        except BaseException as exc:  # noqa: BLE001 - orchestrator must checkpoint/retry process failures
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = 4 * attempt
            print(
                f"[v10-full-self-heal] attempt={attempt} incomplete error={type(exc).__name__}; "
                f"retrying only unfinished work after {delay}s",
                flush=True,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    main()
