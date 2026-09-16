from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bookai.parsers.base import load_book, save_book
from bookai.pipeline import _chapter_groups, _should_translate

SOURCE = Path(os.getenv("BOOKAI_FULL_SOURCE") or "production_source.fb2")
OUTPUT = Path(os.getenv("BOOKAI_FULL_OUTPUT") or f"production_translation_ru{SOURCE.suffix.lower()}")
REPORT = Path(os.getenv("BOOKAI_FULL_REPORT") or "production-full-book.json")
PROVENANCE = Path(os.getenv("BOOKAI_FULL_PROVENANCE") or "production-full-provenance.json")
ENTRYPOINT = Path("scripts/production_book_translation.py")


def _source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "chapter"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def _load_map(path: Path) -> list[dict[str, Any]]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise RuntimeError(f"translation map is not a list: {path}")
    return [row for row in obj if isinstance(row, dict)]


def _validate_map(path: Path, chapter_segments) -> list[dict[str, Any]]:
    rows = _load_map(path)
    expected = {segment.id: str(segment.text or "") for segment in chapter_segments}
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get("id") or "")
        if sid not in expected:
            continue
        if str(row.get("source") or "") != expected[sid] or not str(row.get("translation") or "").strip():
            raise RuntimeError(f"cached map is stale/invalid for {sid}: {path}")
        seen.add(sid)
    missing = sorted(set(expected) - seen)
    if missing:
        raise RuntimeError(f"cached map misses {len(missing)} ids: {missing[:12]}")
    return rows


def _chapter_paths(workdir: Path, slug: str) -> dict[str, Path]:
    return {
        "map": workdir / f"{slug}-translation-map.json",
        "provenance": workdir / f"{slug}-provenance.json",
        "quality": workdir / f"{slug}-quality.json",
        "result": workdir / f"{slug}-result.json",
    }


def _root_paths(slug: str) -> dict[str, Path]:
    return {
        "map": Path(f"chapter-v9-{slug}-translation-map.json"),
        "provenance": Path(f"chapter-v9-{slug}-provenance.json"),
        "quality": Path(f"chapter-v9-{slug}.json"),
    }


def _cached_chapter(name: str, chapter_segments, *, fingerprint: str, workdir: Path) -> dict[str, Any] | None:
    slug = _slug(name)
    paths = _chapter_paths(workdir, slug)
    if not paths["result"].exists() or not paths["map"].exists():
        return None
    try:
        result = _load_json(paths["result"])
        if not isinstance(result, dict):
            return None
        if result.get("source_fingerprint") != fingerprint or result.get("chapter") != name:
            return None
        _validate_map(paths["map"], chapter_segments)
    except Exception as exc:
        print(f"[production-full-resume] invalidate chapter={name!r} reason={type(exc).__name__}: {exc}", flush=True)
        return None
    result = dict(result)
    result.update({
        "reused": True,
        "map": str(paths["map"]),
        "provenance": str(paths["provenance"]) if paths["provenance"].exists() else None,
    })
    print(f"[production-full-resume] reuse chapter={name!r} segments={len(chapter_segments)}", flush=True)
    return result


def _run_chapter(name: str, chapter_segments, timeout_seconds: int, *, fingerprint: str, workdir: Path) -> dict[str, Any]:
    slug = _slug(name)
    cached = _cached_chapter(name, chapter_segments, fingerprint=fingerprint, workdir=workdir)
    if cached is not None:
        return cached

    root = _root_paths(slug)
    for path in root.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    env = os.environ.copy()
    env["BOOKAI_SOURCE"] = str(SOURCE)
    env["BOOKAI_CHAPTER_NAME"] = name
    env["BOOKAI_CHAPTER_OUTPUT"] = str(workdir / f"{slug}-chapter-preview{SOURCE.suffix.lower()}")
    started = time.perf_counter()
    print(f"[production-full] ===== START {name} =====", flush=True)
    try:
        result = subprocess.run([sys.executable, str(ENTRYPOINT)], env=env, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"chapter timeout after {timeout_seconds}s: {name}") from exc

    elapsed = round(time.perf_counter() - started, 2)
    if result.returncode != 0:
        raise RuntimeError(f"production chapter failed rc={result.returncode}: {name}")
    if not root["map"].exists():
        raise RuntimeError(f"production chapter produced no translation map: {root['map']}")
    _validate_map(root["map"], chapter_segments)

    quality: dict[str, Any] = {}
    if root["quality"].exists():
        try:
            raw = _load_json(root["quality"])
            quality = raw if isinstance(raw, dict) else {}
        except Exception:
            pass
    status = str(quality.get("status") or "unknown")
    semantic = dict((quality.get("final_quality") or {}).get("semantic_stats") or {})
    release_residual = list(semantic.get("production_release_residual_after") or [])
    hard_issues = int((quality.get("final_quality") or {}).get("hard_issues") or 0)

    paths = _chapter_paths(workdir, slug)
    for key in ("map", "provenance", "quality"):
        if root[key].exists():
            shutil.copy2(root[key], paths[key])
    chapter_result = {
        "chapter": name,
        "slug": slug,
        "source_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "status": status,
        "hard_issues": hard_issues,
        "production_release_residual_after": release_residual,
        "segment_count": len(chapter_segments),
        "map": str(paths["map"]),
        "provenance": str(paths["provenance"]) if paths["provenance"].exists() else None,
        "reused": False,
    }
    _write_json(paths["result"], chapter_result)
    print(
        f"[production-full] ===== DONE {name} elapsed={elapsed}s status={status} "
        f"hard={hard_issues} release_residual={len(release_residual)} =====",
        flush=True,
    )
    return chapter_result


def _report_payload(*, fingerprint: str, workdir: Path, chapters_total: int, chapter_results: list[dict[str, Any]], segments_total: int, segments_merged: int, started: float, status: str = "running", reading_build_complete: bool = False, error: str | None = None) -> dict[str, Any]:
    review = [
        row["chapter"] for row in chapter_results
        if row.get("status") == "needs_review" or row.get("production_release_residual_after")
    ]
    payload: dict[str, Any] = {
        "status": status,
        "reading_build_complete": reading_build_complete,
        "strategy": "literary-production-v1",
        "source": str(SOURCE),
        "source_fingerprint": fingerprint,
        "workdir": str(workdir),
        "output": str(OUTPUT),
        "chapters_total": chapters_total,
        "chapters_completed": len(chapter_results),
        "chapters_reused": sum(bool(row.get("reused")) for row in chapter_results),
        "chapters_translated": sum(not bool(row.get("reused")) for row in chapter_results),
        "segments_total": segments_total,
        "segments_merged": segments_merged,
        "review_required_chapters": review,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "chapter_results": chapter_results,
    }
    if chapter_results:
        payload["last_chapter"] = chapter_results[-1]["chapter"]
    if error:
        payload["error"] = error
    return payload


def _write_report(payload: dict[str, Any], workdir: Path) -> None:
    _write_json(REPORT, payload)
    _write_json(workdir / "checkpoint.json", payload)
    compact = {key: payload.get(key) for key in (
        "status", "chapters_completed", "chapters_total", "chapters_reused",
        "chapters_translated", "segments_merged", "segments_total", "last_chapter", "elapsed_seconds"
    ) if key in payload}
    print("[production-full] " + json.dumps(compact, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not ENTRYPOINT.exists():
        raise FileNotFoundError(ENTRYPOINT)

    fingerprint = _source_fingerprint(SOURCE)
    workdir = Path(os.getenv("BOOKAI_FULL_WORKDIR") or f".bookai-full-{fingerprint}")
    workdir.mkdir(parents=True, exist_ok=True)

    document = load_book(SOURCE)
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    chapters = _chapter_groups(targets)
    if not chapters:
        raise RuntimeError("source book has no translatable chapters")

    timeout_seconds = max(180, int(os.getenv("BOOKAI_FULL_CHAPTER_TIMEOUT_SECONDS") or "1200"))
    started = time.perf_counter()
    chapter_results: list[dict[str, Any]] = []
    merged: dict[str, str] = {}
    provenance_rows: list[dict[str, Any]] = []
    target_ids = {segment.id for segment in targets}

    _write_report(_report_payload(
        fingerprint=fingerprint, workdir=workdir, chapters_total=len(chapters), chapter_results=chapter_results,
        segments_total=len(targets), segments_merged=0, started=started,
    ), workdir)

    try:
        for name, chapter_segments in chapters:
            result = _run_chapter(name, chapter_segments, timeout_seconds, fingerprint=fingerprint, workdir=workdir)
            rows = _validate_map(Path(result["map"]), chapter_segments)
            chapter_ids = {segment.id for segment in chapter_segments}
            seen: set[str] = set()
            for row in rows:
                sid = str(row.get("id") or "")
                text = str(row.get("translation") or "").strip()
                if sid not in chapter_ids or not text:
                    continue
                previous = merged.get(sid)
                if previous and previous != text:
                    raise RuntimeError(f"conflicting translations for {sid}")
                merged[sid] = text
                seen.add(sid)
            missing = sorted(chapter_ids - seen)
            if missing:
                raise RuntimeError(f"{name} map misses {len(missing)} ids: {missing[:12]}")

            prov_path = result.get("provenance")
            if prov_path and Path(prov_path).exists():
                try:
                    prov = _load_json(Path(prov_path))
                    if isinstance(prov, list):
                        provenance_rows.extend(row for row in prov if isinstance(row, dict))
                except Exception:
                    pass
            chapter_results.append(result)
            _write_report(_report_payload(
                fingerprint=fingerprint, workdir=workdir, chapters_total=len(chapters), chapter_results=chapter_results,
                segments_total=len(targets), segments_merged=len(merged), started=started,
            ), workdir)

        missing_all = sorted(target_ids - set(merged))
        extra = sorted(set(merged) - target_ids)
        if missing_all or extra:
            raise RuntimeError(
                f"full merge invariant failed: missing={len(missing_all)} extra={len(extra)} "
                f"sample_missing={missing_all[:12]} sample_extra={extra[:12]}"
            )

        save_book(document, merged, OUTPUT)
        _write_json(PROVENANCE, provenance_rows)
        review = [
            row["chapter"] for row in chapter_results
            if row.get("status") == "needs_review" or row.get("production_release_residual_after")
        ]
        payload = _report_payload(
            fingerprint=fingerprint, workdir=workdir, chapters_total=len(chapters), chapter_results=chapter_results,
            segments_total=len(targets), segments_merged=len(merged), started=started,
            status="needs_review" if review else "complete", reading_build_complete=True,
        )
        payload["provenance"] = str(PROVENANCE)
        _write_report(payload, workdir)
        return 0
    except BaseException as exc:
        _write_report(_report_payload(
            fingerprint=fingerprint, workdir=workdir, chapters_total=len(chapters), chapter_results=chapter_results,
            segments_total=len(targets), segments_merged=len(merged), started=started,
            status="partial", reading_build_complete=False, error=f"{type(exc).__name__}: {exc}",
        ), workdir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
