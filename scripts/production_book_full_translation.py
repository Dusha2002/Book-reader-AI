from __future__ import annotations

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

SOURCE = Path(os.getenv("BOOKAI_FULL_SOURCE") or "production_source.fb2")
OUTPUT = Path(
    os.getenv("BOOKAI_FULL_OUTPUT")
    or f"production_translation_ru{SOURCE.suffix.lower()}"
)
REPORT = Path(os.getenv("BOOKAI_FULL_REPORT") or "production-full-book.json")
PROVENANCE = Path(os.getenv("BOOKAI_FULL_PROVENANCE") or "production-full-provenance.json")
ENTRYPOINT = Path("scripts/production_book_translation.py")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "chapter"


def _write_report(payload: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print("[production-full] " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _load_map(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text("utf-8"))
    if not isinstance(obj, list):
        raise RuntimeError(f"translation map is not a list: {path}")
    return [row for row in obj if isinstance(row, dict)]


def _chapter_quality(slug: str) -> dict[str, Any]:
    path = Path(f"chapter-v9-{slug}.json")
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text("utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _run_chapter(name: str, timeout_seconds: int) -> dict[str, Any]:
    slug = _slug(name)
    map_path = Path(f"chapter-v9-{slug}-translation-map.json")
    prov_path = Path(f"chapter-v9-{slug}-provenance.json")

    env = os.environ.copy()
    env["BOOKAI_SOURCE"] = str(SOURCE)
    env["BOOKAI_CHAPTER_NAME"] = name
    started = time.perf_counter()
    print(f"[production-full] ===== START {name} =====", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, str(ENTRYPOINT)],
            env=env,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"chapter timeout after {timeout_seconds}s: {name}") from exc

    elapsed = round(time.perf_counter() - started, 2)
    if result.returncode != 0:
        raise RuntimeError(f"production chapter failed rc={result.returncode}: {name}")
    if not map_path.exists():
        raise RuntimeError(f"production chapter produced no translation map: {map_path}")

    quality = _chapter_quality(slug)
    status = str(quality.get("status") or "unknown")
    semantic = dict((quality.get("final_quality") or {}).get("semantic_stats") or {})
    release_residual = list(semantic.get("production_release_residual_after") or [])
    hard_issues = int((quality.get("final_quality") or {}).get("hard_issues") or 0)

    print(
        f"[production-full] ===== DONE {name} elapsed={elapsed}s status={status} "
        f"hard={hard_issues} release_residual={len(release_residual)} =====",
        flush=True,
    )
    return {
        "chapter": name,
        "slug": slug,
        "elapsed_seconds": elapsed,
        "status": status,
        "hard_issues": hard_issues,
        "production_release_residual_after": release_residual,
        "map": str(map_path),
        "provenance": str(prov_path) if prov_path.exists() else None,
    }


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not ENTRYPOINT.exists():
        raise FileNotFoundError(ENTRYPOINT)

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

    _write_report(
        {
            "status": "running",
            "strategy": "literary-production-v1",
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "chapters_total": len(chapters),
            "chapters_completed": 0,
            "segments_total": len(targets),
            "segments_merged": 0,
        }
    )

    try:
        for index, (name, chapter_segments) in enumerate(chapters, 1):
            result = _run_chapter(name, timeout_seconds)
            rows = _load_map(Path(result["map"]))
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
                    prov = json.loads(Path(prov_path).read_text("utf-8"))
                    if isinstance(prov, list):
                        provenance_rows.extend(row for row in prov if isinstance(row, dict))
                except Exception:
                    pass

            chapter_results.append(result)
            _write_report(
                {
                    "status": "running",
                    "strategy": "literary-production-v1",
                    "source": str(SOURCE),
                    "output": str(OUTPUT),
                    "chapters_total": len(chapters),
                    "chapters_completed": index,
                    "segments_total": len(targets),
                    "segments_merged": len(merged),
                    "last_chapter": name,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "review_required_chapters": [
                        row["chapter"]
                        for row in chapter_results
                        if row.get("status") == "needs_review"
                        or row.get("production_release_residual_after")
                    ],
                }
            )

        missing_all = sorted(target_ids - set(merged))
        extra = sorted(set(merged) - target_ids)
        if missing_all or extra:
            raise RuntimeError(
                f"full merge invariant failed: missing={len(missing_all)} extra={len(extra)} "
                f"sample_missing={missing_all[:12]} sample_extra={extra[:12]}"
            )

        save_book(document, merged, OUTPUT)
        PROVENANCE.write_text(json.dumps(provenance_rows, ensure_ascii=False, indent=2), "utf-8")
        review = [
            row["chapter"]
            for row in chapter_results
            if row.get("status") == "needs_review"
            or row.get("production_release_residual_after")
        ]
        final_status = "needs_review" if review else "complete"
        payload = {
            "status": final_status,
            "reading_build_complete": True,
            "strategy": "literary-production-v1",
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "chapters_total": len(chapters),
            "chapters_completed": len(chapter_results),
            "segments_total": len(targets),
            "segments_merged": len(merged),
            "review_required_chapters": review,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "chapter_results": chapter_results,
            "provenance": str(PROVENANCE),
        }
        _write_report(payload)
        return 0
    except BaseException as exc:
        payload = {
            "status": "partial",
            "reading_build_complete": False,
            "strategy": "literary-production-v1",
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "chapters_total": len(chapters),
            "chapters_completed": len(chapter_results),
            "segments_total": len(targets),
            "segments_merged": len(merged),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "chapter_results": chapter_results,
        }
        _write_report(payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
