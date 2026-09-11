from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import full_reference_translation as fullref
import hybrid_reference_translation as hybrid
from bookai.gigachat_mt import GigaChatLightningBackend, benchmark_gigachat
from bookai.hybrid_mt import choose_probe_segments, dumps_report, routing_summary
from bookai.parsers.base import load_book
from bookai.pipeline import PIPELINE_VERSION, _chapter_groups, _should_translate
from bookai.quality import batch_issues

SOURCE = Path("Devices_and_Desires.fb2")
CHAPTER_NAME = os.getenv("BOOKAI_CHAPTER_NAME") or "Chapter Nine"
CHAPTER_SLUG = re.sub(r"[^a-z0-9]+", "-", CHAPTER_NAME.casefold()).strip("-") or "chapter"
OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL.fb2")
CACHE = Path(f".bookai-cache-chapter-eval-v2-{CHAPTER_SLUG}")
PROGRESS = Path("chapter-eval-progress.json")
PROBE = Path("chapter-eval-probe.json")
ROUTING = Path("chapter-eval-routing.json")
REPORT = Path("chapter-eval.json")
SOURCE_TXT = Path("chapter-eval-source.txt")
TRANSLATED_TXT = Path("chapter-eval-translated.txt")
MAP_JSON = Path("chapter-eval-translation-map.json")


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _configure_modules() -> None:
    hybrid.SOURCE = SOURCE
    hybrid.OUTPUT = OUTPUT
    hybrid.CACHE = CACHE
    hybrid.PROBE_REPORT = PROBE
    hybrid.ROUTING_REPORT = ROUTING
    fullref.SOURCE = SOURCE
    fullref.OUTPUT = OUTPUT
    fullref.CACHE = CACHE
    fullref.PROGRESS_REPORT = PROGRESS


def _select_chapter(document) -> tuple[str, list]:
    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    wanted = _norm(CHAPTER_NAME)
    exact = [(name, rows) for name, rows in groups if _norm(name) == wanted]
    if not exact:
        exact = [(name, rows) for name, rows in groups if wanted in _norm(name)]
    if len(exact) != 1:
        names = [name for name, _ in groups]
        raise RuntimeError(f"Expected one chapter matching {wanted!r}; found {len(exact)}. Available={names}")
    return exact[0]


def _write_exports(chapter_name: str, targets: list, state: dict, memory, *, status: str, extra: dict | None = None) -> None:
    translations = {
        str(k): str(v)
        for k, v in dict(state.get("translations") or {}).items()
        if isinstance(v, str) and v.strip()
    }
    SOURCE_TXT.write_text("\n\n".join(segment.text for segment in targets), "utf-8")
    TRANSLATED_TXT.write_text(
        "\n\n".join(translations.get(segment.id, f"[UNTRANSLATED {segment.id}]\n{segment.text}") for segment in targets),
        "utf-8",
    )
    mapping = [
        {
            "id": segment.id,
            "chapter": segment.chapter,
            "source": segment.text,
            "translation": translations.get(segment.id),
            "length_ratio": round(len(translations.get(segment.id, "")) / max(1, len(segment.text)), 3),
        }
        for segment in targets
    ]
    MAP_JSON.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), "utf-8")

    completed = sum(segment.id in translations for segment in targets)
    resolved = [segment for segment in targets if segment.id in translations]
    issues = batch_issues(resolved, translations, memory) if resolved else []
    issue_counts = Counter(f"{issue.severity}:{issue.code}" for issue in issues)
    ratios = [row for row in mapping if row["translation"]]
    ratios.sort(key=lambda row: abs(row["length_ratio"] - 0.78), reverse=True)

    report = {
        "chapter": chapter_name,
        "status": status,
        "segments": len(targets),
        "completed_segments": completed,
        "completion_percent": round(completed / max(1, len(targets)) * 100, 2),
        "source_chars": sum(len(segment.text) for segment in targets),
        "translated_chars": sum(len(translations.get(segment.id, "")) for segment in targets),
        "overall_length_ratio": round(
            sum(len(translations.get(segment.id, "")) for segment in targets)
            / max(1, sum(len(segment.text) for segment in targets)),
            3,
        ),
        "architecture": {
            "primary": "GigaChat-3-Lightning strict-json-schema",
            "reasoning_analysis_refinement": "deepseek/deepseek-v4.1-flash",
            "external_pro_judge": False,
            "full_paid_fallback": False,
        },
        "state_status": state.get("status"),
        "hybrid_stats": state.get("hybrid_stats") or {},
        "gigachat_usage": state.get("gigachat_usage") or {},
        "final_quality": state.get("final_quality") or {},
        "post_export_issue_counts": dict(issue_counts),
        "largest_length_outliers": [
            {"id": row["id"], "ratio": row["length_ratio"], "source": row["source"][:160], "translation": (row["translation"] or "")[:220]}
            for row in ratios[:12]
        ],
        "extra": extra or {},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_modules()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    document = load_book(SOURCE)
    chapter_name, targets = _select_chapter(document)
    chapters = [(chapter_name, targets)]
    total_chars = sum(len(segment.text) for segment in targets)
    print(
        "[chapter-eval] "
        + json.dumps(
            {
                "chapter": chapter_name,
                "segments": len(targets),
                "source_chars": total_chars,
                "first_id": targets[0].id if targets else None,
                "last_id": targets[-1].id if targets else None,
                "cache": str(CACHE),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    routing = routing_summary(targets)
    if "hymt" in routing:
        routing["bulk"] = routing.pop("hymt")
    ROUTING.write_text(dumps_report(routing), "utf-8")

    bulk = GigaChatLightningBackend()
    if not bulk.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing")

    base_memory = hybrid._base_memory()
    probe_segments = choose_probe_segments(
        targets,
        count=max(4, min(len(targets), int(os.getenv("BOOKAI_BULK_PROBE_SEGMENTS") or "8"))),
    )
    probe = benchmark_gigachat(
        bulk,
        probe_segments,
        base_memory,
        source_segments=targets,
        total_source_chars=total_chars,
    )
    qa_translated, qa_errors = bulk.translate_many(probe_segments, base_memory, source_segments=targets)
    qa_bad = [
        segment.id
        for segment in probe_segments
        if segment.id in qa_translated and hybrid._candidate_bad(segment, qa_translated[segment.id], base_memory)
    ]
    probe.update(
        qa_bad=len(qa_bad),
        qa_bad_ids=qa_bad,
        qa_errors=qa_errors,
        token_usage_after_qa_probe=bulk.usage.as_dict(),
    )
    PROBE.write_text(dumps_report(probe), "utf-8")
    print("[chapter-eval-probe] " + json.dumps(probe, ensure_ascii=False, sort_keys=True), flush=True)

    successes = int(probe.get("probe_success") or 0)
    min_success = max(2, len(probe_segments) - 1)
    max_qa_bad = max(2, math.floor(len(probe_segments) * 0.5))
    if successes < min_success or len(qa_bad) > max_qa_bad:
        raise RuntimeError(
            f"GigaChat probe failed for chapter evaluation: success={successes}/{len(probe_segments)} "
            f"qa_bad={len(qa_bad)}/{len(probe_segments)}"
        )

    harness = hybrid.build_reference_harness()
    resume = hybrid._sanitize_resume_cache(SOURCE, CACHE)
    print("[chapter-eval-resume] " + json.dumps(resume, ensure_ascii=False, sort_keys=True), flush=True)
    state = hybrid._cached_state(SOURCE, CACHE)
    if not state or state.get("pipeline_version") != PIPELINE_VERSION:
        state = {
            "pipeline_version": PIPELINE_VERSION,
            "translations": {},
            "completed_chapters": [],
            "chapter_briefs": {},
            "polished_chapters": [],
            "qa_passed_chapters": [],
        }
    state["chapter_evaluation"] = chapter_name
    state["hybrid_probe"] = probe
    state["routing_summary"] = routing
    state["bulk_backend"] = bulk.backend_name
    state["architecture"] = {
        "primary_translation": "GigaChat-3-Lightning-strict-json-schema",
        "reasoning_analysis": "deepseek/deepseek-v4.1-flash",
        "hard_translation": "deepseek/deepseek-v4.1-flash",
        "literary_refinement": "deepseek/deepseek-v4.1-flash",
        "external_pro_judge": None,
        "progressive_publish": True,
        "full_paid_fallback": False,
    }

    memory = base_memory
    try:
        result = hybrid._run_full(harness, bulk, document, targets, chapters, state)
        latest = hybrid._cached_state(SOURCE, CACHE)
        memory = hybrid._base_memory()
        _write_exports(
            chapter_name,
            targets,
            latest,
            memory,
            status="complete",
            extra={"run_result": result, "deepseek_usage": harness.usage},
        )
        print(
            "[chapter-eval-done] "
            + json.dumps(
                {
                    "chapter": chapter_name,
                    "segments": len(targets),
                    "gigachat_usage": bulk.usage.as_dict(),
                    "deepseek_usage": harness.usage.get("total", {}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except BaseException as exc:
        latest = hybrid._cached_state(SOURCE, CACHE)
        _write_exports(
            chapter_name,
            targets,
            latest,
            memory,
            status="partial",
            extra={"error": f"{type(exc).__name__}: {exc}", "deepseek_usage": getattr(harness, "usage", {})},
        )
        raise


if __name__ == "__main__":
    main()
