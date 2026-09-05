from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from bookai.parsers.base import load_book
from bookai.pipeline import (
    _analysis_sample,
    _chapter_groups,
    _context_for,
    _should_translate,
    _translated_context,
)
from bookai.quality import batch_issues, hard_ids
from bookai.reference_harness import build_reference_harness
from literary_benchmark import BENCHMARK
from reference_quality_gate import REFERENCE_RU, _candidate_from_fb2, _judge, _summarize


SOURCE = Path("Devices_and_Desires.fb2")


def _reason_map(findings) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for finding in findings:
        grouped.setdefault(finding.id, []).append(finding.reason)
    return {sid: "; ".join(items) for sid, items in grouped.items()}


def _generate_scene(harness, source_segments, by_id, memory, name: str, ids: list[str]) -> dict:
    segments = [by_id[sid] for sid in ids]
    before, after = _context_for(source_segments, segments, radius=3)
    brief = harness.chapter_brief(segments, memory)
    local_memory = replace(
        memory,
        rolling_summary=(memory.rolling_summary + "\nCURRENT BENCHMARK SCENE: " + brief)[-12000:],
    )

    draft = harness.translate(segments, local_memory, context_before=before, context_after=after)
    draft_hard = hard_ids(batch_issues(segments, draft, local_memory))
    if draft_hard:
        raise RuntimeError(f"{name}: deterministic draft QA failed: {sorted(draft_hard)}")

    context = [
        {"id": s.id, "source": s.text, "translation": ""}
        for s in before + after
    ]
    final = harness.polish(segments, draft, local_memory, context=context)
    polish_hard = hard_ids(batch_issues(segments, final, local_memory))
    if polish_hard:
        raise RuntimeError(f"{name}: deterministic polish QA failed: {sorted(polish_hard)}")

    findings = harness.gate_findings(segments, final, local_memory)
    history = []
    for repair_round in range(3):
        if not findings:
            break
        history.append([asdict(f) for f in findings])
        target_ids = {f.id for f in findings}
        targets = [s for s in segments if s.id in target_ids]
        reasons = _reason_map(findings)
        edit_context = _translated_context(source_segments, targets, final, radius=3)
        edited = harness.edit(
            targets,
            final,
            local_memory,
            reasons=reasons,
            context=edit_context,
        )
        if hard_ids(batch_issues(targets, edited, local_memory)):
            raise RuntimeError(f"{name}: Pro repair introduced deterministic hard QA failure")
        final.update(edited)
        findings = harness.gate_findings(targets, final, local_memory)

    # Internal critics remain useful repair signals, but after three Pro repair
    # rounds the independent user-reference judge is authoritative for literary
    # acceptance. Deterministic invariants remain hard blockers.
    deterministic = batch_issues(segments, final, local_memory)
    deterministic_hard = [i for i in deterministic if i.severity == "hard"]
    if deterministic_hard:
        raise RuntimeError(
            f"{name}: unresolved deterministic hard QA: {[i.id for i in deterministic_hard]}"
        )

    return {
        "scene_brief": brief,
        "segments": [
            {
                "id": s.id,
                "source": s.text,
                "draft": draft[s.id],
                "final": final[s.id],
            }
            for s in segments
        ],
        "critic_history": history,
        "remaining_findings": [asdict(f) for f in findings],
        "deterministic_issues": [asdict(i) for i in deterministic],
    }


def _candidate_from_generated_v2() -> tuple[dict[str, str], dict]:
    doc = load_book(SOURCE)
    source_segments = [s for s in doc.segments if _should_translate(s.text)]
    by_id = {s.id: s for s in source_segments}
    missing = [sid for ids in BENCHMARK.values() for sid in ids if sid not in by_id]
    if missing:
        raise RuntimeError(f"Benchmark ids missing from source: {missing}")

    harness = build_reference_harness()
    chapters = _chapter_groups(source_segments)
    memory = harness.analyze(_analysis_sample(chapters, 90000))

    candidates: dict[str, str] = {}
    reports = {}
    for name, ids in BENCHMARK.items():
        print(f"[reference-preflight] generating {name}", flush=True)
        report = _generate_scene(harness, source_segments, by_id, memory, name, ids)
        reports[name] = report
        for row in report["segments"]:
            candidates[row["id"]] = row["final"]
        print(
            f"[reference-preflight] finished {name} remaining_internal_findings={len(report['remaining_findings'])}",
            flush=True,
        )

    return candidates, {"generation": reports, "generation_usage": harness.usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference-quality-report.json"))
    args = parser.parse_args()

    benchmark_ids = {sid for group in BENCHMARK.values() for sid in group}
    if set(REFERENCE_RU) != benchmark_ids:
        raise RuntimeError("Reference benchmark ids do not match literary benchmark ids")

    if args.candidate:
        candidates, metadata = _candidate_from_fb2(args.candidate)
        mode = "postflight"
    else:
        candidates, metadata = _candidate_from_generated_v2()
        mode = "preflight"

    judgement, judge_usage = _judge(candidates)
    summary = _summarize(judgement)
    report = {
        "mode": mode,
        "reference_used_in_generation": False,
        "summary": summary,
        "judgement": judgement,
        "candidates": candidates,
        "judge_usage": judge_usage,
        **metadata,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("[reference-gate] " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[reference-gate] report={args.output}", flush=True)
    if not summary["passed"]:
        raise RuntimeError("Reference literary benchmark gate failed")


if __name__ == "__main__":
    main()
