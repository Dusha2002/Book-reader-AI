from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from bookai.harness import TranslationHarness
from bookai.pipeline import _analysis_sample, _batches, _chapter_groups, _context_for, _translated_context, _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.parsers.base import load_book

SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("literary_benchmark_output.json")

# Deliberately chosen BEFORE looking at any generated output. These passages
# exercise dry humour/voice, battle narration, and the Melancton passage that
# the previous pipeline semantically corrupted. No reference translation is
# present in this repository or supplied to the model.
BENCHMARK = {
    "chapter_1_opening": ["s000008", "s000009", "s000010", "s000011", "s000012"],
    "chapter_3_opening": ["s000343", "s000344", "s000345"],
    "chapter_22_opening": ["s003324", "s003325", "s003326"],
}


def run_group(harness: TranslationHarness, source_segments, by_id, memory, name: str, ids: list[str]):
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
        raise RuntimeError(f"{name}: draft hard QA failed: {sorted(draft_hard)}")

    context = [
        {"id": s.id, "source": s.text, "translation": ""}
        for s in before + after
    ]
    final = harness.polish(segments, draft, local_memory, context=context)
    polish_hard = hard_ids(batch_issues(segments, final, local_memory))
    if polish_hard:
        raise RuntimeError(f"{name}: polish hard QA failed: {sorted(polish_hard)}")

    findings = harness.gate_findings(segments, final, local_memory)
    for repair_round in range(2):
        if not findings:
            break
        finding_by_id = {f.id: f for f in findings}
        hard_segments = [s for s in segments if s.id in finding_by_id and finding_by_id[s.id].severity == "hard"]
        if hard_segments:
            current = {s.id: final[s.id] for s in hard_segments}
            alt = harness.alternative(hard_segments, local_memory)
            if not hard_ids(batch_issues(hard_segments, alt, local_memory)):
                final.update(harness.choose(hard_segments, current, alt, local_memory))

        targets = [s for s in segments if s.id in finding_by_id]
        reasons = {s.id: finding_by_id[s.id].reason for s in targets}
        edit_context = _translated_context(source_segments, targets, final, radius=3)
        edited = harness.edit(targets, final, local_memory, reasons=reasons, context=edit_context)
        if not hard_ids(batch_issues(targets, edited, local_memory)):
            final.update(edited)
        findings = harness.gate_findings(targets, final, local_memory)

    deterministic = batch_issues(segments, final, local_memory)
    unresolved_hard = [f for f in findings if f.severity == "hard"]
    if any(i.severity == "hard" for i in deterministic) or unresolved_hard:
        raise RuntimeError(f"{name}: unresolved hard literary QA")

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
        "remaining_findings": [asdict(f) for f in findings],
        "deterministic_issues": [asdict(i) for i in deterministic],
    }


def main() -> None:
    doc = load_book(SOURCE)
    source_segments = [s for s in doc.segments if _should_translate(s.text)]
    by_id = {s.id: s for s in source_segments}
    missing = [sid for ids in BENCHMARK.values() for sid in ids if sid not in by_id]
    if missing:
        raise RuntimeError(f"Benchmark ids missing from source: {missing}")

    harness = TranslationHarness.from_env()
    chapters = _chapter_groups(source_segments)
    memory = harness.analyze(_analysis_sample(chapters, 90000))

    result = {
        "model_ceiling": "deepseek/deepseek-v4-flash-0731",
        "reference_used_in_generation": False,
        "scenes": {},
    }
    for name, ids in BENCHMARK.items():
        print(f"[benchmark] starting {name}", flush=True)
        result["scenes"][name] = run_group(harness, source_segments, by_id, memory, name, ids)
        print(f"[benchmark] finished {name}", flush=True)

    result["usage"] = harness.usage
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print("[benchmark] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)
    print(f"[benchmark] output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
