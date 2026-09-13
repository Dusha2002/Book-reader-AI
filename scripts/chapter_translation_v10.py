from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from bookai.llm import OpenAICompatibleProvider
from bookai.parsers.base import load_book, save_book
from bookai.pipeline import _chapter_groups, _should_translate
from bookai.v10 import DeepSeekSemanticSpecialist, GigaSpanPatcher, issue_summary
from bookai.v10_dialogue import DialogueDiscourseGuard
from bookai.v10_integrity import SegmentIntegrityGate
from bookai.v10_local_repair import GigaLocalRewriter
from bookai.v10_name_canon import V9ADSourceOnlyBookBibleBuilder
from bookai.v10_quality import V10QualityQA
from bookai.v10_speaker import DialogueSpeakerContinuityGuard
from bookai.v10_transport import RobustTaggedPrimaryTransport


SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
CHAPTER_NAME = os.getenv("BOOKAI_CHAPTER_NAME") or "Chapter One"
SLUG = re.sub(r"[^a-z0-9]+", "-", CHAPTER_NAME.casefold()).strip("-") or "chapter"
OUTPUT = Path(f"Devices_and_Desires_RU_V10_{SLUG}.fb2")
REPORT = Path(f"v10-{SLUG}-report.json")
MAP = Path(f"v10-{SLUG}-map.json")
SOURCE_TXT = Path(f"v10-{SLUG}-source.txt")
TRANSLATED_TXT = Path(f"v10-{SLUG}-translated.txt")
BIBLE = Path(os.getenv("BOOKAI_V10_BIBLE_CACHE") or ".bookai-cache-v10/book-bible.json")


def _norm(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _select_chapter(document):
    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    wanted = _norm(CHAPTER_NAME)
    exact = [(name, rows) for name, rows in groups if _norm(name) == wanted]
    if not exact:
        exact = [(name, rows) for name, rows in groups if wanted in _norm(name)]
    if len(exact) != 1:
        raise RuntimeError(f"Expected one chapter matching {CHAPTER_NAME!r}; found {len(exact)}; available={[name for name, _ in groups]}")
    return all_targets, exact[0][0], exact[0][1]


def _usage_delta(after: dict, before: dict) -> dict:
    return {key: int(after.get(key) or 0) - int(before.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls")}


def _provider() -> OpenAICompatibleProvider:
    key = os.getenv("BOOKAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base = os.getenv("BOOKAI_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("BOOKAI_GATE_MODEL") or os.getenv("BOOKAI_MODEL") or "deepseek-flash"
    return OpenAICompatibleProvider(key, base, model, os.getenv("BOOKAI_REASONING") or "none", role="v10_semantic")


def main() -> None:
    run_started = time.perf_counter()
    document = load_book(SOURCE)
    all_targets, chapter_name, targets = _select_chapter(document)
    print("[v10] " + json.dumps({
        "chapter": chapter_name,
        "segments": len(targets),
        "source_chars": sum(len(s.text) for s in targets),
        "whole_book_segments": len(all_targets),
        "architecture": "clean-v10:source-only+integrity+v9d-dialogue-speaker+v9ad-canon-risk+quantity-v2:no-v9-imports",
    }, ensure_ascii=False), flush=True)

    giga = RobustTaggedPrimaryTransport()
    if not giga.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing")

    bible_started = time.perf_counter()
    memory, bible_stats = V9ADSourceOnlyBookBibleBuilder(giga, BIBLE).build(all_targets)
    bible_seconds = time.perf_counter() - bible_started
    usage_after_bible = giga.usage.as_dict()

    chapter_started = time.perf_counter()
    translated, primary_errors = giga.translate_many(targets, memory, source_segments=all_targets)
    usage_after_primary = giga.usage.as_dict()

    # Structural fidelity is a prerequisite, not a late QA concern. A segment that
    # contains prompt residue or the wrong neighboring paragraph is isolated and
    # retranslated by Giga before dialogue normalization or semantic routing.
    integrity = SegmentIntegrityGate(giga)
    integrity_changed = integrity.repair(targets, translated, memory, source_segments=all_targets)
    usage_after_integrity = giga.usage.as_dict()

    dialogue_guard = DialogueDiscourseGuard()
    dialogue_after_primary = dialogue_guard.apply(targets, translated)
    speaker_guard = DialogueSpeakerContinuityGuard()
    speaker_after_primary = speaker_guard.apply(targets, translated)

    qa = V10QualityQA()
    initial_issues = qa.scan(targets, translated, memory)
    patcher = GigaSpanPatcher(giga, qa)
    patch_changed = patcher.repair(targets, translated, memory, initial_issues)
    usage_after_patcher = giga.usage.as_dict()
    post_patch_issues = qa.scan(targets, translated, memory)

    local_rewriter = GigaLocalRewriter(giga, qa, max_segments=16)
    local_changed = local_rewriter.repair(targets, translated, memory, post_patch_issues)
    usage_after_local = giga.usage.as_dict()
    post_local_issues = qa.scan(targets, translated, memory)

    provider = _provider()
    specialist = DeepSeekSemanticSpecialist(
        provider,
        qa,
        max_segments=max(4, int(os.getenv("BOOKAI_V10_DEEP_MAX") or "8")),
    )
    specialist_issues = [issue for issue in post_local_issues if issue.code != "missing"]
    deep_changed = specialist.repair(targets, translated, memory, specialist_issues)

    dialogue_after_semantic = dialogue_guard.apply(targets, translated)
    speaker_guard_final = DialogueSpeakerContinuityGuard()
    speaker_after_semantic = speaker_guard_final.apply(targets, translated)
    final_issues = qa.scan(targets, translated, memory)
    final_integrity = integrity.scan(targets, translated)
    chapter_seconds = time.perf_counter() - chapter_started

    missing = [segment.id for segment in targets if not str(translated.get(segment.id) or "").strip()]
    save_book(document, translated, OUTPUT)
    SOURCE_TXT.write_text("\n\n".join(s.text for s in targets), "utf-8")
    TRANSLATED_TXT.write_text("\n\n".join(translated.get(s.id, f"[UNTRANSLATED {s.id}]") for s in targets), "utf-8")

    issue_by_id: dict[str, list[dict]] = {}
    for issue in final_issues:
        issue_by_id.setdefault(issue.id, []).append({
            "code": issue.code, "mode": issue.mode, "severity": issue.severity, "reason": issue.reason
        })
    integrity_by_id: dict[str, list[dict]] = {}
    for issue in final_integrity:
        integrity_by_id.setdefault(issue.id, []).append({"code": issue.code, "reason": issue.reason})
    mapping = [
        {
            "id": s.id,
            "chapter": s.chapter,
            "source": s.text,
            "translation": translated.get(s.id),
            "length_ratio": round(len(str(translated.get(s.id) or "")) / max(1, len(s.text)), 3),
            "final_issues": issue_by_id.get(s.id, []),
            "final_integrity": integrity_by_id.get(s.id, []),
        }
        for s in targets
    ]
    MAP.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), "utf-8")

    report = {
        "version": "v10-clean-6-integrity-quantity-v2",
        "chapter": chapter_name,
        "segments": len(targets),
        "source_chars": sum(len(s.text) for s in targets),
        "completed": len(targets) - len(missing),
        "missing_ids": missing,
        "timing": {
            "book_bible_one_time_seconds": round(bible_seconds, 2),
            "chapter_seconds": round(chapter_seconds, 2),
            "chapter_under_180_seconds": chapter_seconds <= 180,
            "total_cold_seconds": round(time.perf_counter() - run_started, 2),
        },
        "architecture": {
            "book_bible": "SOURCE-ONLY high-coverage v9ad-style spelling canon + conservative technical glossary; no reference seed",
            "primary": "GigaChat tagged batches + bounded Giga-only recovery + prompt-leak rejection",
            "segment_integrity": "pre-QA extreme ratio/protocol gate + isolated one-segment Giga recovery",
            "discourse_dialogue": "v9d source-structural quotation normalization + conservative two-speaker continuity",
            "qa": "deterministic fidelity + proposition-aware QuantityFidelity v2 + direction/kinship/hunting contracts",
            "cheap_repair": "Giga exact-span patch first, then bounded full-segment Giga rewrite only for proven local defects",
            "semantic_repair": "one DeepSeek batch, <=8 v9ad-style risk-ranked semantic segments",
            "final_gate": "deterministic QA + structural integrity scan",
            "reference_seed": False,
            "legacy_sanitizer": False,
            "deepseek_verifier": False,
            "v9_monkey_patch_chain": False,
        },
        "book_bible": bible_stats,
        "segment_integrity": {**dict(integrity.stats), "changed_ids": integrity_changed},
        "final_integrity": {
            "count": len(final_integrity),
            "ids": sorted({issue.id for issue in final_integrity}),
            "issues": [{"id": issue.id, "code": issue.code, "reason": issue.reason} for issue in final_integrity],
        },
        "dialogue_guard": {
            **dict(dialogue_guard.stats),
            "after_primary_changed_ids": dialogue_after_primary,
            "after_semantic_changed_ids": dialogue_after_semantic,
        },
        "speaker_guard": {
            "after_primary": {**dict(speaker_guard.stats), "changed_ids": speaker_after_primary},
            "after_semantic": {**dict(speaker_guard_final.stats), "changed_ids": speaker_after_semantic},
        },
        "primary_transport": dict(giga.transport_stats),
        "usage": {
            "gigachat_bible": usage_after_bible,
            "gigachat_primary": _usage_delta(usage_after_primary, usage_after_bible),
            "gigachat_integrity": _usage_delta(usage_after_integrity, usage_after_primary),
            "gigachat_patcher": _usage_delta(usage_after_patcher, usage_after_integrity),
            "gigachat_local_rewriter": _usage_delta(usage_after_local, usage_after_patcher),
            "gigachat_total": usage_after_local,
            "deepseek": dict(provider.usage),
        },
        "primary_errors": primary_errors,
        "qa_initial": issue_summary(initial_issues),
        "patcher": {**patcher.stats, "changed_ids": patch_changed},
        "qa_after_patch": issue_summary(post_patch_issues),
        "local_rewriter": {**local_rewriter.stats, "changed_ids": local_changed},
        "qa_after_local": issue_summary(post_local_issues),
        "deepseek_specialist": {**specialist.stats, "changed_ids": deep_changed},
        "qa_final": issue_summary(final_issues),
        "output": str(OUTPUT),
        "map": str(MAP),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("[v10-done] " + json.dumps(report, ensure_ascii=False), flush=True)

    if missing:
        raise RuntimeError(f"v10 left {len(missing)} untranslated segments after Giga-only recovery: {missing[:12]}")
    if final_integrity:
        raise RuntimeError(f"v10 structural integrity gate left {len(final_integrity)} suspicious segments: {sorted({i.id for i in final_integrity})[:12]}")


if __name__ == "__main__":
    main()
