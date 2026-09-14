from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from bookai.llm import OpenAICompatibleProvider
from bookai.parsers.base import load_book, save_book
from bookai.release_final import (
    FinalBookBibleBuilder,
    FinalDeepSeekSemanticSpecialist,
    FinalDialogueDiscourseGuard,
    FinalV10QualityQA,
    ResidualHardDeepSeekRepair,
)
from bookai.v10 import issue_summary
from bookai.v10_integrity import SegmentIntegrityGate
from bookai.v10_release import (
    HardenedLocalRewriter,
    HardenedRobustTaggedPrimaryTransport,
    select_numbered_chapter,
)
from bookai.v10_speaker import DialogueSpeakerContinuityGuard


SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
CHAPTER_NAME = os.getenv("BOOKAI_CHAPTER_NAME") or "Chapter One"
SLUG = re.sub(r"[^a-z0-9]+", "-", CHAPTER_NAME.casefold()).strip("-") or "chapter"
OUTPUT = Path(f"Devices_and_Desires_RU_V10_{SLUG}.fb2")
REPORT = Path(f"v10-{SLUG}-report.json")
MAP = Path(f"v10-{SLUG}-map.json")
SOURCE_TXT = Path(f"v10-{SLUG}-source.txt")
TRANSLATED_TXT = Path(f"v10-{SLUG}-translated.txt")
BIBLE = Path(os.getenv("BOOKAI_V10_BIBLE_CACHE") or ".bookai-cache-v10/book-bible.json")


def _usage_delta(after: dict, before: dict) -> dict:
    return {
        key: int(after.get(key) or 0) - int(before.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls")
    }


def _provider() -> OpenAICompatibleProvider:
    key = os.getenv("BOOKAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base = os.getenv("BOOKAI_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("BOOKAI_GATE_MODEL") or os.getenv("BOOKAI_MODEL") or "deepseek-flash"
    return OpenAICompatibleProvider(key, base, model, os.getenv("BOOKAI_REASONING") or "none", role="v10_semantic")


def _hard_issues(issues) -> list:
    return [issue for issue in issues if issue.severity == "hard"]


def main() -> None:
    run_started = time.perf_counter()
    document = load_book(SOURCE)
    all_targets, chapter_name, targets, selection = select_numbered_chapter(document, CHAPTER_NAME)
    print("[v10-release9] " + json.dumps({
        "chapter": chapter_name,
        "segments": len(targets),
        "source_chars": sum(len(s.text) for s in targets),
        "whole_book_segments": len(all_targets),
        "selection": selection,
        "architecture": "clean9.1:complete-chapter+safe-dialogue+evidence-aware-qa+focused-release-deepseek+fail-closed-residual-rescue",
    }, ensure_ascii=False), flush=True)

    giga = HardenedRobustTaggedPrimaryTransport()
    if not giga.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing")

    bible_started = time.perf_counter()
    memory, bible_stats = FinalBookBibleBuilder(giga, BIBLE).build(all_targets)
    bible_seconds = time.perf_counter() - bible_started
    usage_after_bible = giga.usage.as_dict()

    chapter_started = time.perf_counter()
    translated, primary_errors = giga.translate_many(targets, memory, source_segments=all_targets)
    usage_after_primary = giga.usage.as_dict()

    integrity = SegmentIntegrityGate(giga)
    integrity_changed = integrity.repair(targets, translated, memory, source_segments=all_targets)
    usage_after_integrity = giga.usage.as_dict()

    dialogue_guard = FinalDialogueDiscourseGuard()
    dialogue_after_primary = dialogue_guard.apply(targets, translated)
    speaker_primary = DialogueSpeakerContinuityGuard()
    speaker_after_primary = speaker_primary.apply(targets, translated)

    qa = FinalV10QualityQA(demote_clause_order=False)
    initial_issues = qa.scan(targets, translated, memory)

    local_rewriter = HardenedLocalRewriter(giga, qa, max_segments=28)
    local_changed = local_rewriter.repair(targets, translated, memory, initial_issues)
    usage_after_local = giga.usage.as_dict()
    dialogue_after_local = dialogue_guard.apply(targets, translated)
    speaker_local = DialogueSpeakerContinuityGuard()
    speaker_after_local = speaker_local.apply(targets, translated)
    post_local_issues = qa.scan(targets, translated, memory)

    provider = _provider()
    specialist = FinalDeepSeekSemanticSpecialist(
        provider,
        qa,
        max_segments=max(4, int(os.getenv("BOOKAI_V10_DEEP_MAX") or "8")),
    )
    specialist_issues = [issue for issue in post_local_issues if issue.code != "missing"]
    deep_changed = specialist.repair(targets, translated, memory, specialist_issues)

    dialogue_after_semantic = dialogue_guard.apply(targets, translated)
    speaker_semantic = DialogueSpeakerContinuityGuard()
    speaker_after_semantic = speaker_semantic.apply(targets, translated)

    final_integrity_guard = SegmentIntegrityGate(giga)
    final_integrity_changed = final_integrity_guard.repair(targets, translated, memory, source_segments=all_targets)
    usage_after_final_integrity = giga.usage.as_dict()
    dialogue_after_integrity = dialogue_guard.apply(targets, translated)
    speaker_integrity = DialogueSpeakerContinuityGuard()
    speaker_after_integrity = speaker_integrity.apply(targets, translated)

    # Release tail: no second Giga rewrite cascade. Evidence-aware QA first removes
    # proven false positives; one larger DeepSeek batch receives residual HARD rows.
    release_qa = FinalV10QualityQA(demote_clause_order=True)
    release_pre_issues = release_qa.scan(targets, translated, memory)
    release_local_changed: list[str] = []
    usage_after_release_local = usage_after_final_integrity
    release_mid_issues = release_pre_issues

    release_specialist = FinalDeepSeekSemanticSpecialist(
        provider,
        release_qa,
        max_segments=max(8, int(os.getenv("BOOKAI_V10_RELEASE_DEEP_MAX") or "16")),
    )
    release_deep_issues = [issue for issue in release_mid_issues if issue.code != "missing"]
    release_deep_changed = release_specialist.repair(targets, translated, memory, release_deep_issues)
    dialogue_after_release_semantic = dialogue_guard.apply(targets, translated)
    speaker_release_semantic = DialogueSpeakerContinuityGuard()
    speaker_after_release_semantic = speaker_release_semantic.apply(targets, translated)

    # Rare provider/schema-confidence collapse is rescued by one small, proof-gated
    # call over ONLY the rows that still fail deterministic release QA.
    residual_pre_issues = release_qa.scan(targets, translated, memory)
    residual_repair = ResidualHardDeepSeekRepair(
        provider,
        release_qa,
        max_segments=max(4, int(os.getenv("BOOKAI_V10_RESIDUAL_DEEP_MAX") or "12")),
    )
    residual_changed = residual_repair.repair(targets, translated, memory, residual_pre_issues)
    dialogue_after_residual = dialogue_guard.apply(targets, translated)
    speaker_residual = DialogueSpeakerContinuityGuard()
    speaker_after_residual = speaker_residual.apply(targets, translated)
    residual_post_issues = release_qa.scan(targets, translated, memory)

    publication_integrity = SegmentIntegrityGate(giga)
    publication_integrity_changed = publication_integrity.repair(
        targets, translated, memory, source_segments=all_targets
    )
    usage_after_publication_integrity = giga.usage.as_dict()
    dialogue_after_publication_integrity = dialogue_guard.apply(targets, translated)
    speaker_publication_integrity = DialogueSpeakerContinuityGuard()
    speaker_after_publication_integrity = speaker_publication_integrity.apply(targets, translated)

    final_issues = release_qa.scan(targets, translated, memory)
    final_integrity = publication_integrity.scan(targets, translated)
    final_hard = _hard_issues(final_issues)
    chapter_seconds = time.perf_counter() - chapter_started

    missing = [segment.id for segment in targets if not str(translated.get(segment.id) or "").strip()]
    save_book(document, translated, OUTPUT)
    SOURCE_TXT.write_text("\n\n".join(s.text for s in targets), "utf-8")
    TRANSLATED_TXT.write_text(
        "\n\n".join(translated.get(s.id, f"[UNTRANSLATED {s.id}]") for s in targets),
        "utf-8",
    )

    issue_by_id: dict[str, list[dict]] = {}
    for issue in final_issues:
        issue_by_id.setdefault(issue.id, []).append({
            "code": issue.code,
            "mode": issue.mode,
            "severity": issue.severity,
            "reason": issue.reason,
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
        "version": "v10-clean-9.1-focused-release",
        "chapter": chapter_name,
        "chapter_selection": selection,
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
            "chapter_selection": "numbered Chapter N boundary; internal FB2 section headings remain inside chapter",
            "book_bible": "SOURCE-ONLY canon with runtime pruning of false common-word character entries",
            "primary": "GigaChat tagged batches + bounded recovery",
            "dialogue": "source-aware dialogue dash normalization preserving genuine nested Russian guillemets",
            "qa": "evidence-aware quantity/fraction/half-inch/name/mixed-notation filtering; hard semantic contracts retained",
            "release_tail": "NO second Giga cascade; one bounded DeepSeek batch plus one small fail-closed rescue only if HARD remains",
            "final_gate": "zero untranslated + zero structural integrity + zero HARD release QA issues",
            "reference_seed": False,
        },
        "book_bible": bible_stats,
        "segment_integrity": {**dict(integrity.stats), "changed_ids": integrity_changed},
        "final_integrity_repair": {**dict(final_integrity_guard.stats), "changed_ids": final_integrity_changed},
        "publication_integrity_repair": {**dict(publication_integrity.stats), "changed_ids": publication_integrity_changed},
        "final_integrity": {
            "count": len(final_integrity),
            "ids": sorted({issue.id for issue in final_integrity}),
            "issues": [{"id": issue.id, "code": issue.code, "reason": issue.reason} for issue in final_integrity],
        },
        "dialogue_guard": {
            **dict(dialogue_guard.stats),
            "after_primary_changed_ids": dialogue_after_primary,
            "after_local_changed_ids": dialogue_after_local,
            "after_semantic_changed_ids": dialogue_after_semantic,
            "after_integrity_changed_ids": dialogue_after_integrity,
            "after_release_local_changed_ids": release_local_changed,
            "after_release_semantic_changed_ids": dialogue_after_release_semantic,
            "after_residual_changed_ids": dialogue_after_residual,
            "after_publication_integrity_changed_ids": dialogue_after_publication_integrity,
        },
        "speaker_guard": {
            "after_primary": {**dict(speaker_primary.stats), "changed_ids": speaker_after_primary},
            "after_local": {**dict(speaker_local.stats), "changed_ids": speaker_after_local},
            "after_semantic": {**dict(speaker_semantic.stats), "changed_ids": speaker_after_semantic},
            "after_integrity": {**dict(speaker_integrity.stats), "changed_ids": speaker_after_integrity},
            "after_release_semantic": {**dict(speaker_release_semantic.stats), "changed_ids": speaker_after_release_semantic},
            "after_residual": {**dict(speaker_residual.stats), "changed_ids": speaker_after_residual},
            "after_publication_integrity": {**dict(speaker_publication_integrity.stats), "changed_ids": speaker_after_publication_integrity},
        },
        "primary_transport": dict(giga.transport_stats),
        "usage": {
            "gigachat_bible": usage_after_bible,
            "gigachat_primary": _usage_delta(usage_after_primary, usage_after_bible),
            "gigachat_integrity": _usage_delta(usage_after_integrity, usage_after_primary),
            "gigachat_local_rewriter": _usage_delta(usage_after_local, usage_after_integrity),
            "gigachat_final_integrity": _usage_delta(usage_after_final_integrity, usage_after_local),
            "gigachat_release_local_rewriter": _usage_delta(usage_after_release_local, usage_after_final_integrity),
            "gigachat_publication_integrity": _usage_delta(usage_after_publication_integrity, usage_after_release_local),
            "gigachat_total": usage_after_publication_integrity,
            "deepseek": dict(provider.usage),
        },
        "primary_errors": primary_errors,
        "qa_initial": issue_summary(initial_issues),
        "local_rewriter": {**local_rewriter.stats, "changed_ids": local_changed},
        "qa_after_local": issue_summary(post_local_issues),
        "deepseek_specialist": {**specialist.stats, "changed_ids": deep_changed},
        "qa_release_pre": issue_summary(release_pre_issues),
        "release_local_rewriter": {"disabled": True, "reason": "replaced by focused semantic release tail", "changed_ids": []},
        "qa_release_mid": issue_summary(release_mid_issues),
        "release_deepseek_specialist": {**release_specialist.stats, "changed_ids": release_deep_changed},
        "qa_residual_pre": issue_summary(residual_pre_issues),
        "residual_deepseek_repair": dict(residual_repair.stats),
        "qa_residual_post": issue_summary(residual_post_issues),
        "qa_final": issue_summary(final_issues),
        "final_hard": {
            "count": len(final_hard),
            "ids": sorted({issue.id for issue in final_hard}),
            "issues": [
                {"id": issue.id, "code": issue.code, "mode": issue.mode, "reason": issue.reason}
                for issue in final_hard
            ],
        },
        "output": str(OUTPUT),
        "map": str(MAP),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("[v10-release9-done] " + json.dumps(report, ensure_ascii=False), flush=True)

    if missing:
        raise RuntimeError(f"v10 left {len(missing)} untranslated segments after bounded recovery: {missing[:12]}")
    if final_integrity:
        raise RuntimeError(
            f"v10 publication integrity gate left {len(final_integrity)} suspicious segments: "
            f"{sorted({i.id for i in final_integrity})[:12]}"
        )
    if final_hard:
        preview = [(issue.id, issue.code) for issue in final_hard[:12]]
        raise RuntimeError(f"v10 publication QA left {len(final_hard)} HARD issues: {preview}")


if __name__ == "__main__":
    main()
