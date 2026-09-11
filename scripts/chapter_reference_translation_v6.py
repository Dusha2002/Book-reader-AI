from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v4b as v4b
import chapter_reference_translation_v5 as v5
import hybrid_reference_translation as hybrid
from bookai.audit import safe_analyze_memory
from bookai.contracts import issue_rows
from bookai.gigachat_v3 import GigaChatLightningV3Backend
from bookai.harness import TranslationHarness
from bookai.literary_context import locked_glossary_violations
from bookai.llm import extract_json
from bookai.models import BookMemory, GateFinding, Segment
from bookai.pipeline import _chapter_groups, _should_translate
from bookai.quality_v3 import QualityIssueV3, enhanced_batch_issues, enhanced_candidate_issues
from bookai.resilience import resilient_findings


# quality-v6: production-realistic literary translation from ONE uploaded source book.
# No Russian reference, reference profile, seeded reference glossary or gold TM is
# available to any model call. The external reference PDF is evaluation-only.
#
# Source book -> one source-only Book Intelligence pass -> GigaChat Lightning draft
# -> deterministic integrity + one unified bilingual QE -> one targeted edit pass
# -> tiny hard-tail retry/candidate path -> deterministic typography + self-built TM.

_LARGEST_ALIASES = {"largest", "largest-chapter", "__largest__"}
_SOURCE_SEGMENTS: list[Segment] = []
_SOURCE_MEMORY: BookMemory | None = None
_SOURCE_READY = threading.Event()
_V6_BOOK_STATS: dict = {}
_V6_QE_STATS: dict = {}
_V6_REMAINING_HARD: set[str] = set()
_V6_REMAINING_MEDIUM: set[str] = set()
_V6_REASONS: dict[str, str] = {}
_ORIGINAL_ENHANCED_BATCH_ISSUES = enhanced_batch_issues

_COMMON_CAPS = {
    "The", "This", "That", "Then", "There", "When", "While", "With", "Without", "What", "Why",
    "How", "He", "She", "His", "Her", "They", "Their", "It", "Its", "I", "We", "You", "But",
    "And", "Or", "If", "As", "At", "By", "For", "From", "In", "Into", "No", "Not", "Of", "On",
    "So", "To", "Was", "Were", "A", "An", "Chapter",
}
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
_DIALOGUE_ONLY = re.compile(r"^\s*([\"'“‘«])(.+?)([\"'”’»])\s*([.!?…]?)\s*$", re.S)


def _source_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TERM_RE.findall(text or "")
        if len(token) >= 4 and token not in _COMMON_CAPS
    }


def _entity_census(segments: list[Segment], limit: int = 90) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for segment in segments:
        words = _TERM_RE.findall(segment.text)
        for word in words:
            if word in _COMMON_CAPS or not word[:1].isupper():
                continue
            counts[word] += 1
        # Capture recurring title/name bigrams without trying to understand them.
        for a, b in zip(words, words[1:]):
            if a[:1].isupper() and b[:1].isupper() and a not in _COMMON_CAPS and b not in _COMMON_CAPS:
                counts[f"{a} {b}"] += 1
    return [(name, count) for name, count in counts.most_common(limit) if count >= 2]


def _balanced_source_sample(segments: list[Segment], budget: int = 28500) -> str:
    translatable = [segment for segment in segments if _should_translate(segment.text)]
    groups = _chapter_groups(translatable)
    census = _entity_census(translatable)
    census_text = "\n".join(f"{name} | occurrences={count}" for name, count in census)
    prefix = (
        "### RECURRING SOURCE ENTITY CENSUS\n"
        "These are source-only observations, not translations. Infer a stable Russian rendering only when high-confidence.\n"
        + census_text
        + "\n"
    )
    remaining = max(8000, budget - len(prefix))
    if not groups:
        return (prefix + "\n" + "\n".join(segment.text for segment in translatable))[:budget]
    per = max(650, remaining // len(groups))
    rows = [prefix]
    for name, chapter in groups:
        text = "\n".join(segment.text for segment in chapter)
        if len(text) <= per:
            excerpt = text
        else:
            third = max(180, per // 3)
            mid = max(0, len(text) // 2 - third // 2)
            excerpt = text[:third] + "\n...[middle]...\n" + text[mid : mid + third] + "\n...[end]...\n" + text[-third:]
        rows.append(f"### {name}\n{excerpt[:per]}")
    return "\n".join(rows)[:budget]


def _parse_source_memory(provider, sample: str) -> BookMemory:
    system = """Build a SOURCE-ONLY BOOK INTELLIGENCE bible for EN→RU literary translation.
You have only the English book. No published/reference translation exists. Never claim otherwise.
Infer reusable evidence from the source: authorial voice, rhythm, irony, dialogue behavior, recurring proper
names/places/institutions/technical terms, character identity/voice cues, relationships/register, and continuity.
Choose stable Russian renderings only for HIGH-CONFIDENCE recurring named entities or genuinely recurring
technical/world terms. Do NOT lock ordinary vocabulary or idioms; those require context.
Return ONLY compact JSON and do not translate passages.
Schema:
{
 "style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"...","taboos":["..."]},
 "glossary":{"Recurring English name/term":"preferred Russian rendering"},
 "characters":{"English name":"ru=<name>;gender=male|female|unknown;voice=<brief>;role=<brief>;register=<brief>"},
 "rolling_summary":"compact book-wide continuity/relationship/world summary in Russian"
}
Keep glossary <=90 entries and characters <=45. Prefer omission over guessing."""
    user = "SOURCE_BOOK_EVIDENCE:\n" + sample
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.0))
        if not isinstance(obj, dict):
            raise ValueError("book intelligence returned non-object JSON")
        style_obj = obj.get("style") or {}
        fallback = BookMemory()
        style = fallback.style
        style.narrative_voice = str(style_obj.get("narrative_voice") or style.narrative_voice)
        style.rhythm = str(style_obj.get("rhythm") or style.rhythm)
        style.dialogue = str(style_obj.get("dialogue") or style.dialogue)
        style.humor = str(style_obj.get("humor") or style.humor)
        style.taboos = [str(x) for x in (style_obj.get("taboos") or style.taboos)][:12]
        glossary = {
            str(k).strip(): str(v).strip()
            for k, v in dict(obj.get("glossary") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        characters = {
            str(k).strip(): str(v).strip()
            for k, v in dict(obj.get("characters") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        return BookMemory(
            style=style,
            glossary=dict(list(glossary.items())[:90]),
            characters=dict(list(characters.items())[:45]),
            rolling_summary=" ".join(str(obj.get("rolling_summary") or "").split())[:7000],
        )
    except Exception as exc:
        print(f"[v6-book-intelligence] structured_pass_failed error={type(exc).__name__}; fallback=safe_analyze_memory", flush=True)
        return safe_analyze_memory(provider, sample, attempts=2)


def _select_chapter(document):
    global _SOURCE_SEGMENTS
    _SOURCE_SEGMENTS = [segment for segment in document.segments if _should_translate(segment.text)]
    return v5._select_chapter(document)


def _source_analysis_task(harness, chapters, state_snapshot: dict):
    global _SOURCE_MEMORY, _V6_BOOK_STATS
    started = time.perf_counter()
    sample = _balanced_source_sample(
        _SOURCE_SEGMENTS,
        budget=max(18000, min(32000, int(os.getenv("BOOKAI_ANALYSIS_CHARS") or "28500"))),
    )
    memory = _parse_source_memory(harness.analyzer, sample)
    _SOURCE_MEMORY = memory
    _SOURCE_READY.set()

    # A chapter benchmark still receives the whole-book source bible. No target
    # language reference, seed glossary or previous gold translation is involved.
    compact = " ".join(str(memory.rolling_summary or "").split())[:6000]
    chapter_digests = {name: compact for name, _ in chapters if compact}
    book_synopsis = compact

    cached_tm = list(state_snapshot.get("v6_translation_memory") or [])
    GigaChatLightningV6Backend.set_translation_memory(cached_tm)

    elapsed = time.perf_counter() - started
    _V6_BOOK_STATS = {
        "source_only": True,
        "source_segments_seen": len(_SOURCE_SEGMENTS),
        "sample_chars": len(sample),
        "glossary_entries": len(memory.glossary),
        "characters": len(memory.characters),
        "cached_tm_entries": len(cached_tm),
        "elapsed_seconds": round(elapsed, 2),
    }
    state_snapshot["v6_book_intelligence"] = dict(_V6_BOOK_STATS)
    state_snapshot["chapter_briefs"] = chapter_digests
    state_snapshot["book_synopsis"] = book_synopsis
    state_snapshot["context_strategy"] = {
        "book_intelligence": "whole-source-book-only",
        "reference_translation": False,
        "seed_reference_glossary": False,
        "retrieval": "local+distant-source+approved-self-TM",
        "qe": "single-unified-bilingual-publication-gate",
    }
    hybrid.progress({"phase": "v6_book_intelligence_ready", **_V6_BOOK_STATS})
    return memory, chapter_digests, book_synopsis, state_snapshot, elapsed


class GigaChatLightningV6Backend(GigaChatLightningV3Backend):
    name = "gigachat-3-lightning-v6-source-only"
    _tm_entries: list[dict] = []

    @classmethod
    def set_translation_memory(cls, entries: list[dict]) -> None:
        clean: list[dict] = []
        for row in entries[-5000:]:
            if not isinstance(row, dict):
                continue
            source = str(row.get("source") or "").strip()
            translation = str(row.get("translation") or "").strip()
            if source and translation:
                clean.append({
                    "source": source,
                    "translation": translation,
                    "chapter": str(row.get("chapter") or ""),
                    "quality": float(row.get("quality") or 1.0),
                })
        cls._tm_entries = clean

    @classmethod
    def _retrieve_tm(cls, batch: list[Segment], k: int = 4) -> list[dict]:
        if not cls._tm_entries or not batch:
            return []
        query = set().union(*(_source_terms(segment.text) for segment in batch))
        if not query:
            return []
        chapter = batch[0].chapter
        scored: list[tuple[float, dict]] = []
        for row in cls._tm_entries:
            terms = _source_terms(str(row.get("source") or ""))
            if not terms:
                continue
            overlap = query & terms
            if not overlap:
                continue
            score = 2.0 * len(overlap) / max(1, len(query) + len(terms))
            score *= max(0.5, min(1.0, float(row.get("quality") or 1.0)))
            if chapter and row.get("chapter") == chapter:
                score *= 1.08
            if score >= 0.08:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: max(0, min(5, k))]]

    def _prompt(self, batch, memory, *, source_segments=None, minimal=False):
        base = super()._prompt(batch, memory, source_segments=source_segments, minimal=minimal)
        examples = self._retrieve_tm(batch, k=int(os.getenv("BOOKAI_TM_RETRIEVAL_K") or "4"))
        if not examples:
            return base
        payload = [
            {"source": row["source"][:1200], "approved_ru": row["translation"][:1400]}
            for row in examples
        ]
        return (
            base
            + "\n\nAPPROVED_SELF_TRANSLATION_MEMORY (READ ONLY; examples from this same book, never copy blindly):\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\nUse these examples only for established naming, register and recurring terminology. Source TARGETS remain authoritative."
        )


def _translate_gigachat_v6(harness, bulk, source_segments, batch, memory, stats):
    # v6 is a true preflight architecture: the first real batch waits briefly for
    # whole-book source analysis, rather than translating early pages with no bible.
    wait_seconds = max(0, min(45, int(os.getenv("BOOKAI_BOOK_INTELLIGENCE_WAIT") or "30")))
    if _SOURCE_MEMORY is None and wait_seconds:
        _SOURCE_READY.wait(timeout=wait_seconds)
    effective_memory = _SOURCE_MEMORY or memory
    return v5._translate_gigachat_only_batch(
        harness, bulk, source_segments, batch, effective_memory, stats
    )


def _source_locked_glossary(memory: BookMemory) -> dict[str, str]:
    locked: dict[str, str] = {}
    character_names = {str(name).casefold() for name in memory.characters}
    for source, target in memory.glossary.items():
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            continue
        words = _TERM_RE.findall(source)
        named = any(word[:1].isupper() for word in words)
        if named or source.casefold() in character_names:
            locked[source] = target
    for source, desc in memory.characters.items():
        match = re.search(r"\bru=([^;]+)", str(desc), flags=re.I)
        if match:
            locked[str(source)] = match.group(1).strip()
    return locked


def _source_only_hard(targets, translations, memory) -> tuple[set[str], dict[str, str]]:
    hard: set[str] = set()
    reasons: dict[str, list[str]] = {}
    locked = _source_locked_glossary(memory)
    for segment in targets:
        candidate = str(translations.get(segment.id) or "")
        for issue in enhanced_candidate_issues(
            segment,
            candidate,
            memory,
            source_segments=targets,
        ):
            if issue.severity == "hard":
                hard.add(segment.id)
                reasons.setdefault(segment.id, []).append(f"{issue.code}: {issue.reason}")
        if locked and locked_glossary_violations([segment], {segment.id: candidate}, locked):
            hard.add(segment.id)
            reasons.setdefault(segment.id, []).append("source-derived named entity inconsistency")
    return hard, {sid: "; ".join(dict.fromkeys(rows)) for sid, rows in reasons.items()}


def _unified_gate_batch(provider, originals, draft, memory) -> list[GateFinding]:
    if not originals:
        return []
    system = """You are a conservative bilingual EN→RU PUBLICATION QUALITY auditor for literary fiction.
Do NOT rewrite. Report only defects that a professional editor should actually change, not preferences.
In ONE pass check both fidelity and Russian literary quality:
- omissions/additions and actor→action→object/pronoun relations;
- numbers, comparisons, negation, modality, chronology, cause/effect and enumerations;
- named entities and technical referents against the SOURCE-DERIVED book bible;
- idioms, jokes, metaphors and dry irony: preserve function/effect, not English wording;
- clear English calques, broken Russian grammar/government/collocation;
- character/narrator register and obvious dialogue naturalness problems.
Do not flag harmless paraphrase, normal Russian restructuring, or merely different but equally good wording.
Severity hard = meaning/coverage/entity/number/referent corruption or unusable Russian.
Severity medium = clear publishable-prose defect worth editing, but meaning remains intact.
Return ONLY JSON {"issues":[{"id":"exact id","severity":"hard|medium","code":"omission|addition|relation|number|entity|term|idiom|calque|grammar|voice|irony|dialogue|other","reason":"specific concise defect"}]}.
Use [] for no issues. Use only supplied ids."""
    pairs = {segment.id: {"en": segment.text, "ru": draft.get(segment.id, "")} for segment in originals}
    compact_memory = {
        "style": asdict(memory.style),
        "glossary": memory.glossary,
        "characters": memory.characters,
        "continuity": str(memory.rolling_summary or "")[:4500],
    }
    user = (
        "SOURCE_DERIVED_BOOK_BIBLE:" + json.dumps(compact_memory, ensure_ascii=False, separators=(",", ":"))
        + "\nPAIRS:" + json.dumps(pairs, ensure_ascii=False)
    )
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    valid_ids = {segment.id for segment in originals}
    findings: list[GateFinding] = []
    for raw in issue_rows(obj, "v6 unified gate"):
        sid = str(raw.get("id") or "").strip()
        if sid not in valid_ids:
            if len(valid_ids) == 1:
                sid = next(iter(valid_ids))
            else:
                raise ValueError(f"v6 unified gate returned unknown id {sid!r}")
        severity = str(raw.get("severity") or "medium").strip().lower()
        if severity not in {"hard", "medium"}:
            severity = "medium"
        code = str(raw.get("code") or "other").strip().lower()
        reason = " ".join(str(raw.get("reason") or "").split())
        if not reason:
            continue
        findings.append(GateFinding(sid, severity, f"{code}: {reason}"))
    return findings


def _qe_batches(targets, max_chars: int) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    chars = 0
    for segment in targets:
        if current and chars + len(segment.text) > max_chars:
            batches.append(current)
            current, chars = [], 0
        current.append(segment)
        chars += len(segment.text)
    if current:
        batches.append(current)
    return batches


def _parallel_unified_qe(harness, batches, translations, memory, *, workers: int):
    findings: list[GateFinding] = []
    if not batches:
        return findings

    def run(batch):
        return resilient_findings(
            list(batch),
            lambda part: _unified_gate_batch(harness.gate, part, translations, memory),
            label="v6_unified_qe",
            attempts=2,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bookai-v6-qe") as pool:
        futures = [pool.submit(run, batch) for batch in batches]
        for future in as_completed(futures):
            findings.extend(future.result())
    return findings


def _reason_map(findings) -> tuple[dict[str, str], dict[str, str]]:
    reasons: dict[str, list[str]] = {}
    severity: dict[str, str] = {}
    for finding in findings:
        reasons.setdefault(finding.id, []).append(finding.reason)
        if finding.severity == "hard" or finding.id not in severity:
            severity[finding.id] = finding.severity
    return ({sid: "; ".join(dict.fromkeys(rows)) for sid, rows in reasons.items()}, severity)


def _merge_reasons(*maps: dict[str, str]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for mapping in maps:
        for sid, reason in mapping.items():
            if reason:
                out.setdefault(sid, []).append(reason)
    return {sid: "; ".join(dict.fromkeys(rows)) for sid, rows in out.items()}


def _single_fresh_candidates(harness, targets, selected, memory, *, workers: int = 3) -> dict[str, str]:
    position = {segment.id: index for index, segment in enumerate(targets)}

    def run(segment):
        idx = position.get(segment.id, 0)
        before = targets[max(0, idx - 3) : idx]
        after = targets[idx + 1 : idx + 4]
        try:
            rows = harness.translator.translate(
                [segment],
                memory,
                context_before=before,
                context_after=after,
            )
            return segment.id, str(rows.get(segment.id) or "").strip()
        except Exception as exc:
            print(f"[v6-fresh-single] id={segment.id} error={type(exc).__name__}", flush=True)
            return segment.id, ""

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bookai-v6-fresh") as pool:
        futures = [pool.submit(run, segment) for segment in selected]
        for future in as_completed(futures):
            sid, value = future.result()
            if value:
                out[sid] = value
    return out


def _normalize_typography(segment: Segment, text: str) -> str:
    value = re.sub(r"[ \t]+", " ", str(text or "").strip())
    # Safe deterministic case: a whole source paragraph is one quoted utterance and
    # the Russian output is also wholly wrapped in straight/smart quotes.
    if segment.text.lstrip().startswith(("'", '"', "‘", "“")):
        match = _DIALOGUE_ONLY.match(value)
        if match:
            body = match.group(2).strip()
            punct = match.group(4) or ""
            value = "— " + body + punct
    return value


def _semantic_qe_repair(harness, targets, translated, memory) -> dict:
    global _V6_QE_STATS, _V6_REMAINING_HARD, _V6_REMAINING_MEDIUM, _V6_REASONS

    qe_chars = max(8000, int(os.getenv("BOOKAI_UNIFIED_QE_BATCH_CHARS") or "14500"))
    qe_workers = max(1, min(6, int(os.getenv("BOOKAI_UNIFIED_QE_WORKERS") or "5")))
    edit_workers = max(1, min(4, int(os.getenv("BOOKAI_TARGETED_EDIT_WORKERS") or "4")))
    repair_cap = max(0, int(os.getenv("BOOKAI_DEEPSEEK_SEGMENT_CAP") or "60"))
    second_cap = max(0, int(os.getenv("BOOKAI_SECOND_REPAIR_MAX") or "10"))
    fresh_cap = max(0, int(os.getenv("BOOKAI_FRESH_RETRANSLATE_MAX") or "6"))

    batches = _qe_batches(targets, qe_chars)
    started = time.perf_counter()
    hybrid.progress({
        "phase": "v6_unified_qe",
        "audited": len(targets),
        "batches": len(batches),
        "workers": qe_workers,
        "strategy": "one-publication-QE→one-targeted-edit→hard-only-second-pass→single-fresh-tail",
    })
    findings = _parallel_unified_qe(harness, batches, translated, memory, workers=qe_workers)
    critic_reasons, critic_severity = _reason_map(findings)
    det_ids, det_reasons = _source_only_hard(targets, translated, memory)
    reasons = _merge_reasons(critic_reasons, det_reasons)
    severity = dict(critic_severity)
    for sid in det_ids:
        severity[sid] = "hard"

    position = {segment.id: index for index, segment in enumerate(targets)}
    by_id = {segment.id: segment for segment in targets}
    hard_ids = [sid for sid in severity if severity[sid] == "hard"]
    medium_ids = [sid for sid in severity if severity[sid] != "hard"]
    ordered = sorted(hard_ids, key=lambda sid: position.get(sid, 10**9)) + sorted(
        medium_ids, key=lambda sid: position.get(sid, 10**9)
    )
    selected_ids = ordered[:repair_cap] if repair_cap else []
    overflow = set(ordered[repair_cap:]) if repair_cap else set(ordered)

    selected = [by_id[sid] for sid in selected_ids if sid in by_id]
    edits = v4b._targeted_edit_batched(
        harness,
        targets,
        selected,
        translated,
        memory,
        reasons,
        workers=edit_workers,
    )
    candidate_map = dict(translated)
    candidate_map.update(edits)
    edited_targets = [by_id[sid] for sid in edits if sid in by_id]
    recheck = _parallel_unified_qe(
        harness,
        _qe_batches(edited_targets, qe_chars),
        candidate_map,
        memory,
        workers=qe_workers,
    )
    recheck_reasons, recheck_severity = _reason_map(recheck)
    det_bad, det_bad_reasons = _source_only_hard(edited_targets, candidate_map, memory)
    for sid in det_bad:
        recheck_severity[sid] = "hard"
    recheck_reasons = _merge_reasons(recheck_reasons, det_bad_reasons)

    accepted: set[str] = set()
    for sid in edits:
        original_severity = severity.get(sid, "medium")
        new_severity = recheck_severity.get(sid)
        if new_severity == "hard":
            continue
        if original_severity == "hard" or new_severity is None:
            translated[sid] = edits[sid]
            accepted.add(sid)

    pending_hard = {
        sid for sid in hard_ids
        if sid not in accepted
    } | {sid for sid in overflow if severity.get(sid) == "hard"}
    remaining_medium = {
        sid for sid in medium_ids
        if sid not in accepted
    } | {sid for sid in overflow if severity.get(sid) != "hard"}
    reasons = _merge_reasons(reasons, recheck_reasons)

    second_accepted = 0
    if pending_hard and second_cap:
        second_ids = sorted(pending_hard, key=lambda sid: position.get(sid, 10**9))[:second_cap]
        second_targets = [by_id[sid] for sid in second_ids if sid in by_id]
        second_edits = v4b._targeted_edit_batched(
            harness,
            targets,
            second_targets,
            translated,
            memory,
            reasons,
            workers=edit_workers,
        )
        second_map = dict(translated)
        second_map.update(second_edits)
        second_changed = [by_id[sid] for sid in second_edits if sid in by_id]
        second_findings = _parallel_unified_qe(
            harness,
            _qe_batches(second_changed, qe_chars),
            second_map,
            memory,
            workers=qe_workers,
        )
        _, second_severity = _reason_map(second_findings)
        second_det, _ = _source_only_hard(second_changed, second_map, memory)
        for sid in second_det:
            second_severity[sid] = "hard"
        for sid, value in second_edits.items():
            if second_severity.get(sid) != "hard":
                translated[sid] = value
                pending_hard.discard(sid)
                second_accepted += 1

    fresh_attempted = 0
    fresh_accepted = 0
    if pending_hard and fresh_cap:
        fresh_ids = sorted(pending_hard, key=lambda sid: position.get(sid, 10**9))[:fresh_cap]
        fresh_targets = [by_id[sid] for sid in fresh_ids if sid in by_id]
        fresh = _single_fresh_candidates(harness, targets, fresh_targets, memory, workers=3)
        fresh_attempted = len(fresh)
        fresh_map = dict(translated)
        fresh_map.update(fresh)
        changed = [by_id[sid] for sid in fresh if sid in by_id]
        fresh_findings = _parallel_unified_qe(
            harness,
            _qe_batches(changed, qe_chars),
            fresh_map,
            memory,
            workers=min(qe_workers, 3),
        )
        _, fresh_severity = _reason_map(fresh_findings)
        fresh_det, _ = _source_only_hard(changed, fresh_map, memory)
        for sid in fresh_det:
            fresh_severity[sid] = "hard"
        for sid, value in fresh.items():
            if fresh_severity.get(sid) != "hard":
                translated[sid] = value
                pending_hard.discard(sid)
                fresh_accepted += 1

    for segment in targets:
        if segment.id in translated:
            translated[segment.id] = _normalize_typography(segment, translated[segment.id])

    _V6_REMAINING_HARD = set(pending_hard)
    _V6_REMAINING_MEDIUM = set(remaining_medium)
    _V6_REASONS = {sid: reasons.get(sid, "v6 unified QE unresolved") for sid in pending_hard | remaining_medium}
    _V6_QE_STATS = {
        "audited": len(targets),
        "flagged": len(severity),
        "hard_flagged": len(hard_ids),
        "medium_flagged": len(medium_ids),
        "first_edit_attempted": len(selected),
        "first_edit_accepted": len(accepted),
        "second_hard_accepted": second_accepted,
        "fresh_attempted": fresh_attempted,
        "fresh_accepted": fresh_accepted,
        "remaining_hard": len(pending_hard),
        "remaining_medium": len(remaining_medium),
        "qe_elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    hybrid.progress({"phase": "v6_unified_qe_done", **_V6_QE_STATS})
    return dict(_V6_QE_STATS)


def _finalize_without_second_llm_pass(
    harness,
    targets,
    translated,
    memory,
    state,
    *,
    chapter_digests,
    book_synopsis,
    context_index,
):
    # ThreadWeaver-lite: use the source-derived entity ledger to check recurrent
    # named threads mechanically. No second chapter-wide literary LLM read.
    locked = _source_locked_glossary(memory)
    violations = locked_glossary_violations(targets, translated, locked) if locked else {}
    state["v6_thread_consistency"] = {
        "locked_named_entities": len(locked),
        "segments_with_entity_mismatch": len(violations),
        "ids": sorted(violations)[:30],
    }

    # Self-learning TM contains only our own post-QE translations. This is what a
    # real user has: no gold reference. Future chapters may retrieve these examples.
    hard_now, _ = _source_only_hard(targets, translated, memory)
    approved = []
    for segment in targets:
        value = str(translated.get(segment.id) or "").strip()
        if not value or segment.id in hard_now or segment.id in _V6_REMAINING_HARD:
            continue
        approved.append({
            "source": segment.text,
            "translation": value,
            "chapter": segment.chapter,
            "quality": 1.0,
        })
    previous = [row for row in list(state.get("v6_translation_memory") or []) if isinstance(row, dict)]
    merged = (previous + approved)[-5000:]
    state["v6_translation_memory"] = merged
    state["v6_tm_stats"] = {
        "previous": len(previous),
        "added": len(approved),
        "total": len(merged),
        "source": "self-approved-only",
    }
    GigaChatLightningV6Backend.set_translation_memory(merged)
    hybrid.progress({
        "phase": "v6_finalize",
        "tm_added": len(approved),
        "thread_entity_mismatches": len(violations),
        "second_literary_llm_pass": 0,
    })
    return translated


def _final_issues(segments, translations, memory=None, *, source_segments=None):
    issues = list(
        _ORIGINAL_ENHANCED_BATCH_ISSUES(
            segments,
            translations,
            memory,
            source_segments=source_segments,
        )
    )
    existing = {(issue.id, issue.code) for issue in issues}
    valid = {segment.id for segment in segments}
    for sid in sorted(_V6_REMAINING_HARD):
        if sid in valid and (sid, "v6_qe_unresolved") not in existing:
            issues.append(QualityIssueV3(
                sid,
                "hard",
                "v6_qe_unresolved",
                _V6_REASONS.get(sid, "unified publication QE still reports a hard defect"),
            ))
    return issues


def _configure() -> None:
    _SOURCE_READY.clear()
    slug = "largest" if str(v3.CHAPTER_NAME or "").strip().casefold() in _LARGEST_ALIASES else v3.CHAPTER_SLUG
    v3.OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V6.fb2")
    v3.CACHE = Path(f".bookai-cache-chapter-eval-v6-{slug}")
    v3.PROGRESS = Path("chapter-v6-progress.json")
    v3.PROBE = Path("chapter-v6-probe.json")
    v3.ROUTING = Path("chapter-v6-routing.json")
    v3.REPORT = Path("chapter-v6.json")
    v3.SOURCE_TXT = Path("chapter-v6-source.txt")
    v3.TRANSLATED_TXT = Path("chapter-v6-translated.txt")
    v3.MAP_JSON = Path("chapter-v6-translation-map.json")

    # Critical no-reference guarantees.
    hybrid._base_memory = lambda: BookMemory()
    hybrid.build_reference_harness = TranslationHarness.from_env

    v3._select_chapter = _select_chapter
    v3.GigaChatLightningV3Backend = GigaChatLightningV6Backend
    v3._candidate_bad_v3 = v5._draft_bad
    v3._analysis_task = _source_analysis_task
    hybrid._translate_hybrid_batch = _translate_gigachat_v6
    hybrid._deepseek_final_recovery = v5._direct_deepseek_recovery

    v3._semantic_short_repair = _semantic_qe_repair
    hybrid._repair_hard_failures = v5._skip_duplicate_hard_repair
    hybrid._selective_literary_refinement = _finalize_without_second_llm_pass
    v3.enhanced_batch_issues = _final_issues


def _annotate_report() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        "version": "quality-v6-book-intelligence-source-only",
        "gold_reference_available_to_pipeline": False,
        "reference_profile": False,
        "reference_glossary_seed": False,
        "book_intelligence": "single whole-English-book source-only preflight",
        "primary": "GigaChat-3-Lightning",
        "translation_memory": "self-approved translations only; retrieval for later chapters",
        "qe": "single unified fidelity+literary publication audit",
        "repair": "one main targeted edit; second pass hard-only and bounded",
        "hard_tail": "single-segment independent DeepSeek candidates",
        "thread_weaver": "deterministic source-derived named-entity consistency",
        "literary_second_full_pass": False,
        "deepseek_v4_pro": False,
    }
    report["v6_book_intelligence"] = dict(_V6_BOOK_STATS)
    report["v6_qe"] = dict(_V6_QE_STATS)
    report["v6_remaining_hard"] = [
        {"id": sid, "reason": _V6_REASONS.get(sid, "unresolved")}
        for sid in sorted(_V6_REMAINING_HARD)
    ]
    report["v6_remaining_medium"] = [
        {"id": sid, "reason": _V6_REASONS.get(sid, "advisory")}
        for sid in sorted(_V6_REMAINING_MEDIUM)
    ]
    if _V6_REMAINING_HARD:
        report["status"] = "needs_review"
        report["state_status"] = "needs_review"
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure()
    try:
        v3.main()
    finally:
        _annotate_report()


if __name__ == "__main__":
    main()
