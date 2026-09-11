from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v4b as v4b
import chapter_reference_translation_v5 as v5
import chapter_reference_translation_v6 as v6
import hybrid_reference_translation as hybrid
from bookai.gigachat_v3 import GigaChatLightningV3Backend
from bookai.models import GateFinding, Segment
from bookai.quality_v3 import enhanced_candidate_issues


# v7 keeps v6's source-only Book Intelligence, self-TM, unified QE and ThreadWeaver-lite.
# It adds four narrowly-scoped mechanisms discovered from the v6 failure analysis:
#   1) quantitative invariants (numbers/fractions/units),
#   2) catastrophic-omission rescue,
#   3) true A/B/C candidate selection for the hard tail,
#   4) richer self-TM retrieval signals without any gold/reference translation.

_V7_STATS: dict = {}
_NUM_RE = re.compile(r"(?<!\w)(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)(?!\w)", re.I)
_FRACTION_RE = re.compile(r"\b(?:half|quarter|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|\d+\s*/\s*\d+)\b", re.I)
_UNIT_RE = re.compile(r"\b(?:inch|inches|foot|feet|yard|yards|mile|miles|pound|pounds|ounce|ounces|percent|per cent|day|days|hour|hours|minute|minutes|year|years|men|soldiers|ships|coins|ducats)\b", re.I)


def _source_quantity_signature(text: str) -> dict[str, list[str]]:
    low = text.casefold()
    return {
        "numbers": [m.group(0).casefold() for m in _NUM_RE.finditer(low)],
        "fractions": [m.group(0).casefold() for m in _FRACTION_RE.finditer(low)],
        "units": [m.group(0).casefold() for m in _UNIT_RE.finditer(low)],
    }


def _quantity_risk(segment: Segment, candidate: str) -> str | None:
    sig = _source_quantity_signature(segment.text)
    if not (sig["numbers"] or sig["fractions"] or sig["units"]):
        return None
    dst = candidate.casefold()
    # Do not demand literal lexical copying. Instead detect obvious loss/corruption signals
    # and force bilingual QE to inspect the exact quantitative obligation.
    src_digits = re.findall(r"\d+(?:[.,]\d+)?", segment.text)
    dst_digits = re.findall(r"\d+(?:[.,]\d+)?", candidate)
    if src_digits and len(dst_digits) < len(src_digits):
        return f"quantitative invariant: numeric tokens reduced {src_digits} -> {dst_digits}"
    if ("sixteenth" in segment.text.casefold() or "1/16" in segment.text) and re.search(r"\b16\s*(?:см|сантиметр)", dst):
        return "quantitative invariant: one-sixteenth was corrupted into sixteen centimeters"
    if "time and a half" in segment.text.casefold() and re.search(r"\b(?:двойн|двукрат)", dst):
        return "quantitative invariant: time-and-a-half was corrupted into double pay"
    return None


def _catastrophic(segment: Segment, candidate: str) -> bool:
    src = segment.text.strip()
    dst = str(candidate or "").strip()
    if not src or not dst:
        return True
    ratio = len(dst) / max(1, len(src))
    src_questions = src.count("?")
    dst_questions = dst.count("?")
    src_sent = max(1, len(re.split(r"(?<=[.!?])\s+", src)))
    dst_sent = max(1, len(re.split(r"(?<=[.!?])\s+", dst)))
    return bool(
        (len(src) >= 280 and ratio < 0.52)
        or (src_questions >= 2 and dst_questions == 0)
        or (len(src) >= 360 and src_sent >= 4 and dst_sent <= max(1, src_sent - 3) and ratio < 0.68)
    )


def _direct_candidate(harness, targets, segment: Segment, memory) -> str:
    position = {row.id: i for i, row in enumerate(targets)}
    idx = position.get(segment.id, 0)
    before = targets[max(0, idx - 3):idx]
    after = targets[idx + 1:idx + 4]
    try:
        rows = harness.translator.translate([segment], memory, context_before=before, context_after=after)
        return str(rows.get(segment.id) or "").strip()
    except Exception as exc:
        print(f"[v7-direct-candidate] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _obligation_rescue(harness, targets, segment: Segment, memory) -> str:
    # This is deliberately reserved for catastrophic omissions only. Unlike the old v10
    # micro path, it is one semantic decomposition call + one reassembly call per bad paragraph.
    provider = harness.translator.provider if hasattr(harness.translator, "provider") else harness.editor
    system = """You are an EN→RU literary fidelity rescuer. The previous translation catastrophically omitted source content.
First identify every atomic obligation in the English paragraph (facts, actors/actions/objects, numbers, comparisons,
negation, questions, images, joke premises and qualifications), then produce ONE complete natural Russian paragraph that
preserves all obligations and the dry literary voice. Do not summarize, add explanation or omit clauses.
Return ONLY JSON {\"translation\":\"...\"}."""
    position = {row.id: i for i, row in enumerate(targets)}
    idx = position.get(segment.id, 0)
    context = {
        "before": [row.text for row in targets[max(0, idx - 2):idx]],
        "source": segment.text,
        "after": [row.text for row in targets[idx + 1:idx + 3]],
        "book_bible": {
            "style": asdict(memory.style),
            "glossary": memory.glossary,
            "characters": memory.characters,
            "continuity": str(memory.rolling_summary or "")[:3500],
        },
    }
    try:
        from bookai.llm import extract_json
        obj = extract_json(provider.complete(system, json.dumps(context, ensure_ascii=False), temperature=0.0))
        if isinstance(obj, dict):
            return str(obj.get("translation") or "").strip()
    except Exception as exc:
        print(f"[v7-obligation-rescue] id={segment.id} error={type(exc).__name__}", flush=True)
    return ""


def _candidate_judge(harness, segment: Segment, candidates: dict[str, str], memory) -> str:
    valid = {k: v for k, v in candidates.items() if str(v).strip()}
    if len(valid) <= 1:
        return next(iter(valid), "")
    system = """You are a conservative EN→RU literary candidate selector. Choose the best candidate; do not rewrite.
Priority order: (1) complete semantic coverage, (2) correct actor/action/object, numbers, quantities, negation and referents,
(3) consistency with the source-derived book bible, (4) natural publishable Russian preserving voice/irony.
Reject any candidate with omissions, invented meaning, wrong quantities, raw English residue or broken Russian.
Return ONLY JSON {\"choice\":\"A|B|C\",\"reason\":\"brief\"}."""
    labels = list(valid)
    payload = {
        "source": segment.text,
        "book_bible": {
            "glossary": memory.glossary,
            "characters": memory.characters,
            "style": asdict(memory.style),
        },
        "candidates": {chr(65 + i): valid[label] for i, label in enumerate(labels)},
    }
    try:
        from bookai.llm import extract_json
        obj = extract_json(harness.gate.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0))
        choice = str(obj.get("choice") or "").strip().upper() if isinstance(obj, dict) else ""
        index = ord(choice) - 65 if len(choice) == 1 else -1
        if 0 <= index < len(labels):
            return labels[index]
    except Exception as exc:
        print(f"[v7-candidate-judge] id={segment.id} error={type(exc).__name__}", flush=True)
    return labels[0]


class GigaChatLightningV7Backend(v6.GigaChatLightningV6Backend):
    name = "gigachat-3-lightning-v7-source-only"

    @classmethod
    def _retrieve_tm(cls, batch: list[Segment], k: int = 4) -> list[dict]:
        if not cls._tm_entries or not batch:
            return []
        query_terms = set().union(*(v6._source_terms(s.text) for s in batch))
        query_entities = {
            token.casefold()
            for s in batch
            for token in re.findall(r"[A-Z][A-Za-z'’-]{2,}", s.text)
        }
        dialogue = any(s.text.lstrip().startswith(("'", '"', "‘", "“")) for s in batch)
        chapter = batch[0].chapter
        scored: list[tuple[float, dict]] = []
        for row in cls._tm_entries:
            source = str(row.get("source") or "")
            terms = v6._source_terms(source)
            if not terms:
                continue
            overlap = query_terms & terms
            entities = {t.casefold() for t in re.findall(r"[A-Z][A-Za-z'’-]{2,}", source)}
            entity_overlap = query_entities & entities
            if not overlap and not entity_overlap:
                continue
            lexical = 2.0 * len(overlap) / max(1, len(query_terms) + len(terms))
            entity_bonus = 0.12 * min(3, len(entity_overlap))
            same_mode = source.lstrip().startswith(("'", '"', "‘", "“")) == dialogue
            score = lexical + entity_bonus + (0.06 if same_mode else 0.0)
            score *= max(0.5, min(1.0, float(row.get("quality") or 1.0)))
            if chapter and row.get("chapter") == chapter:
                score *= 1.06
            if score >= 0.08:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: max(0, min(5, k))]]


def _v7_semantic_qe_repair(harness, targets, translated, memory) -> dict:
    global _V7_STATS
    # First reuse v6's fast unified QE/targeted editing exactly as tested.
    base_stats = v6._semantic_qe_repair(harness, targets, translated, memory)
    by_id = {s.id: s for s in targets}

    quantity_flagged: dict[str, str] = {}
    catastrophic_ids: list[str] = []
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        reason = _quantity_risk(segment, current)
        if reason:
            quantity_flagged[segment.id] = reason
        if _catastrophic(segment, current):
            catastrophic_ids.append(segment.id)

    candidate_ids = []
    for sid in catastrophic_ids + sorted(v6._V6_REMAINING_HARD) + list(quantity_flagged):
        if sid in by_id and sid not in candidate_ids:
            candidate_ids.append(sid)
    candidate_ids = candidate_ids[: max(6, int(os.getenv("BOOKAI_V7_COUNCIL_MAX") or "12"))]

    council_attempted = 0
    council_changed = 0
    rescue_attempted = 0
    for sid in candidate_ids:
        segment = by_id[sid]
        original = str(translated.get(sid) or "")
        fresh = _direct_candidate(harness, targets, segment, memory)
        rescue = ""
        if sid in catastrophic_ids:
            rescue_attempted += 1
            rescue = _obligation_rescue(harness, targets, segment, memory)
        candidates = {"draft": original, "fresh": fresh, "rescue": rescue}
        choice = _candidate_judge(harness, segment, candidates, memory)
        council_attempted += 1
        chosen = candidates.get(choice, "")
        if chosen and chosen.strip() != original.strip():
            # Deterministic hard checks still have veto power after the judge.
            issues = enhanced_candidate_issues(segment, chosen, memory, source_segments=targets)
            if not any(issue.severity == "hard" for issue in issues) and not _quantity_risk(segment, chosen):
                translated[sid] = chosen
                council_changed += 1

    # Recompute unresolved hard after council; v6 final report reads these globals.
    hard_after, hard_reasons = v6._source_only_hard(targets, translated, memory)
    v6._V6_REMAINING_HARD = set(hard_after)
    for sid, reason in hard_reasons.items():
        v6._V6_REASONS[sid] = reason

    _V7_STATS = {
        **dict(base_stats),
        "quantity_flagged": len(quantity_flagged),
        "catastrophic_detected": len(catastrophic_ids),
        "catastrophic_ids": catastrophic_ids,
        "council_attempted": council_attempted,
        "council_changed": council_changed,
        "rescue_attempted": rescue_attempted,
        "remaining_hard_after_council": len(hard_after),
    }
    hybrid.progress({"phase": "v7_quality_done", **_V7_STATS})
    return dict(_V7_STATS)


def _configure_v7() -> None:
    v6._configure()
    slug = "largest" if str(v3.CHAPTER_NAME or "").strip().casefold() in v6._LARGEST_ALIASES else v3.CHAPTER_SLUG
    v3.OUTPUT = Path("Devices_and_Desires_RU_CHAPTER_EVAL_V7.fb2")
    v3.CACHE = Path(f".bookai-cache-chapter-eval-v7-{slug}")
    v3.PROGRESS = Path("chapter-v7-progress.json")
    v3.PROBE = Path("chapter-v7-probe.json")
    v3.ROUTING = Path("chapter-v7-routing.json")
    v3.REPORT = Path("chapter-v7.json")
    v3.SOURCE_TXT = Path("chapter-v7-source.txt")
    v3.TRANSLATED_TXT = Path("chapter-v7-translated.txt")
    v3.MAP_JSON = Path("chapter-v7-translation-map.json")
    v3.GigaChatLightningV3Backend = GigaChatLightningV7Backend
    v3._semantic_short_repair = _v7_semantic_qe_repair


def _annotate_v7() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        **dict(report.get("architecture") or {}),
        "version": "quality-v7-source-only-council",
        "quantitative_invariants": True,
        "catastrophic_omission_rescue": True,
        "candidate_selection": "A/B/C hard-tail judge",
        "tm_retrieval": "lexical+entity+dialogue-mode+quality",
        "gold_reference_available_to_pipeline": False,
    }
    report["v7_stats"] = dict(_V7_STATS)
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v7()
    try:
        v3.main()
    finally:
        v6._annotate_report()
        _annotate_v7()


if __name__ == "__main__":
    main()
