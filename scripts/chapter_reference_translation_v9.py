from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v6 as v6
import chapter_reference_translation_v7 as v7
import chapter_reference_translation_v8 as v8
import chapter_reference_translation_v8c as v8c
import hybrid_reference_translation as hybrid
from bookai.gigachat_v3 import GigaChatLightningV3Backend
from bookai.llm import extract_json
from bookai.models import BookMemory, Segment
from bookai.quality_v3 import QualityIssueV3, enhanced_batch_issues


# quality-v9: production architecture synthesized from the v8c benchmark and
# patterns used by mature translation/localization products:
#   structural book parser -> persistent source-only Book Blueprint
#   -> weighted trusted/working self-TM retrieval -> GigaChat bulk draft
#   -> deterministic fidelity invariants + continuous confidence-aware QE
#   -> specialist repairs only for high-confidence defects
#   -> tiny ambiguity candidate path for idiom/dialogue/referent cases
#   -> Thread Index consistency -> deterministic typography -> trusted TM update.
# No gold/reference Russian text is available to any model call.

_V9_STATS: dict[str, Any] = {}
_V9_SCORES: dict[str, float] = {}
_V9_FINAL_FINDINGS: dict[str, list[dict[str, Any]]] = {}
_V9_BLUEPRINT: dict[str, Any] = {}
_V9_THREAD_MISMATCHES: set[str] = set()
_V9_RELIABILITY: dict[str, Any] = {}

_ASCII_WORD = re.compile(r"\b[A-Za-z]{3,}\b")
_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
_SOURCE_DIALOGUE = re.compile(r"^\s*[\"'“‘]")

_MATERIALS = {
    "brass": ("латун",),
    "bronze": ("бронз",),
    "steel": ("стал",),
    "iron": ("желез",),
    "copper": ("мед",),
    "silver": ("серебр",),
    "gold": ("золот",),
    "lead": ("свинц",),
    "tin": ("олов",),
    "wood": ("дерев", "древес"),
    "coal": ("угол",),
}
_CONTRAST_RE = re.compile(
    r"\b(?:not|rather\s+than|instead\s+of|as\s+opposed\s+to|but\s+not)\b",
    re.I,
)
_REFERENT_RE = re.compile(
    r"\b(?:either|neither|both|former|latter|the other|each other|one another|them|they|their|his|her)\b",
    re.I,
)

_SEVERITY_PENALTY = {"critical": 52.0, "major": 24.0, "minor": 7.0, "ambiguous": 4.0}
_SPECIAL_CODES = {"idiom", "dialogue", "referent", "relation", "irony", "voice"}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _memory_to_dict(memory: BookMemory) -> dict[str, Any]:
    return {
        "style": asdict(memory.style),
        "glossary": dict(memory.glossary),
        "characters": dict(memory.characters),
        "rolling_summary": str(memory.rolling_summary or ""),
    }


def _memory_from_dict(data: dict[str, Any]) -> BookMemory:
    memory = BookMemory()
    style_data = dict(data.get("style") or {})
    for key, value in style_data.items():
        if hasattr(memory.style, key):
            setattr(memory.style, key, value)
    memory.glossary = {str(k): str(v) for k, v in dict(data.get("glossary") or {}).items()}
    memory.characters = {str(k): str(v) for k, v in dict(data.get("characters") or {}).items()}
    memory.rolling_summary = str(data.get("rolling_summary") or "")
    return memory


def _source_terms(text: str) -> set[str]:
    return {token.casefold() for token in _EN_WORD.findall(text or "") if len(token) >= 4}


def _source_entities(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _EN_WORD.findall(text or "")
        if token[:1].isupper() and len(token) >= 4
    }


def _is_dialogue(text: str) -> bool:
    return bool(_SOURCE_DIALOGUE.search(text or ""))


def _tm_score(batch: list[Segment], row: dict[str, Any], *, trusted: bool) -> float:
    if not batch:
        return 0.0
    source = str(row.get("source") or "")
    if not source:
        return 0.0
    query_text = " ".join(segment.text for segment in batch)
    q_terms = _source_terms(query_text)
    r_terms = _source_terms(source)
    if not q_terms or not r_terms:
        return 0.0
    overlap = q_terms & r_terms
    lexical = len(overlap) / max(1, len(q_terms | r_terms))
    q_entities = _source_entities(query_text)
    r_entities = _source_entities(source)
    entity = len(q_entities & r_entities) / max(1, len(q_entities | r_entities)) if (q_entities or r_entities) else 0.0
    mode = 1.0 if _is_dialogue(batch[0].text) == _is_dialogue(source) else 0.0
    chapter = 1.0 if batch[0].chapter and str(row.get("chapter") or "") == batch[0].chapter else 0.0
    quality = max(0.0, min(1.0, float(row.get("quality") or 0.75)))
    tier = 1.0 if trusted else 0.45
    return 0.38 * lexical + 0.20 * entity + 0.08 * mode + 0.05 * chapter + 0.14 * quality + 0.15 * tier


class GigaChatLightningV9Backend(GigaChatLightningV3Backend):
    name = "gigachat-3-lightning-v9-weighted-self-tm"
    _working_tm: list[dict[str, Any]] = []
    _trusted_tm: list[dict[str, Any]] = []

    @classmethod
    def set_memories(cls, working: list[dict[str, Any]], trusted: list[dict[str, Any]]) -> None:
        def clean(rows):
            out = []
            for row in rows[-6000:]:
                if not isinstance(row, dict):
                    continue
                source = _norm_text(row.get("source") or "")
                translation = _norm_text(row.get("translation") or "")
                if source and translation:
                    out.append({
                        "source": source,
                        "translation": translation,
                        "chapter": str(row.get("chapter") or ""),
                        "quality": float(row.get("quality") or 0.8),
                    })
            return out
        cls._working_tm = clean(working)
        cls._trusted_tm = clean(trusted)

    @classmethod
    def _retrieve_weighted(cls, batch: list[Segment], k: int) -> list[dict[str, Any]]:
        scored: list[tuple[float, bool, dict[str, Any]]] = []
        seen_sources: set[str] = set()
        for trusted, rows in ((True, cls._trusted_tm), (False, cls._working_tm)):
            for row in rows:
                score = _tm_score(batch, row, trusted=trusted)
                if score < 0.11:
                    continue
                scored.append((score, trusted, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = []
        for score, trusted, row in scored:
            key = str(row.get("source") or "").casefold()
            if key in seen_sources:
                continue
            seen_sources.add(key)
            result.append({**row, "retrieval_score": round(score, 4), "tier": "trusted" if trusted else "working"})
            if len(result) >= max(0, min(6, k)):
                break
        return result

    def _prompt(self, batch, memory, *, source_segments=None, minimal=False):
        base = GigaChatLightningV3Backend._prompt(
            self,
            batch,
            memory,
            source_segments=source_segments,
            minimal=minimal,
        )
        examples = self._retrieve_weighted(batch, int(os.getenv("BOOKAI_TM_RETRIEVAL_K") or "4"))
        if not examples:
            return base
        payload = [
            {
                "source": row["source"][:1100],
                "approved_ru": row["translation"][:1300],
                "tier": row["tier"],
                "quality": round(float(row.get("quality") or 0.8), 3),
                "relevance": row["retrieval_score"],
            }
            for row in examples
        ]
        return (
            base
            + "\n\nWEIGHTED_SELF_TRANSLATION_MEMORY (same source book only):\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\nTrusted examples are stronger evidence than working examples. Use memory only for recurring names, terminology, register and voice. Never override the current SOURCE TARGETS."
        )


def _analysis_task_v9(harness, chapters, state_snapshot: dict):
    global _V9_BLUEPRINT
    started = time.perf_counter()
    cached = state_snapshot.get("v9_book_blueprint")
    if isinstance(cached, dict) and cached.get("glossary") is not None:
        memory = _memory_from_dict(cached)
        v6._SOURCE_MEMORY = memory
        v6._SOURCE_READY.set()
        compact = _norm_text(memory.rolling_summary)[:6000]
        chapter_digests = {name: compact for name, _ in chapters if compact}
        book_synopsis = compact
        elapsed = time.perf_counter() - started
        v6._V6_BOOK_STATS = {
            "source_only": True,
            "source_segments_seen": len(v6._SOURCE_SEGMENTS),
            "sample_chars": 0,
            "glossary_entries": len(memory.glossary),
            "characters": len(memory.characters),
            "cached_tm_entries": len(state_snapshot.get("v9_trusted_tm") or []),
            "elapsed_seconds": round(elapsed, 2),
            "blueprint_reused": True,
        }
        state_snapshot["v6_book_intelligence"] = dict(v6._V6_BOOK_STATS)
        state_snapshot["chapter_briefs"] = chapter_digests
        state_snapshot["book_synopsis"] = book_synopsis
    else:
        memory, chapter_digests, book_synopsis, state_snapshot, _ = v6._source_analysis_task(
            harness, chapters, state_snapshot
        )
        state_snapshot["v9_book_blueprint"] = _memory_to_dict(memory)
        elapsed = time.perf_counter() - started
        v6._V6_BOOK_STATS["blueprint_reused"] = False

    _V9_BLUEPRINT = _memory_to_dict(memory)
    working = list(state_snapshot.get("v9_working_tm") or [])
    trusted = list(state_snapshot.get("v9_trusted_tm") or [])
    GigaChatLightningV9Backend.set_memories(working, trusted)
    state_snapshot["v9_blueprint_stats"] = {
        "reused": bool(cached),
        "glossary_entries": len(memory.glossary),
        "characters": len(memory.characters),
        "working_tm": len(working),
        "trusted_tm": len(trusted),
        "elapsed_seconds": round(elapsed, 2),
    }
    hybrid.progress({"phase": "v9_blueprint_ready", **state_snapshot["v9_blueprint_stats"]})
    return memory, chapter_digests, book_synopsis, state_snapshot, elapsed


def _reliability_recovery(harness, targets, translated, pending_ids, memory, stats):
    global _V9_RELIABILITY
    pending = set(pending_ids or [])
    if not pending:
        return set()
    by_id = {segment.id: segment for segment in targets}
    selected = [by_id[sid] for sid in pending if sid in by_id]
    fresh = v6._single_fresh_candidates(
        harness,
        targets,
        selected,
        memory,
        workers=max(1, min(5, int(os.getenv("BOOKAI_RELIABILITY_WORKERS") or "4"))),
    )
    translated.update(fresh)
    remaining = {sid for sid in pending if sid not in translated}
    try:
        stats.add(
            deep_direct=len(fresh),
            source_chars_deep=sum(len(by_id[sid].text) for sid in fresh if sid in by_id),
        )
    except Exception:
        pass
    _V9_RELIABILITY = {
        "requested": len(pending),
        "recovered": len(fresh),
        "remaining": len(remaining),
        "ids_remaining": sorted(remaining),
    }
    hybrid.progress({"phase": "v9_backend_fallback", **_V9_RELIABILITY})
    return remaining


def _extract_invariants(text: str) -> dict[str, Any]:
    lower = str(text or "").casefold()
    materials = [name for name in _MATERIALS if re.search(rf"\b{re.escape(name)}\b", lower)]
    digits = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?", text or "")
    percentages = re.findall(r"\b(\d+(?:[.,]\d+)?)\s*(?:%|per\s*cent|percent)\b", lower)
    return {
        "materials": materials,
        "material_contrast": bool(len(materials) >= 2 and _CONTRAST_RE.search(lower)),
        "digits": digits,
        "percentages": percentages,
        "time_and_a_half": bool(re.search(r"\btime\s+and\s+a\s+half\b", lower)),
        "referent_risk": bool(_REFERENT_RE.search(lower)),
        "dialogue": _is_dialogue(text),
    }


def _target_has_material(target: str, material: str) -> bool:
    low = str(target or "").casefold()
    return any(stem in low for stem in _MATERIALS.get(material, ()))


def _deterministic_findings(segment: Segment, candidate: str, memory: BookMemory) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    source = segment.text
    value = str(candidate or "")
    inv = _extract_invariants(source)

    if v7._catastrophic(segment, value):
        findings.append({
            "id": segment.id, "severity": "critical", "confidence": 1.0,
            "code": "coverage", "source_span": "", "target_span": "",
            "reason": "catastrophic omission/truncation detected by deterministic coverage check",
            "repairability": "local",
        })

    raw = [w for w in _ASCII_WORD.findall(value) if w.casefold() not in {"chapter"}]
    if raw:
        findings.append({
            "id": segment.id, "severity": "critical", "confidence": 0.99,
            "code": "raw_english", "source_span": "", "target_span": " ".join(raw[:5]),
            "reason": "raw English words remain in Russian output",
            "repairability": "local",
        })

    source_low = source.casefold()
    technical_material_context = bool(re.search(
        r"\b(?:bushing|gear|spring|metal|blade|wire|ore|sheet|rod|bar|plate|alloy|material|cast|forg|tool|machine|mechanism)\w*\b",
        source_low,
    ))
    missing_materials = [m for m in inv["materials"] if not _target_has_material(value, m)]
    if missing_materials and (inv["material_contrast"] or technical_material_context):
        findings.append({
            "id": segment.id, "severity": "critical" if inv["material_contrast"] else "major",
            "confidence": 0.98, "code": "material",
            "source_span": ", ".join(missing_materials), "target_span": "",
            "reason": f"material concept(s) lost or substituted: {', '.join(missing_materials)}",
            "repairability": "local",
        })

    # Explicit Arabic-number preservation. This only checks literal numeric tokens;
    # worded quantities are handled by the bilingual audit and specialist editor.
    low_target = value.replace(",", ".")
    for token in inv["digits"]:
        normalized = token.replace(",", ".").rstrip("%")
        if normalized and normalized not in low_target:
            findings.append({
                "id": segment.id, "severity": "major", "confidence": 0.82,
                "code": "number", "source_span": token, "target_span": "",
                "reason": f"explicit source number {token!r} is not visibly preserved in target; verify it was not changed",
                "repairability": "local",
            })
            break

    if inv["time_and_a_half"]:
        low = value.casefold()
        if not any(marker in low for marker in ("полутор", "в полтора", "150%", "150 %")):
            findings.append({
                "id": segment.id, "severity": "critical", "confidence": 0.99,
                "code": "number", "source_span": "time and a half", "target_span": "",
                "reason": "time and a half means 1.5x / полуторная оплата, not double",
                "repairability": "local",
            })

    for source_name, target_name in v8._entity_map(memory).items():
        if v8._source_mentions(source, source_name) and not v8._entity_present(value, target_name):
            findings.append({
                "id": segment.id, "severity": "major", "confidence": 0.9,
                "code": "entity", "source_span": source_name, "target_span": "",
                "reason": f"canonical entity rendering is missing/inconsistent: {source_name} -> {target_name}",
                "repairability": "local",
            })
    return findings


def _audit_batch(provider, batch: list[Segment], translations: dict[str, str], memory: BookMemory, targets: list[Segment]) -> list[dict[str, Any]]:
    if not batch:
        return []
    pos = {segment.id: i for i, segment in enumerate(targets)}
    pairs = {}
    for segment in batch:
        i = pos.get(segment.id, 0)
        pairs[segment.id] = {
            "en": segment.text,
            "ru": translations.get(segment.id, ""),
            "context_before_en": [row.text for row in targets[max(0, i - 2):i]],
            "context_after_en": [row.text for row in targets[i + 1:i + 3]],
            "invariants": _extract_invariants(segment.text),
        }
    compact_memory = {
        "style": asdict(memory.style),
        "glossary": memory.glossary,
        "characters": memory.characters,
        "continuity": str(memory.rolling_summary or "")[:4500],
    }
    system = """You are a conservative bilingual EN→RU literary translation quality estimator. Do NOT rewrite.
Return only defects that materially justify editor attention. Use neighboring source context to resolve pronouns/referents
(either/neither/both/former/latter/they/them/his/her). Respect the source-derived Book Blueprint.
Explicitly verify: omissions/additions; actor-action-object and referents; numbers/fractions/units/materials and contrasts
(e.g. brass vs bronze, not X but Y); negation/modality/chronology/causality; entities/technical referents; idioms/jokes/irony;
clear calques or broken Russian grammar; character/narrator voice and dialogue naturalness.
Do not flag harmless paraphrase or mere stylistic preference.
Severity: critical=meaning/coverage/number/material/referent corruption; major=clear publication defect; minor=real but optional polish;
ambiguous=possible issue that needs context, not enough evidence to assert an error.
Confidence is 0..1. Include exact short source_span and target_span where possible.
Return ONLY JSON {"issues":[{"id":"...","severity":"critical|major|minor|ambiguous","confidence":0.0,
"code":"omission|addition|relation|referent|number|material|entity|term|idiom|calque|grammar|voice|irony|dialogue|other",
"source_span":"...","target_span":"...","reason":"concise","repairability":"local|contextual|none"}]}.
Use [] when no professional edit is warranted. Use only supplied ids."""
    user = "BOOK_BLUEPRINT:" + json.dumps(compact_memory, ensure_ascii=False, separators=(",", ":")) + "\nPAIRS:" + json.dumps(pairs, ensure_ascii=False)
    obj = extract_json(provider.complete(system, user, temperature=0.0))
    rows = obj.get("issues") if isinstance(obj, dict) else []
    valid = {segment.id for segment in batch}
    out = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("id") or "").strip()
        if sid not in valid:
            if len(valid) == 1:
                sid = next(iter(valid))
            else:
                continue
        severity = str(raw.get("severity") or "minor").strip().lower()
        if severity not in _SEVERITY_PENALTY:
            severity = "minor"
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.5)))
        except Exception:
            confidence = 0.5
        reason = _norm_text(raw.get("reason") or "")
        if not reason:
            continue
        out.append({
            "id": sid,
            "severity": severity,
            "confidence": confidence,
            "code": str(raw.get("code") or "other").strip().lower(),
            "source_span": _norm_text(raw.get("source_span") or "")[:240],
            "target_span": _norm_text(raw.get("target_span") or "")[:240],
            "reason": reason[:500],
            "repairability": str(raw.get("repairability") or "local").strip().lower(),
        })
    return out


def _parallel_audit(harness, targets: list[Segment], translations: dict[str, str], memory: BookMemory, selected: list[Segment] | None = None) -> list[dict[str, Any]]:
    rows = selected if selected is not None else targets
    max_chars = max(7000, int(os.getenv("BOOKAI_V9_QE_BATCH_CHARS") or "13500"))
    workers = max(1, min(6, int(os.getenv("BOOKAI_V9_QE_WORKERS") or "5")))
    batches = v6._qe_batches(rows, max_chars)

    def run(batch):
        try:
            return _audit_batch(harness.gate, list(batch), translations, memory, targets)
        except Exception as exc:
            if len(batch) <= 1:
                print(f"[v9-audit] id={batch[0].id if batch else '?'} error={type(exc).__name__}", flush=True)
                return []
            mid = len(batch) // 2
            out = []
            for part in (batch[:mid], batch[mid:]):
                try:
                    out.extend(_audit_batch(harness.gate, list(part), translations, memory, targets))
                except Exception as inner:
                    print(f"[v9-audit-split] size={len(part)} error={type(inner).__name__}", flush=True)
            return out

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9-qe") as pool:
        futures = [pool.submit(run, batch) for batch in batches]
        for future in as_completed(futures):
            out.extend(future.result())
    return out


def _merge_findings(*groups: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for row in group:
            key = (row["id"], row.get("code", ""), row.get("reason", ""))
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(row["id"], []).append(row)
    return out


def _score(rows: list[dict[str, Any]]) -> float:
    penalty = 0.0
    for row in rows:
        penalty += _SEVERITY_PENALTY.get(str(row.get("severity")), 7.0) * float(row.get("confidence") or 0.5)
    return max(0.0, round(100.0 - min(100.0, penalty), 2))


def _needs_repair(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        severity = row.get("severity")
        conf = float(row.get("confidence") or 0)
        if severity == "critical" and conf >= 0.55:
            return True
        if severity == "major" and conf >= 0.78:
            return True
    return False


def _active_context(targets: list[Segment], segment: Segment) -> dict[str, Any]:
    pos = {row.id: i for i, row in enumerate(targets)}
    i = pos.get(segment.id, 0)
    return {
        "before": [row.text for row in targets[max(0, i - 2):i]],
        "after": [row.text for row in targets[i + 1:i + 3]],
    }


def _specialist_candidates(harness, targets: list[Segment], segment: Segment, current: str, memory: BookMemory, findings: list[dict[str, Any]]) -> list[str]:
    provider = v8._provider(harness)
    if provider is None:
        return []
    codes = {str(row.get("code") or "") for row in findings}
    special = bool(codes & _SPECIAL_CODES)
    payload = {
        "source": segment.text,
        "current_ru": current,
        "issues": findings,
        "context": _active_context(targets, segment),
        "book_blueprint": {
            "style": asdict(memory.style),
            "glossary": memory.glossary,
            "characters": memory.characters,
            "continuity": str(memory.rolling_summary or "")[:3500],
        },
        "fidelity_invariants": _extract_invariants(segment.text),
    }
    system = """You are a surgical EN→RU literary translation editor. Repair ONLY the listed defects; preserve every correct
fact, relation, number, material, joke, image and stylistic choice. Use neighboring source context to resolve pronouns and
relationships. Do not summarize or shorten. Preserve the author's dry voice. Return natural publication-quality Russian.
For idiom/dialogue/referent/voice ambiguity, provide two genuinely different faithful candidates; otherwise provide one.
Return ONLY JSON {"candidates":["...", "optional second..."]}."""
    try:
        obj = v8._complete_json(provider, system, payload)
        values = obj.get("candidates") or []
        if isinstance(values, str):
            values = [values]
        clean = [_norm_text(value) for value in values if _norm_text(value)]
        return clean[:2] if special else clean[:1]
    except Exception as exc:
        print(f"[v9-specialist] id={segment.id} error={type(exc).__name__}", flush=True)
        return []


def _fatal_count(segment: Segment, candidate: str, memory: BookMemory) -> int:
    return sum(1 for row in _deterministic_findings(segment, candidate, memory) if row["severity"] == "critical")


def _judge_candidates(harness, targets, segment, candidates: list[str], memory) -> str:
    candidates = [_norm_text(x) for x in candidates if _norm_text(x)]
    if not candidates:
        return ""
    fatal = [_fatal_count(segment, value, memory) for value in candidates]
    best = min(fatal)
    pool = [value for value, count in zip(candidates, fatal) if count == best]
    if len(pool) == 1:
        return pool[0]
    payload = {
        "source": segment.text,
        "context": _active_context(targets, segment),
        "candidates": {chr(65 + i): value for i, value in enumerate(pool)},
        "book_blueprint": {"style": asdict(memory.style), "characters": memory.characters, "glossary": memory.glossary},
    }
    system = """Choose the best EN→RU literary translation candidate; do not rewrite. Fidelity dominates style. Reject
omissions, wrong referents/relations, corrupted numbers/materials, raw English and broken Russian. Then prefer natural prose,
voice and idiomatic dialogue. Return ONLY JSON {"choice":"A|B"}."""
    try:
        obj = v8._complete_json(harness.gate, system, payload)
        choice = str(obj.get("choice") or "").strip().upper()
        idx = ord(choice) - 65 if len(choice) == 1 else -1
        if 0 <= idx < len(pool):
            return pool[idx]
    except Exception:
        pass
    return pool[0]


def _format_dialogue_v9(segment: Segment, text: str) -> tuple[str, int]:
    original = _norm_text(text)
    value = original.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    if _is_dialogue(segment.text):
        value = re.sub(r"^\s*[\"'«]\s*", "— ", value, count=1)
        value = re.sub(r"[\"'»]\s*$", "", value, count=1)
        value = re.sub(r"([,!?…])\s*[\"'»]\s*,?\s*—\s*", r"\1, — ", value)
        value = re.sub(r"—\s*[\"'«]\s*", "— ", value)
        value = re.sub(r"([.!?…])\s*[\"'»]\s*(?=[А-ЯЁ])", r"\1 — ", value)
    else:
        value = re.sub(r"[\"']([^\"'\n]{1,180}[А-Яа-яЁё][^\"'\n]{0,180})[\"']", r"«\1»", value)

    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()

    def imbalance(s: str) -> int:
        return abs(s.count("«") - s.count("»"))
    if imbalance(value) > imbalance(original):
        value = original
    return value, int(value != original)


def _thread_terms(memory: BookMemory) -> dict[str, str]:
    out = dict(v8._entity_map(memory))
    source_book = "\n".join(segment.text for segment in v6._SOURCE_SEGMENTS)
    for source, target in memory.glossary.items():
        s = str(source or "").strip()
        t = str(target or "").strip()
        if not s or not t or s in out:
            continue
        if re.search(r"(?:ine|ian|ese|ish)$", s, re.I) and t[:1].islower():
            continue
        occurrences = len(re.findall(r"(?<![A-Za-z])" + re.escape(s) + r"(?![A-Za-z])", source_book, re.I))
        if occurrences >= 3:
            out[s] = t
    return out


def _thread_present(candidate: str, target: str) -> bool:
    if target[:1].isupper():
        return v8._entity_present(candidate, target)
    low = candidate.casefold()
    token = re.sub(r"[^а-яё]", "", target.casefold())
    if not token:
        return target.casefold() in low
    stem = token[: max(5, len(token) - 3)] if len(token) >= 6 else token
    return stem in re.sub(r"[^а-яё ]", "", low)


def _thread_findings(targets: list[Segment], translations: dict[str, str], memory: BookMemory) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terms = _thread_terms(memory)
    rows = []
    index: dict[str, Any] = {}
    for source, target in terms.items():
        ids = [segment.id for segment in targets if v8._source_mentions(segment.text, source)]
        if not ids:
            continue
        mismatches = [sid for sid in ids if not _thread_present(translations.get(sid, ""), target)]
        index[source] = {"preferred_ru": target, "occurrences": len(ids), "mismatches": mismatches[:30]}
        for sid in mismatches:
            rows.append({
                "id": sid, "severity": "major", "confidence": 0.9,
                "code": "thread", "source_span": source, "target_span": "",
                "reason": f"recurring book thread/term drift: {source} should stay consistent with {target}",
                "repairability": "local",
            })
    return rows, index


def _quality_v9(harness, targets, translated, memory) -> dict[str, Any]:
    global _V9_STATS, _V9_SCORES, _V9_FINAL_FINDINGS, _V9_THREAD_MISMATCHES
    started = time.perf_counter()
    hybrid.progress({"phase": "v9_quality_audit", "audited": len(targets), "strategy": "continuous-QE+invariants+specialists"})

    llm_rows = _parallel_audit(harness, targets, translated, memory)
    det_rows = [row for segment in targets for row in _deterministic_findings(segment, translated.get(segment.id, ""), memory)]
    thread_rows, thread_index = _thread_findings(targets, translated, memory)
    finding_map = _merge_findings(llm_rows, det_rows, thread_rows)
    score_map = {segment.id: _score(finding_map.get(segment.id, [])) for segment in targets}

    repair_ids = [segment.id for segment in targets if _needs_repair(finding_map.get(segment.id, []))]
    repair_ids.sort(key=lambda sid: score_map.get(sid, 100.0))
    cap = max(0, int(os.getenv("BOOKAI_V9_REPAIR_MAX") or "42"))
    repair_ids = repair_ids[:cap]
    by_id = {segment.id: segment for segment in targets}
    workers = max(1, min(6, int(os.getenv("BOOKAI_V9_REPAIR_WORKERS") or "5")))

    results: dict[str, str] = {}
    ambiguity_calls = 0

    def repair_one(sid: str):
        segment = by_id[sid]
        current = str(translated.get(sid) or "")
        rows = finding_map.get(sid, [])
        candidates = _specialist_candidates(harness, targets, segment, current, memory, rows)
        if not candidates:
            return sid, "", 0
        chosen = _judge_candidates(harness, targets, segment, [current, *candidates], memory)
        special = int(any(str(row.get("code")) in _SPECIAL_CODES for row in rows) and len(candidates) > 1)
        return sid, chosen, special

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9-repair") as pool:
        futures = [pool.submit(repair_one, sid) for sid in repair_ids]
        for future in as_completed(futures):
            sid, value, special = future.result()
            ambiguity_calls += special
            if value and value.strip() != str(translated.get(sid) or "").strip():
                results[sid] = value

    candidate_map = dict(translated)
    candidate_map.update(results)
    changed_segments = [by_id[sid] for sid in results]
    re_llm = _parallel_audit(harness, targets, candidate_map, memory, selected=changed_segments) if changed_segments else []
    re_det = [row for segment in changed_segments for row in _deterministic_findings(segment, candidate_map.get(segment.id, ""), memory)]
    re_thread, _ = _thread_findings(changed_segments, candidate_map, memory) if changed_segments else ([], {})
    re_map = _merge_findings(re_llm, re_det, re_thread)

    accepted = 0
    for sid, value in results.items():
        old_rows = finding_map.get(sid, [])
        new_rows = re_map.get(sid, [])
        old_score = score_map.get(sid, 100.0)
        new_score = _score(new_rows)
        old_fatal = sum(row["severity"] == "critical" and float(row.get("confidence") or 0) >= 0.55 for row in old_rows)
        new_fatal = sum(row["severity"] == "critical" and float(row.get("confidence") or 0) >= 0.55 for row in new_rows)
        if new_fatal < old_fatal or (new_fatal == 0 and new_score >= old_score + 3.5):
            translated[sid] = value
            finding_map[sid] = new_rows
            score_map[sid] = new_score
            accepted += 1

    formatted = 0
    entity_fixes = 0
    for segment in targets:
        current = str(translated.get(segment.id) or "")
        fixed, count = v8._fix_near_entity_typos(segment, current, memory)
        entity_fixes += count
        fixed, changed = _format_dialogue_v9(segment, fixed)
        formatted += changed
        translated[segment.id] = fixed

    final_det = [row for segment in targets for row in _deterministic_findings(segment, translated.get(segment.id, ""), memory)]
    final_thread, thread_index = _thread_findings(targets, translated, memory)
    final_det_map = _merge_findings(final_det, final_thread)
    for sid, rows in final_det_map.items():
        existing = [row for row in finding_map.get(sid, []) if row.get("severity") in {"critical", "major"}]
        seen = {(x.get("code"), x.get("reason")) for x in existing}
        finding_map[sid] = existing + [row for row in rows if (row.get("code"), row.get("reason")) not in seen]
        score_map[sid] = min(score_map.get(sid, 100.0), _score(finding_map[sid]))

    unresolved_critical = {
        sid for sid, rows in finding_map.items()
        if any(row["severity"] == "critical" and float(row.get("confidence") or 0) >= 0.55 for row in rows)
    }
    unresolved_major = {
        sid for sid, rows in finding_map.items()
        if sid not in unresolved_critical and any(row["severity"] == "major" and float(row.get("confidence") or 0) >= 0.88 for row in rows)
    }
    _V9_THREAD_MISMATCHES = {sid for sid, rows in final_det_map.items() if any(row.get("code") == "thread" for row in rows)}
    _V9_SCORES = score_map
    _V9_FINAL_FINDINGS = finding_map

    v6._V6_REMAINING_HARD = set(unresolved_critical)
    v6._V6_REMAINING_MEDIUM = set(unresolved_major)
    v6._V6_REASONS = {
        sid: "; ".join(row["reason"] for row in finding_map.get(sid, []) if row["severity"] in {"critical", "major"})
        for sid in unresolved_critical | unresolved_major
    }

    counts = Counter(row["severity"] for rows in finding_map.values() for row in rows)
    _V9_STATS = {
        "audited": len(targets),
        "initial_llm_findings": len(llm_rows),
        "initial_deterministic_findings": len(det_rows),
        "thread_findings": len(thread_rows),
        "repair_selected": len(repair_ids),
        "repair_candidates": len(results),
        "repair_accepted": accepted,
        "ambiguity_candidate_cases": ambiguity_calls,
        "entity_typo_fixes": entity_fixes,
        "dialogue_segments_formatted": formatted,
        "remaining_critical": len(unresolved_critical),
        "remaining_major_high_confidence": len(unresolved_major),
        "mean_quality_score": round(sum(score_map.values()) / max(1, len(score_map)), 2),
        "severity_counts": dict(counts),
        "qe_elapsed_seconds": round(time.perf_counter() - started, 2),
        "thread_terms": len(thread_index),
    }
    hybrid.progress({"phase": "v9_quality_done", **_V9_STATS})
    return dict(_V9_STATS)


def _finalize_v9(harness, targets, translated, memory, state, *, chapter_digests, book_synopsis, context_index):
    _, thread_index = _thread_findings(targets, translated, memory)
    state["v9_thread_index_current"] = thread_index
    global_index = dict(state.get("v9_thread_index") or {})
    for source, row in thread_index.items():
        prev = dict(global_index.get(source) or {})
        global_index[source] = {
            "preferred_ru": row.get("preferred_ru"),
            "occurrences": int(prev.get("occurrences") or 0) + int(row.get("occurrences") or 0),
            "mismatches": (list(prev.get("mismatches") or []) + list(row.get("mismatches") or []))[-80:],
        }
    state["v9_thread_index"] = global_index

    working_add = []
    trusted_add = []
    for segment in targets:
        value = _norm_text(translated.get(segment.id) or "")
        if not value:
            continue
        score = float(_V9_SCORES.get(segment.id, 100.0))
        rows = _V9_FINAL_FINDINGS.get(segment.id, [])
        critical = any(row["severity"] == "critical" and float(row.get("confidence") or 0) >= 0.55 for row in rows)
        major = any(row["severity"] == "major" and float(row.get("confidence") or 0) >= 0.78 for row in rows)
        raw = bool(_ASCII_WORD.search(value))
        base = {
            "source": segment.text,
            "translation": value,
            "chapter": segment.chapter,
            "quality": round(score / 100.0, 4),
        }
        if score >= 78 and not critical and not raw:
            working_add.append(base)
        if score >= 92 and not critical and not major and not raw and segment.id not in _V9_THREAD_MISMATCHES:
            trusted_add.append(base)

    def merge(existing, additions, cap):
        by_source: dict[str, dict[str, Any]] = {}
        for row in [*list(existing or []), *additions]:
            if not isinstance(row, dict):
                continue
            source = _norm_text(row.get("source") or "")
            if not source:
                continue
            old = by_source.get(source)
            if old is None or float(row.get("quality") or 0) >= float(old.get("quality") or 0):
                by_source[source] = row
        return list(by_source.values())[-cap:]

    working = merge(state.get("v9_working_tm"), working_add, 6000)
    trusted = merge(state.get("v9_trusted_tm"), trusted_add, 5000)
    state["v9_working_tm"] = working
    state["v9_trusted_tm"] = trusted
    state["v6_translation_memory"] = working[-5000:]
    state["v9_tm_stats"] = {
        "working_added": len(working_add), "working_total": len(working),
        "trusted_added": len(trusted_add), "trusted_total": len(trusted),
    }
    GigaChatLightningV9Backend.set_memories(working, trusted)
    hybrid.progress({"phase": "v9_finalize", **state["v9_tm_stats"], "thread_terms": len(global_index)})
    return translated


def _final_issues_v9(segments, translations, memory=None, *, source_segments=None):
    memory = memory or BookMemory()
    issues: list[QualityIssueV3] = []
    try:
        legacy = enhanced_batch_issues(segments, translations, memory, source_segments=source_segments)
        keep_codes = {"empty", "too_short", "question_loss", "semantic_omission", "unexpected_script", "english_leftover", "service_leak", "repetition_loop"}
        for issue in legacy:
            if issue.severity == "hard" and issue.code in keep_codes:
                issues.append(issue)
    except Exception:
        pass
    existing = {(issue.id, issue.code) for issue in issues}
    valid = {segment.id for segment in segments}
    for sid in sorted(v6._V6_REMAINING_HARD):
        if sid in valid and (sid, "v9_qe_unresolved") not in existing:
            issues.append(QualityIssueV3(sid, "hard", "v9_qe_unresolved", v6._V6_REASONS.get(sid, "high-confidence v9 critical defect remains")))
    return issues


def _configure_v9() -> None:
    v8c._configure_v8c()
    slug = re.sub(r"[^a-z0-9]+", "-", str(v3.CHAPTER_NAME or "chapter").casefold()).strip("-") or "chapter"
    v3.GigaChatLightningV3Backend = GigaChatLightningV9Backend
    v3._analysis_task = _analysis_task_v9
    hybrid._deepseek_final_recovery = _reliability_recovery
    v3._semantic_short_repair = _quality_v9
    hybrid._selective_literary_refinement = _finalize_v9
    v3.enhanced_batch_issues = _final_issues_v9
    v8._normalize_dialogue_v8 = _format_dialogue_v9

    v3.OUTPUT = Path(f"Devices_and_Desires_RU_{slug}_EVAL_V9.fb2")
    v3.PROGRESS = Path(f"chapter-v9-{slug}-progress.json")
    v3.PROBE = Path(f"chapter-v9-{slug}-probe.json")
    v3.ROUTING = Path(f"chapter-v9-{slug}-routing.json")
    v3.REPORT = Path(f"chapter-v9-{slug}.json")
    v3.SOURCE_TXT = Path(f"chapter-v9-{slug}-source.txt")
    v3.TRANSLATED_TXT = Path(f"chapter-v9-{slug}-translated.txt")
    v3.MAP_JSON = Path(f"chapter-v9-{slug}-translation-map.json")


def _annotate_v9() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        **dict(report.get("architecture") or {}),
        "version": "quality-v9-production-memory-fidelity-routing",
        "gold_reference_available_to_pipeline": False,
        "book_structure": "top-level chapter spans from parsed book structure/groups",
        "book_blueprint": "persistent source-only blueprint reused across chapters",
        "translation_memory": "separate trusted + working weighted self-TM",
        "qe": "continuous confidence-aware bilingual QE with source neighbors",
        "fidelity": "deterministic coverage/raw-English/material/number/entity invariants",
        "repair": "high-confidence specialist routing only; tiny ambiguity candidate path",
        "threadweaver": "persistent recurring entity/technical-term Thread Index",
        "formatter": "deterministic safety-checked Russian typography",
        "reliability": "unbounded-to-chapter pending segment fallback from GigaChat to DeepSeek",
    }
    report["v9_stats"] = dict(_V9_STATS)
    report["v9_reliability"] = dict(_V9_RELIABILITY)
    report["v9_book_blueprint"] = {
        "glossary_entries": len((_V9_BLUEPRINT or {}).get("glossary") or {}),
        "characters": len((_V9_BLUEPRINT or {}).get("characters") or {}),
    }
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v9()
    try:
        v3.main()
    finally:
        v6._annotate_report()
        _annotate_v9()


if __name__ == "__main__":
    main()
