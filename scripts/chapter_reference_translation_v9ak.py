from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9aj as v9aj

v9ag = v9ah.v9ag
v9af = v9ag.v9af
v9ad = v9ag.v9ad
v9ac = v9ad.v9ac
v9ab = v9ag.v9ab
v9aa = v9ab.v9aa
v9 = v9ab.v9
v8 = v9ab.v8
v6 = v9ab.v6
v3 = v9ab.v3
v9s = v9af.v9s

# Freeze the v9ah implementations before patching.
_BASE_AH_ROUTE = v9ah._BASE_AB_ROUTE
_BASE_AH_QUALITY = v9ah._BASE_QUALITY

_V9AK_STATS: dict[str, Any] = {}
_NAME_STATS: dict[str, Any] = {}
_ROUTER_STATS: dict[str, Any] = {}
_POSTCONDITION_STATS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 1) Fast GigaChat recovery: retry a degenerate strict-schema response once as
# plain JSON BEFORE recursively splitting 12 -> 6 -> 3 -> 1.
# ---------------------------------------------------------------------------

class GigaChatLightningV9AKBackend(v9.GigaChatLightningV9Backend):
    name = "gigachat-3-lightning-v9ak-fast-degenerate-recovery"

    def __init__(self) -> None:
        super().__init__()
        self._v9ak_internal_calls = 0
        self._v9ak_degenerate_retries = 0

    @staticmethod
    def _degenerate_response(response, batch) -> bool:
        try:
            content = str(response.choices[0].message.content or "").strip()
        except Exception:
            return True
        usage = v9.GigaChatLightningV9Backend._usage(response)
        completion = int(usage.get("completion_tokens") or 0)
        if completion <= 2:
            return True
        if len(content) < max(8, 4 * len(batch)):
            return True
        return False

    @staticmethod
    def _merge_usage(total: dict[str, int], usage: dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)

    def _plain_retry_payload(self, messages: list[dict[str, str]], payload: dict) -> dict:
        fallback = dict(payload)
        fallback.pop("response_format", None)
        fallback["messages"] = [dict(messages[0]), dict(messages[1])]
        fallback["messages"][0]["content"] += (
            " Верни только валидный JSON-объект с РОВНО всеми требуемыми ID. "
            "Предыдущий structured-output ответ был пустым/повреждённым; не сокращай ответ."
        )
        return fallback

    def _translate_batch(
        self,
        batch,
        memory,
        *,
        source_segments=None,
        minimal: bool = False,
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты литературный переводчик EN→RU. Переводи только TARGETS, "
                    "никогда не переводя CONTEXT_ONLY. Не добавляй ничего от себя."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(
                    batch,
                    memory,
                    source_segments=source_segments,
                    minimal=minimal,
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "response_format": self._strict_response_format(batch),
        }
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._v9ak_internal_calls = 0
        plain_used = False

        def do_call(request: dict, *, strict: bool):
            self._v9ak_internal_calls += 1
            response = self._chat(request, batch, strict=strict)
            self._merge_usage(total_usage, self._usage(response))
            return response

        try:
            response = do_call(payload, strict=True)
        except Exception as exc:
            if not self._schema_incompatibility(exc):
                raise
            print(
                "[v9ak-giga-recovery] strict_schema_rejected=true action=plain_json_same_batch",
                flush=True,
            )
            plain_used = True
            response = do_call(self._plain_retry_payload(messages, payload), strict=False)

        if self._degenerate_response(response, batch):
            self._v9ak_degenerate_retries += 1
            print(
                f"[v9ak-giga-recovery] degenerate=true segments={len(batch)} "
                f"action={'parse_then_split' if plain_used else 'plain_json_same_batch'}",
                flush=True,
            )
            if not plain_used:
                plain_used = True
                response = do_call(self._plain_retry_payload(messages, payload), strict=False)

        parsed = self._parse_json_object(str(response.choices[0].message.content or ""))
        expected = {segment.id for segment in batch}
        translated = {sid: text for sid, text in parsed.items() if sid in expected and text}

        # If structured output returned only a small fraction of a non-trivial
        # batch, try the WHOLE batch once in plain JSON before recursive splitting.
        missing = expected - set(translated)
        if (
            missing
            and not plain_used
            and len(batch) > 1
            and len(missing) >= max(2, math.ceil(len(batch) * 0.4))
        ):
            self._v9ak_degenerate_retries += 1
            print(
                f"[v9ak-giga-recovery] partial={len(translated)}/{len(batch)} "
                "action=plain_json_same_batch",
                flush=True,
            )
            response2 = do_call(self._plain_retry_payload(messages, payload), strict=False)
            parsed2 = self._parse_json_object(str(response2.choices[0].message.content or ""))
            translated2 = {sid: text for sid, text in parsed2.items() if sid in expected and text}
            if len(translated2) >= len(translated):
                translated = translated2

        return translated, total_usage

    def _add_attempt_usage(self, usage: dict[str, int]) -> None:
        self.usage.add(usage, calls=max(1, int(self._v9ak_internal_calls or 1)))

    def _single_strict_recovery(self, segment, memory, source_segments, original_error=None):
        try:
            rows, usage = self._translate_batch(
                [segment], memory, source_segments=source_segments, minimal=True
            )
            self._add_attempt_usage(usage)
            if segment.id not in rows:
                raise ValueError("strict single response omitted target id")
            return rows, {}
        except Exception as exc:
            self.usage.add({}, calls=max(1, int(self._v9ak_internal_calls or 1)))
            prefix = f"{type(original_error).__name__}: {original_error}; " if original_error else ""
            return {}, {segment.id: prefix + f"single recovery {type(exc).__name__}: {exc}"}

    def _translate_resilient(self, batch, memory, *, source_segments, depth: int = 0):
        try:
            rows, usage = self._translate_batch(batch, memory, source_segments=source_segments)
            self._add_attempt_usage(usage)
        except Exception as exc:
            self.usage.add({}, calls=max(1, int(self._v9ak_internal_calls or 1)))
            if len(batch) == 1:
                return self._single_strict_recovery(batch[0], memory, source_segments, exc)
            if depth < self.max_split_depth:
                mid = max(1, len(batch) // 2)
                left_rows, left_errors = self._translate_resilient(
                    batch[:mid], memory, source_segments=source_segments, depth=depth + 1
                )
                right_rows, right_errors = self._translate_resilient(
                    batch[mid:], memory, source_segments=source_segments, depth=depth + 1
                )
                return {**left_rows, **right_rows}, {**left_errors, **right_errors}
            return {}, {segment.id: f"{type(exc).__name__}: {exc}" for segment in batch}

        missing = [segment for segment in batch if segment.id not in rows]
        if not missing:
            return rows, {}
        if len(batch) == 1:
            retry_rows, retry_errors = self._single_strict_recovery(
                batch[0], memory, source_segments
            )
            rows.update(retry_rows)
            return rows, retry_errors
        if depth < self.max_split_depth:
            retry_rows, retry_errors = self._translate_resilient(
                missing, memory, source_segments=source_segments, depth=depth + 1
            )
            rows.update(retry_rows)
            return rows, retry_errors
        return rows, {segment.id: "missing from GigaChat batch response" for segment in missing}


# ---------------------------------------------------------------------------
# 2) General semantic-risk router. These are CATEGORY detectors, not benchmark
# phrase patches. They make the sparse DeepSeek specialist book-agnostic.
# ---------------------------------------------------------------------------

_NUMBER_REL_RE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?%?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|dozen|hundred|thousand|million|half|quarter|third|twice|double|"
    r"triple|percent|per cent|times|more than|less than|fewer than|as many as|"
    r"outnumber(?:ed|ing)?)\b",
    re.I,
)
_DEICTIC_TIME_RE = re.compile(
    r"\b(?:today|tomorrow|yesterday|tonight|this morning|this afternoon|this evening|"
    r"tomorrow morning|tomorrow afternoon|tomorrow night|last night|next morning)\b",
    re.I,
)
_TIME_REL_RE = re.compile(
    r"\b(?:morning|afternoon|evening|night|noon|midnight|earlier|later|ago|before|"
    r"after|until|during|while|meanwhile|then|by the time)\b",
    re.I,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|none|neither|nor|without|hardly|scarcely|barely|unless|"
    r"cannot|can't|won't|wouldn't|didn't|doesn't|isn't|aren't|wasn't|weren't|"
    r"shouldn't|couldn't|mustn't)\b",
    re.I,
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|thus|hence|so that|as a result|owing to|due to|"
    r"since|unless|if|although|though|whereas|consequently)\b",
    re.I,
)
_STRONG_REF_RE = re.compile(
    r"\b(?:either|neither|both|former|latter|the other|one of them|each of them|"
    r"all of them|one another|each other|who|whom|whose|which)\b",
    re.I,
)
_PRONOUN_START_RE = re.compile(
    r"^\s*(?:['\"“‘—-]\s*)?(?:he|she|they|it|this|that|these|those|his|her|their)\b",
    re.I,
)
_PASSIVE_AGENT_RE = re.compile(
    r"\b(?:was|were|is|are|been|being)\s+(?:\w+\s+){0,3}\w+(?:ed|en)\b[^.!?]{0,100}\bby\b",
    re.I,
)
_MODAL_RE = re.compile(
    r"\b(?:must|should|could|would|might|may|can|have to|had to|ought to|need to)\b",
    re.I,
)


def _merge_route(
    ranked_by_id: dict[str, dict[str, Any]],
    *,
    segment,
    index: int,
    code: str,
    priority: int,
    reason: str,
) -> None:
    current = dict(ranked_by_id.get(segment.id) or {})
    if not current:
        current = {
            "id": segment.id,
            "index": index,
            "code": code,
            "priority": priority,
            "reason": reason,
            "glossary_hits": [],
        }
    else:
        old_priority = int(current.get("priority") or 0)
        current["priority"] = max(old_priority, priority)
        current["reason"] = (
            str(current.get("reason") or "").strip() + "; " + reason
        ).strip("; ")
        if priority > old_priority:
            current["code"] = code
    generic = list(current.get("generic_codes") or [])
    if code not in generic:
        generic.append(code)
    current["generic_codes"] = generic
    ranked_by_id[segment.id] = current


def _generic_routes(targets, translated, memory) -> list[dict[str, Any]]:
    ranked: dict[str, dict[str, Any]] = {}
    glossary = dict(getattr(memory, "glossary", {}) or {})
    for i, segment in enumerate(targets):
        source = str(segment.text or "")
        low = source.casefold()
        if not source.strip():
            continue

        if _NUMBER_REL_RE.search(source):
            priority = 22 if re.search(
                r"\d|percent|per cent|twice|double|triple|more than|less than|fewer than|outnumber",
                source,
                re.I,
            ) else 20
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="quantitative_relation",
                priority=priority,
                reason="generic quantity/ratio/comparison cue: preserve every number and relation direction",
            )

        if _DEICTIC_TIME_RE.search(source):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="temporal_relation",
                priority=22,
                reason="explicit deictic time cue: preserve day/part-of-day and chronology exactly",
            )
        elif _TIME_REL_RE.search(source):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="temporal_relation",
                priority=19,
                reason="temporal ordering cue: verify before/after/later/earlier chronology",
            )

        negations = _NEGATION_RE.findall(source)
        if negations:
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="polarity_modality",
                priority=22 if len(negations) >= 2 else 20,
                reason="explicit negation/polarity cue: preserve scope and do not invert proposition",
            )

        if _CAUSAL_RE.search(source):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="causal_relation",
                priority=20,
                reason="causal/conditional/concessive connector: preserve which proposition causes/conditions which",
            )

        if _PASSIVE_AGENT_RE.search(source):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="participant_roles",
                priority=22,
                reason="passive construction with explicit agent: verify actor/action/object roles",
            )
        elif _MODAL_RE.search(source) and re.search(
            r"\b(?:he|she|they|I|we|you)\b", source, re.I
        ):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="participant_roles",
                priority=18,
                reason="modal participant relation: verify who is obliged/able/allowed to act",
            )

        if _STRONG_REF_RE.search(source):
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="reference_resolution",
                priority=22,
                reason="explicit anaphora/relative reference: resolve antecedent set from neighboring source context",
            )
        elif _PRONOUN_START_RE.search(source) and len(source) >= 80:
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="reference_resolution",
                priority=18,
                reason="context-dependent pronoun at segment start: verify antecedent and grammatical role",
            )

        hits = []
        for src in glossary:
            key = str(src or "").strip()
            if len(key) >= 4 and key.casefold() in low:
                hits.append(key)
                if len(hits) >= 4:
                    break
        if hits:
            _merge_route(
                ranked,
                segment=segment,
                index=i,
                code="terminology",
                priority=19 + min(2, len(hits)),
                reason="source contains book-wide terminology candidates: " + ", ".join(hits),
            )

    return list(ranked.values())


def _route_v9ak(targets, translated, memory):
    _, ranked0 = _BASE_AH_ROUTE(targets, translated, memory)
    ranked_by_id = {str(row.get("id") or ""): dict(row) for row in ranked0}

    duplicate_rows = v9aj._cross_segment_duplicate_routes(targets, translated)
    for row in duplicate_rows:
        sid = str(row.get("id") or "")
        if not sid:
            continue
        current = ranked_by_id.get(sid)
        if current is None:
            ranked_by_id[sid] = dict(row)
        else:
            current = dict(current)
            current["priority"] = max(
                int(current.get("priority") or 0), int(row.get("priority") or 24)
            )
            current["reason"] = (
                str(current.get("reason") or "") + "; " + str(row.get("reason") or "")
            ).strip("; ")
            current["code"] = "cross_segment_duplicate"
            ranked_by_id[sid] = current

    generic = _generic_routes(targets, translated, memory)
    for row in generic:
        sid = str(row.get("id") or "")
        if not sid:
            continue
        current = ranked_by_id.get(sid)
        if current is None:
            ranked_by_id[sid] = dict(row)
            continue
        current = dict(current)
        old_priority = int(current.get("priority") or 0)
        new_priority = int(row.get("priority") or 0)
        current["priority"] = max(old_priority, new_priority)
        current["reason"] = (
            str(current.get("reason") or "") + "; " + str(row.get("reason") or "")
        ).strip("; ")
        if new_priority > old_priority:
            current["code"] = row.get("code")
        codes = list(current.get("generic_codes") or [])
        for code in list(row.get("generic_codes") or [row.get("code")]):
            if code and code not in codes:
                codes.append(code)
        current["generic_codes"] = codes
        ranked_by_id[sid] = current

    ranked = sorted(
        (row for sid, row in ranked_by_id.items() if sid),
        key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)),
    )
    by_code = Counter(
        code
        for row in ranked
        for code in (row.get("generic_codes") or [])
        if code
    )
    _ROUTER_STATS.clear()
    _ROUTER_STATS.update(
        {
            "ranked_total": len(ranked),
            "generic_candidates": len(generic),
            "duplicate_alignment_candidates": len(duplicate_rows),
            "generic_by_code": dict(by_code),
        }
    )
    print("[v9ak-router] " + json.dumps(_ROUTER_STATS, ensure_ascii=False), flush=True)
    return [], ranked


# ---------------------------------------------------------------------------
# 3) Rich whole-book name registry. GigaChat proposes; one cached DeepSeek batch
# reviews only important uncertain names once for the whole book.
# ---------------------------------------------------------------------------

def _registry_path() -> Path:
    root = Path(os.getenv("BOOKAI_V8_SHARED_CACHE") or ".bookai-cache-v9ak")
    root.mkdir(parents=True, exist_ok=True)
    return root / "source-name-registry-v9ak.json"


def _load_registry() -> dict[str, Any]:
    path = _registry_path()
    try:
        obj = json.loads(path.read_text("utf-8")) if path.exists() else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_registry(data: dict[str, Any]) -> None:
    _registry_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def _registry_canon(data: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in list(data.get("names") or []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        ru = v9._norm_text(row.get("ru") or "")
        src = str(row.get("source") or "").strip()
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        if src and ru and status in {"stable", "reviewed"} and confidence >= 0.72:
            out[src] = ru
    return out


def _learn_name_registry_v9ak(memory) -> dict[str, str]:
    data = _load_registry()
    if data.get("names"):
        canon = _registry_canon(data)
        _NAME_STATS.update(
            {
                "cache_hit": True,
                "entries": len(data.get("names") or []),
                "stable_or_reviewed": len(canon),
                "uncertain": sum(
                    1
                    for x in data.get("names") or []
                    if isinstance(x, dict) and x.get("status") == "uncertain"
                ),
                "deepseek_review_complete": bool(data.get("deepseek_review_complete")),
            }
        )
        return canon

    candidates = v9aa._whole_book_name_items()
    if not candidates:
        return {}

    max_names = max(30, int(os.getenv("BOOKAI_V9AK_NAME_MAX") or "70"))
    batch_size = max(10, int(os.getenv("BOOKAI_V9AK_NAME_BATCH") or "20"))
    candidates = candidates[:max_names]
    system = """Build a SOURCE-ONLY whole-book Russian name registry for an English novel.
No Russian reference translation exists. For every genuine recurring proper name return:
source, one dictionary-form Cyrillic spelling, confidence 0..1, and status stable|uncertain.
Use supplied occurrence counts and several English contexts. Prefer information-preserving transliteration for
invented names: do not silently delete visible vowel sequences or consonant clusters. Omit ordinary words/titles.
Mark uncertain when pronunciation/transliteration is genuinely ambiguous instead of pretending certainty.
ONLY JSON {"names":[{"source":"...","ru":"...","confidence":0.0,"status":"stable|uncertain","reason":"..."}]}.
"""
    gathered: dict[str, dict[str, Any]] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        try:
            obj = v9ab._giga_json(system, {"candidates": batch}, max_tokens=4200)
            raw = obj.get("names") or []
        except Exception as exc:
            print(
                f"[v9ak-name-registry] giga_batch={start // batch_size + 1} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            raw = []
        allowed = {x["source"]: x for x in batch}
        for row in raw if isinstance(raw, list) else []:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source") or "").strip()
            ru = v9._norm_text(row.get("ru") or "")
            if src not in allowed or not ru or not re.search(r"[А-Яа-яЁё]", ru):
                continue
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence") or 0)))
            except Exception:
                confidence = 0.0
            status = str(row.get("status") or "").strip().lower()
            if status not in {"stable", "uncertain"}:
                status = "stable" if confidence >= 0.86 else "uncertain"
            if confidence < 0.82:
                status = "uncertain"
            source_row = allowed[src]
            gathered[src] = {
                "source": src,
                "ru": ru,
                "confidence": round(confidence, 4),
                "status": status,
                "occurrences": int(source_row.get("occurrences") or 0),
                "contexts": list(source_row.get("source_contexts") or [])[:4],
                "reason": v9._norm_text(row.get("reason") or "")[:400],
                "review_model": "gigachat",
            }

    data = {
        "version": 1,
        "source_only": True,
        "deepseek_review_complete": False,
        "deepseek_review_attempts": 0,
        "names": sorted(
            gathered.values(),
            key=lambda x: (-int(x.get("occurrences") or 0), x["source"]),
        ),
    }
    _save_registry(data)
    canon = _registry_canon(data)
    _NAME_STATS.update(
        {
            "cache_hit": False,
            "entries": len(data["names"]),
            "stable_or_reviewed": len(canon),
            "uncertain": sum(1 for x in data["names"] if x.get("status") == "uncertain"),
            "deepseek_review_complete": False,
        }
    )
    print("[v9ak-name-registry] " + json.dumps(_NAME_STATS, ensure_ascii=False), flush=True)
    return canon


def _apply_registry_to_memory(memory, data: dict[str, Any]) -> int:
    canon = _registry_canon(data)
    for src, ru in canon.items():
        memory.glossary[src] = ru
        desc = str(memory.characters.get(src) or "")
        if desc:
            if re.search(r"\bru=[^;]+", desc):
                memory.characters[src] = re.sub(r"\bru=[^;]+", f"ru={ru}", desc, count=1)
            else:
                memory.characters[src] = f"ru={ru};" + desc
        else:
            memory.characters[src] = f"ru={ru};role=proper_name"
    return len(canon)


def _resolve_uncertain_names_once(harness, targets, translated, memory) -> int:
    data = _load_registry()
    if not data.get("names"):
        return 0
    _apply_registry_to_memory(memory, data)
    if data.get("deepseek_review_complete"):
        return 0

    uncertain = [
        row
        for row in data["names"]
        if isinstance(row, dict)
        and row.get("status") == "uncertain"
        and int(row.get("occurrences") or 0) >= 3
    ]
    uncertain.sort(
        key=lambda x: (-int(x.get("occurrences") or 0), float(x.get("confidence") or 0))
    )
    cap = max(4, min(24, int(os.getenv("BOOKAI_V9AK_NAME_REVIEW_MAX") or "16")))
    selected = uncertain[:cap]
    success = not selected

    if selected:
        system = """You are a one-time SOURCE-ONLY proper-name reviewer for an EN→RU novel pipeline.
Review ONLY genuinely uncertain recurring fictional names using source spelling, frequency and English contexts.
No Russian reference exists. Choose one stable dictionary-form Cyrillic spelling that preserves visible source
information and normal Russian orthography. Do not infer a famous/reference spelling. Return every supplied source.
ONLY JSON {"names":[{"source":"...","ru":"...","confidence":0.0,"reason":"..."}]}.
"""
        try:
            obj = v8._complete_json(harness.gate, system, {"names": selected})
            raw = obj.get("names") or []
            success = isinstance(raw, list)
        except Exception as exc:
            print(f"[v9ak-name-review] error={type(exc).__name__}", flush=True)
            raw = []
            success = False
        reviewed = {
            str(row.get("source") or ""): row for row in raw if isinstance(row, dict)
        }
        for entry in data["names"]:
            if not isinstance(entry, dict):
                continue
            src = str(entry.get("source") or "")
            row = reviewed.get(src)
            if not row:
                continue
            ru = v9._norm_text(row.get("ru") or "")
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence") or 0)))
            except Exception:
                confidence = 0.0
            if ru and re.search(r"[А-Яа-яЁё]", ru) and confidence >= 0.70:
                entry["ru"] = ru
                entry["confidence"] = round(confidence, 4)
                entry["status"] = "reviewed"
                entry["reason"] = v9._norm_text(row.get("reason") or "")[:400]
                entry["review_model"] = "deepseek-flash"

    attempts = int(data.get("deepseek_review_attempts") or 0) + (1 if selected else 0)
    data["deepseek_review_attempts"] = attempts
    data["deepseek_review_complete"] = bool(success or not selected or attempts >= 2)
    data["deepseek_reviewed_count"] = len(selected) if success else 0
    _save_registry(data)
    _apply_registry_to_memory(memory, data)
    fixes = v9ab._apply_name_canon(targets, translated, memory)
    _NAME_STATS.update(
        {
            "deepseek_review_complete": bool(data["deepseek_review_complete"]),
            "deepseek_review_attempts": attempts,
            "deepseek_review_requested": len(selected),
            "stable_or_reviewed": len(_registry_canon(data)),
            "entity_fixes_after_review": fixes,
        }
    )
    print("[v9ak-name-review] " + json.dumps(_NAME_STATS, ensure_ascii=False), flush=True)
    return 1 if selected else 0


# ---------------------------------------------------------------------------
# 4) Hard final postcondition. Giga gets one fresh-from-source regeneration; any
# remaining objective defect goes to one sparse DeepSeek batch. Raw Latin prose
# is fail-closed: it can never silently ship.
# ---------------------------------------------------------------------------

def _objective_issues(targets, translated):
    return v9ag._objective_issues(targets, translated)


def _fresh_giga_retranslate(targets, translated, issues) -> tuple[list[str], int]:
    if not issues:
        return [], 0
    cap = max(4, int(os.getenv("BOOKAI_V9AK_FRESH_GIGA_MAX") or "18"))
    batch_size = max(4, int(os.getenv("BOOKAI_V9AK_FRESH_GIGA_BATCH") or "9"))
    issues = issues[:cap]
    changed: list[str] = []
    calls = 0
    system = """Fresh EN→RU retranslation from SOURCE. Ignore the previous Russian wording except for neighboring
context: the previous attempt failed an objective postcondition. Translate each supplied SOURCE completely from
scratch into natural literary Russian. Preserve participants, propositions, numbers, polarity, chronology, causality,
dialogue intent and established Cyrillic names. ZERO ordinary Latin-script prose may remain. If issue_codes contain a
time marker, preserve it exactly. If legal_function is present, preserve the actual prosecuting/defending function.
Return exactly one row per id. ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}.
"""
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        payload = []
        for row in batch:
            i = int(row["index"])
            seg = targets[i]
            payload.append(
                {
                    "id": seg.id,
                    "issue_codes": row.get("codes") or [],
                    "source": str(seg.text or ""),
                    "before_en": [x.text for x in targets[max(0, i - 2):i]],
                    "after_en": [x.text for x in targets[i + 1:i + 3]],
                }
            )
        try:
            obj = v9ab._giga_json(system, {"items": payload}, max_tokens=6200)
            raw = obj.get("items") or []
            calls += 1
        except Exception as exc:
            print(f"[v9ak-fresh-giga] error={type(exc).__name__}", flush=True)
            continue
        parsed = {str(x.get("id") or ""): x for x in raw if isinstance(x, dict)}
        for row in batch:
            sid = str(row["id"])
            candidate = v9._norm_text((parsed.get(sid) or {}).get("corrected_ru") or "")
            if not candidate:
                continue
            old = str(translated.get(sid) or "")
            translated[sid] = candidate
            remaining = {x["id"]: x for x in _objective_issues(targets, translated)}
            if sid in remaining:
                translated[sid] = old
            else:
                changed.append(sid)
    return changed, calls


def _deepseek_postcondition_fallback(harness, targets, translated, issues) -> tuple[list[str], int]:
    if not issues:
        return [], 0
    cap = max(4, int(os.getenv("BOOKAI_V9AK_POST_DEEP_MAX") or "16"))
    issues = issues[:cap]
    payload = []
    for row in issues:
        i = int(row["index"])
        seg = targets[i]
        payload.append(
            {
                "id": seg.id,
                "failed_checks": row.get("codes") or [],
                "source": str(seg.text or ""),
                "current_ru": str(translated.get(seg.id) or ""),
                "before_en": [x.text for x in targets[max(0, i - 2):i]],
                "after_en": [x.text for x in targets[i + 1:i + 3]],
            }
        )
    system = """LAST-RESORT objective EN→RU postcondition repair. These rows failed deterministic validation after
GigaChat repair. Correct every row from English SOURCE. This is NOT a style pass. Preserve exact meaning, actor/action/
object, numbers, polarity, temporal and causal relations. Remove all accidental Latin-script English prose; render
proper names in stable Cyrillic. Preserve explicit time facts and legal function. Return complete Russian for every id.
ONLY JSON {"items":[{"id":"...","corrected_ru":"..."}]}.
"""
    try:
        obj = v8._complete_json(harness.gate, system, {"items": payload})
        raw = obj.get("items") or []
        calls = 1
    except Exception as exc:
        print(f"[v9ak-post-deep] error={type(exc).__name__}", flush=True)
        return [], 1
    parsed = {str(x.get("id") or ""): x for x in raw if isinstance(x, dict)}
    changed = []
    for row in issues:
        sid = str(row["id"])
        candidate = v9._norm_text((parsed.get(sid) or {}).get("corrected_ru") or "")
        if candidate:
            translated[sid] = candidate
            changed.append(sid)
    return changed, calls


def _hard_postcondition(harness, targets, translated, memory) -> dict[str, Any]:
    initial = _objective_issues(targets, translated)
    changed_giga, giga_calls = _fresh_giga_retranslate(targets, translated, initial)
    residual_after_giga = _objective_issues(targets, translated)
    changed_deep, deep_calls = _deepseek_postcondition_fallback(
        harness, targets, translated, residual_after_giga
    )
    residual_final = _objective_issues(targets, translated)

    if changed_giga or changed_deep:
        v9ab._apply_name_canon(targets, translated, memory)
        for segment in targets:
            translated[segment.id] = v9s._format_dialogue_v9s(
                segment, translated.get(segment.id, "")
            )[0]
        residual_final = _objective_issues(targets, translated)

    hard_residual = [
        row
        for row in residual_final
        if any(
            code == "latin_leak"
            or str(code).startswith("time_marker:")
            or code == "legal_function"
            for code in row.get("codes", [])
        )
    ]
    stats = {
        "initial": len(initial),
        "after_fresh_giga": len(residual_after_giga),
        "final": len(residual_final),
        "hard_final": len(hard_residual),
        "fresh_giga_calls": giga_calls,
        "deepseek_fallback_calls": deep_calls,
        "giga_changed": changed_giga,
        "deep_changed": changed_deep,
        "residual_ids": [x["id"] for x in residual_final],
        "hard_residual_ids": [x["id"] for x in hard_residual],
    }
    _POSTCONDITION_STATS.clear()
    _POSTCONDITION_STATS.update(stats)
    print("[v9ak-postcondition] " + json.dumps(stats, ensure_ascii=False), flush=True)

    if hard_residual:
        raise RuntimeError(
            "v9ak hard postcondition failed; publication output blocked for ids="
            + ",".join(x["id"] for x in hard_residual)
        )
    return stats


def _quality_base_v9ak(harness, targets, translated, memory):
    name_review_calls = _resolve_uncertain_names_once(harness, targets, translated, memory)
    stats = dict(_BASE_AH_QUALITY(harness, targets, translated, memory) or {})
    post = _hard_postcondition(harness, targets, translated, memory)
    stats.update(
        {
            "v9ak_name_registry": dict(_NAME_STATS),
            "v9ak_name_deepseek_review_calls": name_review_calls,
            "v9ak_generic_router": dict(_ROUTER_STATS),
            "v9ak_postcondition": post,
            "quality_contract": (
                "source-only; objective publication blockers fail closed; "
                "legacy quality score is heuristic, regression suite is authoritative for tracked cases"
            ),
        }
    )
    _V9AK_STATS.clear()
    _V9AK_STATS.update(stats)
    return stats


def _annotate_report() -> None:
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    architecture = dict(data.get("architecture") or {})
    architecture.update(
        {
            "version": "quality-v9ak-hardened-v9ah",
            "base": "v9ah adaptive sparse DeepSeek",
            "gigachat_recovery": "degenerate/tokens<=2 same-batch plain-JSON retry before recursive split",
            "router": "general semantic categories + cross-segment alignment guard + adaptive budget",
            "name_memory": "rich source-only registry with confidence/status/contexts; one cached DeepSeek review of important uncertain names",
            "sanitizer": "validated Giga repair -> fresh source retranslation -> sparse DeepSeek fallback -> fail-closed objective postcondition",
            "quality_eval": "tracked regression dataset separated from heuristic critical/major score",
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    data["v9ak_stats"] = {
        "router": dict(_ROUTER_STATS),
        "name_registry": dict(_NAME_STATS),
        "postcondition": dict(_POSTCONDITION_STATS),
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_backend = v9.GigaChatLightningV9Backend
    old_route = v9ah._BASE_AB_ROUTE
    old_quality = v9ah._BASE_QUALITY
    old_name = v9ad._learn_giga_name_canon

    v9.GigaChatLightningV9Backend = GigaChatLightningV9AKBackend
    v9ah._BASE_AB_ROUTE = _route_v9ak
    v9ah._BASE_QUALITY = _quality_base_v9ak
    v9ad._learn_giga_name_canon = _learn_name_registry_v9ak
    try:
        v9ah.main()
    finally:
        v9.GigaChatLightningV9Backend = old_backend
        v9ah._BASE_AB_ROUTE = old_route
        v9ah._BASE_QUALITY = old_quality
        v9ad._learn_giga_name_canon = old_name
        _annotate_report()


if __name__ == "__main__":
    main()
