from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import chapter_reference_translation_v9b as v9b

v9 = v9b.v9


# v9c addresses the remaining blind-test defects from v9:
# - short dialogue idioms/ellipses ("Cocky with it", "I should do")
# - contextual referents ("either of them")
# - Thread Index mismatches must not crowd semantic defects out of the repair budget
# - typography must be mechanically safe: never create double commas/unbalanced quotes.

_BASE_PARALLEL_AUDIT = v9._parallel_audit
_BASE_THREAD_FINDINGS = v9._thread_findings


def _risk_segment(segment) -> bool:
    text = str(segment.text or "")
    if v9._extract_invariants(text).get("referent_risk"):
        return True
    return v9._is_dialogue(text) and len(text) <= max(180, int(os.getenv("BOOKAI_V9C_DIALOGUE_MAX_CHARS") or "520"))


def _micro_audit_batch(provider, batch, translations, targets):
    pos = {segment.id: i for i, segment in enumerate(targets)}
    pairs = {}
    for segment in batch:
        i = pos.get(segment.id, 0)
        pairs[segment.id] = {
            "en": segment.text,
            "ru": translations.get(segment.id, ""),
            "before_en": [row.text for row in targets[max(0, i - 2):i]],
            "after_en": [row.text for row in targets[i + 1:i + 3]],
        }
    system = """You are a narrow EN→RU FICTION DIALOGUE + REFERENT auditor. Do not rewrite and do not review general style.
Only report these high-value defects:
1) wrong contextual referent/relationship: either/neither/both/them/they/his/her/former/latter/the other;
2) English conversational idiom or elliptical phrase translated literally or with the wrong pragmatic meaning;
3) a short dialogue reply whose Russian meaning is materially wrong because English omitted recoverable words (e.g. 'I should do' = 'of course I know/can', not obligation);
4) dialogue attribution changes speaker gender/identity.
Use BEFORE/AFTER source context. Do not flag merely different but natural Russian wording.
Severity critical = factual/referent/speaker relation is wrong or lost. Severity major = idiom/elliptical dialogue meaning is clearly wrong.
Return ONLY JSON {"issues":[{"id":"exact id","severity":"critical|major","confidence":0.0,
"code":"referent|idiom|dialogue|relation|voice","source_span":"short exact span","target_span":"short span",
"reason":"specific concise explanation","repairability":"contextual|local"}]}. Use [] if correct."""
    raw = provider.complete(system, "PAIRS:" + json.dumps(pairs, ensure_ascii=False), temperature=0.0)
    obj = v9.extract_json(raw)
    rows = obj.get("issues") if isinstance(obj, dict) else []
    valid = set(pairs)
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if sid not in valid:
            if len(valid) == 1:
                sid = next(iter(valid))
            else:
                continue
        sev = str(row.get("severity") or "major").lower()
        if sev not in {"critical", "major"}:
            sev = "major"
        try:
            conf = max(0.0, min(1.0, float(row.get("confidence") or 0.5)))
        except Exception:
            conf = 0.5
        reason = v9._norm_text(row.get("reason") or "")
        if not reason:
            continue
        out.append({
            "id": sid,
            "severity": sev,
            "confidence": conf,
            "code": str(row.get("code") or "dialogue").lower(),
            "source_span": v9._norm_text(row.get("source_span") or "")[:220],
            "target_span": v9._norm_text(row.get("target_span") or "")[:220],
            "reason": reason[:500],
            "repairability": str(row.get("repairability") or "contextual").lower(),
        })
    return out


def _parallel_micro_audit(harness, targets, translations, selected=None):
    source_rows = selected if selected is not None else targets
    risky = [segment for segment in source_rows if _risk_segment(segment)]
    if not risky:
        return []
    max_chars = max(4000, int(os.getenv("BOOKAI_V9C_MICRO_BATCH_CHARS") or "9000"))
    workers = max(1, min(5, int(os.getenv("BOOKAI_V9C_MICRO_WORKERS") or "4")))
    batches = v9.v6._qe_batches(risky, max_chars)

    def run(batch):
        try:
            return _micro_audit_batch(harness.gate, list(batch), translations, targets)
        except Exception as exc:
            if len(batch) <= 1:
                print(f"[v9c-micro-audit] id={batch[0].id if batch else '?'} error={type(exc).__name__}", flush=True)
                return []
            mid = len(batch) // 2
            out = []
            for part in (batch[:mid], batch[mid:]):
                try:
                    out.extend(_micro_audit_batch(harness.gate, list(part), translations, targets))
                except Exception as inner:
                    print(f"[v9c-micro-audit-split] size={len(part)} error={type(inner).__name__}", flush=True)
            return out

    out = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-v9c-micro") as pool:
        futures = [pool.submit(run, batch) for batch in batches]
        for future in as_completed(futures):
            out.extend(future.result())
    return out


def _parallel_audit_v9c(harness, targets, translations, memory, selected=None):
    general = _BASE_PARALLEL_AUDIT(harness, targets, translations, memory, selected=selected)
    micro = _parallel_micro_audit(harness, targets, translations, selected=selected)
    # Do not duplicate identical reports from general + micro auditors.
    out = []
    seen = set()
    for row in [*general, *micro]:
        key = (row.get("id"), row.get("code"), row.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _thread_findings_v9c(targets, translations, memory):
    rows, index = _BASE_THREAD_FINDINGS(targets, translations, memory)
    # Thread drift is useful evidence for trusted-TM filtering/reporting, but it is
    # not by itself proof of a semantic error. Entity typos are already covered by
    # the deterministic entity layer. Keep thread signals advisory so they cannot
    # consume the high-confidence semantic repair budget.
    for row in rows:
        row["severity"] = "minor"
        row["confidence"] = min(0.82, float(row.get("confidence") or 0.7))
    return rows, index


def _remove_unmatched_guillemets(value: str) -> str:
    stack = []
    remove = set()
    for i, char in enumerate(value):
        if char == "«":
            stack.append(i)
        elif char == "»":
            if stack:
                stack.pop()
            else:
                remove.add(i)
    remove.update(stack)
    if not remove:
        return value
    return "".join(char for i, char in enumerate(value) if i not in remove)


def _format_dialogue_v9c(segment, text: str):
    before = v9._norm_text(text)
    value = before.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    # Mechanical punctuation hygiene first. Never duplicate punctuation around a
    # removed quotation mark.
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r"(?<=[А-Яа-яЁё0-9])['\"](?=[,!?….])", "", value)
    value = re.sub(r"([,!?….])['\"](?=\s*(?:—|-))", r"\1", value)

    if v9._is_dialogue(segment.text):
        value = re.sub(r"^\s*['\"]\s*", "— ", value, count=1)
        # A new quoted utterance after a completed dialogue tag becomes another
        # dialogue dash. This handles: 'A,' he said. 'B.'
        value = re.sub(r"(?<=[.!?…])\s*['\"]\s*(?=[А-ЯЁ])", " — ", value)
        # Remove closing straight quotes that clearly terminate an utterance.
        value = re.sub(r"([,!?….])['\"](?=\s|$)", r"\1", value)
        value = re.sub(r"['\"]\s*$", "", value)
    else:
        # Narration: paired straight quotes are ordinary Russian guillemets.
        value = re.sub(r"['\"]([^'\"\n]{1,220})['\"]", r"«\1»", value)

    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    value = _remove_unmatched_guillemets(value)
    value = re.sub(r",\s*,+", ",", value)
    return value, int(value != before)


# Patch v9 runtime globals. v9's quality/finalizer functions resolve these symbols
# dynamically, so the entire pipeline gets the narrow audit and advisory threads.
v9._parallel_audit = _parallel_audit_v9c
v9._thread_findings = _thread_findings_v9c
v9._format_dialogue_v9 = _format_dialogue_v9c


def main() -> None:
    v9b.main()


if __name__ == "__main__":
    main()
