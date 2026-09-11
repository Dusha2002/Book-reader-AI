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
#
# Research-informed additions:
# - discourse audit sees narrowly relevant source AND our own neighbouring Russian
#   hypotheses (not a gold translation), because document-MT research shows this is
#   more useful for pronouns/cohesion than indiscriminate long context;
# - special dialogue/referent defects receive an independent fresh translation
#   candidate in addition to editor candidates, then v9's QE judge ranks them.

_BASE_PARALLEL_AUDIT = v9._parallel_audit
_BASE_THREAD_FINDINGS = v9._thread_findings
_BASE_SPECIALIST_CANDIDATES = v9._specialist_candidates


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
        before = targets[max(0, i - 4):i]
        after = targets[i + 1:i + 3]
        pairs[segment.id] = {
            "en": segment.text,
            "ru": translations.get(segment.id, ""),
            "before_en": [row.text for row in before],
            "before_ru": [translations.get(row.id, "") for row in before],
            "after_en": [row.text for row in after],
            "after_ru": [translations.get(row.id, "") for row in after],
        }
    system = """You are a narrow EN→RU FICTION DIALOGUE + REFERENT auditor. Do not rewrite and do not review general style.
The BEFORE/AFTER Russian lines are the system's own imperfect hypotheses, never a gold/reference translation. Use them only as
soft discourse evidence; SOURCE English is authoritative.
Only report these high-value defects:
1) wrong contextual referent/relationship: either/neither/both/them/they/his/her/former/latter/the other;
2) English conversational idiom or elliptical phrase translated literally or with the wrong pragmatic meaning;
3) a short dialogue reply whose Russian meaning is materially wrong because English omitted recoverable words
   (for example a response such as 'I should do' can mean 'of course I know/can', not obligation);
4) dialogue attribution changes speaker gender/identity;
5) a reply contradicts the immediately preceding question/statement because the ellipsis was expanded incorrectly.
Resolve antecedents explicitly from relevant context before judging. Do not flag merely different but natural Russian wording.
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


def _fresh_context_candidate(harness, targets, segment, memory) -> str:
    position = {row.id: i for i, row in enumerate(targets)}
    i = position.get(segment.id, 0)
    try:
        rows = harness.translator.translate(
            [segment],
            memory,
            context_before=targets[max(0, i - 4):i],
            context_after=targets[i + 1:i + 3],
        )
        return v9._norm_text(rows.get(segment.id, ""))
    except Exception as exc:
        print(f"[v9c-independent-candidate] id={segment.id} error={type(exc).__name__}", flush=True)
        return ""


def _specialist_candidates_v9c(harness, targets, segment, current, memory, findings):
    candidates = list(_BASE_SPECIALIST_CANDIDATES(harness, targets, segment, current, memory, findings))
    codes = {str(row.get("code") or "") for row in findings}
    high_value = bool(codes & {"idiom", "dialogue", "referent", "relation", "voice"})
    if high_value:
        fresh = _fresh_context_candidate(harness, targets, segment, memory)
        if fresh and fresh.casefold() not in {v9._norm_text(x).casefold() for x in [current, *candidates] if v9._norm_text(x)}:
            candidates.append(fresh)
    # At most three alternatives + current in the v9 judge. This keeps the MBR/QE
    # tail cheap while making the candidates genuinely less correlated.
    return candidates[:3]


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
    # removed quotation mark, and never rewrite lexical content.
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r"(?<=[А-Яа-яЁё0-9])['\"](?=[,!?….])", "", value)
    value = re.sub(r"([,!?….])['\"](?=\s*(?:—|-))", r"\1", value)

    if v9._is_dialogue(segment.text):
        value = re.sub(r"^\s*['\"]\s*", "— ", value, count=1)
        value = re.sub(r"(?<=[.!?…])\s*['\"]\s*(?=[А-ЯЁ])", " — ", value)
        value = re.sub(r"([,!?….])['\"](?=\s|$)", r"\1", value)
        value = re.sub(r"['\"]\s*$", "", value)
        # A quote immediately after a dialogue dash is an English-format artifact.
        value = re.sub(r"—\s*['\"]\s*(?=[А-ЯЁ])", "— ", value)
    else:
        value = re.sub(r"['\"]([^'\"\n]{1,220})['\"]", r"«\1»", value)

    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    value = _remove_unmatched_guillemets(value)
    value = re.sub(r",\s*,+", ",", value)
    return value, int(value != before)


# Patch v9 runtime globals. v9's quality/finalizer functions resolve these symbols
# dynamically, so the whole production path gets the focused discourse audit,
# independent hard-tail candidate, advisory threads and safe formatter.
v9._parallel_audit = _parallel_audit_v9c
v9._thread_findings = _thread_findings_v9c
v9._specialist_candidates = _specialist_candidates_v9c
v9._format_dialogue_v9 = _format_dialogue_v9c


def _annotate_v9c() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9c-focused-discourse-mbr-tail",
        "discourse_context": "relevant source + self-generated neighboring RU hypotheses; no gold target",
        "special_tail": "editor candidates + independent fresh candidate + QE ranking",
        "thread_index": "advisory for repair routing; still blocks trusted-TM promotion on mismatch",
        "formatter": "lexically inert Russian dialogue punctuation hygiene",
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9b.main()
    finally:
        _annotate_v9c()


if __name__ == "__main__":
    main()
