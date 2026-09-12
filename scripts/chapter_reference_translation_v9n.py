from __future__ import annotations

import json
import re

import chapter_reference_translation_v9m as v9m

v9l = v9m.v9l
v9k = v9m.v9k
v9f = v9m.v9f
v9g = v9m.v9g
v9 = v9m.v9

# v9n makes priority acceptance semantic rather than metadata-only.  The first
# specialist proposal must visibly realize the recovered discourse relation in
# Russian. Rejected items get one bounded retry with several alternatives.

_ADD_RU = re.compile(r"\b(?:ещ[её]|к\s+тому\s+же|вдобавок|да\s+ещ[её]|плюс\s+ко\s+всему|заодно)\b", re.I)
_WITH_IT = re.compile(r"\bwith\s+it\b", re.I)
_MULTI_REFERENT = re.compile(r"\b(?:either|neither|both)\s+of\s+them\b", re.I)
_ELLIPTICAL_MODAL = v9l._ELLIPTICAL_MODAL
_RU_OBLIGATION = v9l._RU_OBLIGATION
_RU_WEAK_HEDGE = v9l._RU_WEAK_HEDGE
_RU_WITH_IT_LITERAL = v9l._RU_WITH_IT_LITERAL


def _stem_ru(label: str) -> str:
    tokens = re.findall(r"[А-Яа-яЁё]+", str(label or "").casefold())
    stop = {"его", "ее", "её", "их", "одна", "один", "вторая", "второй", "человек", "мужчина", "женщина"}
    tokens = [x for x in tokens if x not in stop and len(x) >= 3]
    if not tokens:
        return ""
    token = max(tokens, key=len)
    return token[: max(3, len(token) - 2)]


def _validate_priority_candidate(segment, candidate: str, code: str, item: dict) -> tuple[bool, str]:
    source = str(segment.text or "")
    target = v9._norm_text(candidate)
    low = target.casefold()

    # Retain v9l's generic fatal/literalization constraints first.
    bad, reason = v9l._candidate_still_bad(segment, target, code, item)
    if bad:
        return False, reason

    if code == "idiom" and _WITH_IT.search(source):
        relation = str(item.get("pragmatic_relation") or "").casefold()
        if relation != "additive":
            return False, "with-it pragmatic relation was not resolved as additive"
        if _RU_WITH_IT_LITERAL.search(target):
            return False, "with-it remains literal in Russian"
        if not _ADD_RU.search(target):
            return False, "additive relation is not visibly realized in Russian"

    if code == "idiom" and _ELLIPTICAL_MODAL.search(source):
        source_low = source.casefold()
        source_has_hedge = any(x in source_low for x in ("suppose", "perhaps", "maybe", "probably", "possibly"))
        if _RU_OBLIGATION.search(target):
            return False, "elliptical auxiliary remains obligation"
        if not source_has_hedge and _RU_WEAK_HEDGE.search(target):
            return False, "confident elliptical reply contains invented hedge"
        if not source_has_hedge and str(item.get("certainty") or "").casefold() == "hedged":
            return False, "model classified unhedged reply as hedged"

    if code == "referent" and _MULTI_REFERENT.search(source):
        antecedents_ru = [v9._norm_text(x) for x in (item.get("antecedents_ru") or []) if v9._norm_text(x)]
        if len(antecedents_ru) < 2:
            return False, "Russian antecedent labels are incomplete"
        stems = [_stem_ru(x) for x in antecedents_ru]
        stems = [x for x in stems if x]
        if len(stems) < 2:
            return False, "could not derive two Russian antecedent stems"
        missing = [stem for stem in stems if stem not in re.sub(r"[^а-яё]", "", low)]
        if missing:
            return False, "translation does not explicitly lexicalize both resolved antecedents"

    return True, ""


def _repair_prompt() -> str:
    return """You are a narrow EN→RU FICTION DISCOURSE AND IDIOM REPAIRER. SOURCE English is authoritative. Neighbouring Russian strings are imperfect machine hypotheses, never a gold/reference translation.

For each item return the COMPLETE Russian translation of its source segment, changing only what is needed to repair the flagged pragmatic defect.

REFERENTS: resolve either/neither/both of them to concrete discourse entities. Return both labels in antecedents AND natural Russian labels in antecedents_ru. If both antecedents are not already explicitly named in the same Russian sentence, NAME BOTH in the translation; do not hide them behind их/них/обоих. Preserve negative scope naturally.

ELLIPTICAL AUXILIARIES: reconstruct the omitted proposition from context. A confident reply such as an auxiliary/modal answer should remain confident; do not translate it as obligation and do not add perhaps/maybe/пожалуй unless the English is genuinely hedged.

ATTITUDE + WITH IT: decide whether "with it" means additive "as well / on top of that / besides". When it does, set pragmatic_relation="additive" and EXPRESS that additive relation explicitly and idiomatically in Russian using a natural equivalent such as an ещё-и / к-тому-же / вдобавок relation. Never use literal "с этим" or temporal/discourse "при этом" for the additive meaning. Preserve the speaker's attitude rather than explaining it.

Return ONLY JSON:
{"items":[{"id":"exact id","verdict":"keep|replace","antecedents":["..."],"antecedents_ru":["..."],"certainty":"strong|neutral|hedged","pragmatic_relation":"additive|other","meaning":"brief pragmatic gloss","translation":"complete Russian segment"}]}"""


def _retry_prompt() -> str:
    return """You are repairing a previously rejected EN→RU fiction idiom/referent candidate. SOURCE English and the supplied failure_reason are authoritative. Do not return the failed formulation again.
Generate THREE genuinely different COMPLETE Russian translations of the source segment. Preserve every clause and speaker attribution.
For additive attitude + "with it", every alternative must make the additive 'also/on top of that' relation explicit in natural Russian and must not use literal 'с этим' or 'при этом'.
For an unhedged elliptical auxiliary reply, preserve confident pragmatic force; no obligation and no invented uncertainty/hedging.
For either/neither/both-of-them, resolve exactly two discourse antecedents and explicitly name both in Russian if they are not both named in the current sentence; do not use a bare plural pronoun as a substitute.
Return ONLY JSON:
{"items":[{"id":"exact id","antecedents":["..."],"antecedents_ru":["..."],"certainty":"strong|neutral|hedged","pragmatic_relation":"additive|other","meaning":"brief gloss","alternatives":["complete RU 1","complete RU 2","complete RU 3"]}]}"""


def _priority_semantic_repair_v9n(harness, targets, translated, memory):
    rows = v9m._priority_rows_v9m(targets, translated)
    payload = v9k._priority_payload(targets, translated, rows)
    if not payload:
        return {"flagged": 0, "replaced": 0, "rejected": 0, "ids": []}

    try:
        obj = v9.v8._complete_json(harness.gate, _repair_prompt(), {"items": payload})
    except Exception as exc:
        print(f"[v9n-priority] error={type(exc).__name__}", flush=True)
        return {"flagged": len(payload), "replaced": 0, "rejected": len(payload), "ids": [], "rejected_ids": [x["id"] for x in payload]}

    returned = obj.get("items") if isinstance(obj, dict) else []
    result_by_id = {str(x.get("id") or ""): x for x in returned if isinstance(x, dict)} if isinstance(returned, list) else {}
    by_segment = {segment.id: segment for segment in targets}
    by_finding = {str(row.get("id") or ""): row for row in rows}
    replaced = []
    retry_items = []

    def try_candidate(sid, finding, item, candidate):
        segment = by_segment[sid]
        current = v9._norm_text(translated.get(sid, ""))
        candidate = v9._norm_text(candidate)
        if not candidate or candidate == current:
            return False, "empty or unchanged"
        if v9._fatal_count(segment, candidate, memory) > v9._fatal_count(segment, current, memory):
            return False, "fatal guard"
        ok, reason = _validate_priority_candidate(segment, candidate, str(finding.get("code") or ""), item)
        if not ok:
            return False, reason
        translated[sid] = candidate
        replaced.append(sid)
        print("[v9n-priority-repair] " + json.dumps({
            "id": sid, "code": finding.get("code"), "meaning": v9._norm_text(item.get("meaning") or "")[:180],
            "antecedents": item.get("antecedents") or [], "antecedents_ru": item.get("antecedents_ru") or [],
            "certainty": item.get("certainty"), "relation": item.get("pragmatic_relation"), "before": current[:240], "after": candidate[:240]
        }, ensure_ascii=False), flush=True)
        return True, ""

    for sid, finding in by_finding.items():
        segment = by_segment.get(sid)
        item = result_by_id.get(sid)
        if segment is None or item is None:
            retry_items.append({"id": sid, "finding": finding, "failure_reason": "missing first-pass result"})
            continue
        candidate = item.get("translation") if str(item.get("verdict") or "").casefold() == "replace" else translated.get(sid, "")
        ok, reason = try_candidate(sid, finding, item, candidate)
        if not ok:
            retry_items.append({"id": sid, "finding": finding, "failure_reason": reason})
            print("[v9n-priority-first-reject] " + json.dumps({"id": sid, "reason": reason}, ensure_ascii=False), flush=True)

    # Exactly one bounded retry, only for candidates that failed semantic invariants.
    if retry_items:
        retry_payload = []
        for row in retry_items:
            sid = row["id"]
            segment = by_segment[sid]
            retry_payload.append({
                "id": sid,
                "issue": row["finding"].get("code"),
                "failure_reason": row["failure_reason"],
                "source": segment.text,
                "current_ru": translated.get(sid, ""),
                "context": v9f._context_payload(targets, translated, segment),
            })
        try:
            retry_obj = v9.v8._complete_json(harness.gate, _retry_prompt(), {"items": retry_payload})
        except Exception as exc:
            print(f"[v9n-priority-retry] error={type(exc).__name__}", flush=True)
            retry_obj = {}
        retry_rows = retry_obj.get("items") if isinstance(retry_obj, dict) else []
        retry_by_id = {str(x.get("id") or ""): x for x in retry_rows if isinstance(x, dict)} if isinstance(retry_rows, list) else {}
        unresolved = []
        for row in retry_items:
            sid = row["id"]
            if sid in replaced:
                continue
            item = retry_by_id.get(sid)
            if not item:
                unresolved.append(sid)
                continue
            accepted = False
            for candidate in (item.get("alternatives") or [])[:3]:
                ok, reason = try_candidate(sid, row["finding"], item, candidate)
                if ok:
                    accepted = True
                    break
            if not accepted:
                unresolved.append(sid)
                print("[v9n-priority-final-reject] " + json.dumps({"id": sid, "reason": "all retry alternatives failed invariants"}, ensure_ascii=False), flush=True)
    else:
        unresolved = []

    return {"flagged": len(payload), "replaced": len(set(replaced)), "rejected": len(unresolved), "ids": sorted(set(replaced)), "rejected_ids": unresolved}


# Reuse v9l's fast priority-only quality/export, swapping only the specialist.
v9l._priority_semantic_repair_v9l = _priority_semantic_repair_v9n


def main():
    v9m.main()
    report = v9.v3.REPORT
    if report.exists():
        try:
            data = json.loads(report.read_text("utf-8"))
            data["architecture"] = {
                **dict(data.get("architecture") or {}),
                "version": "quality-v9n-semantic-invariant-retry",
                "priority_candidate_validation": "visible additive relation + confident ellipsis + explicit lexicalized multi-antecedents",
                "priority_retry": "one bounded 3-alternative retry only for rejected priority items",
                "gold_reference_available_to_pipeline": False,
            }
            report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
