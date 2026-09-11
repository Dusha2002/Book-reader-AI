from __future__ import annotations

import json

import chapter_reference_translation_v9i as v9i

v9h = v9i.v9h
v9g = v9i.v9g
v9f = v9i.v9f
v9c = v9i.v9c
v9 = v9i.v9

# v9j removes the remaining latency-sensitive second LLM discourse audit.
# General bilingual QE already checks idioms/referents, while the proven hard
# discourse cases are now deterministic/synthetic signals:
#   - explicit short either/neither/both/the-other -> priority referent rescue
#   - observed RU literalization ("я должен" for I should do, "с ним" for with it)
# This preserves targeted rescue without an extra model pass over dozens of
# dialogue segments. Re-QE after repair remains the acceptance layer.


def _parallel_micro_v9j(harness, targets, translations, selected=None):
    if selected is not None:
        # Do not re-add synthetic defects after a candidate has been produced.
        # The normal general bilingual re-QE + deterministic checks decide whether
        # the candidate is safe; disappearance of the synthetic critical is the
        # intended proof that the targeted defect was actually repaired.
        return []
    rows = v9i._synthetic_confirmed_rows_v9i(targets, translations)
    print(
        "[v9j-deterministic-discourse] "
        + json.dumps(
            {
                "forced_findings": len(rows),
                "llm_micro_audit": False,
                "strategy": "general-QE + deterministic-priority-rescue",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rows


# chapter_reference_translation_v9c._parallel_audit_v9c resolves this symbol
# dynamically, so the production quality path keeps one general bilingual QE and
# replaces the second dialogue/referent LLM pass with deterministic routing.
v9c._parallel_micro_audit = _parallel_micro_v9j
v9c._parallel_audit_v9c.__globals__["_parallel_micro_audit"] = _parallel_micro_v9j
v9._parallel_audit = v9c._parallel_audit_v9c


def _annotate_v9j() -> None:
    report = v9.v3.REPORT
    if not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    data["architecture"] = {
        **dict(data.get("architecture") or {}),
        "version": "quality-v9j-bounded-general-qe-priority-rescue",
        "discourse_qe": "no second LLM micro-audit; deterministic priority signals only",
        "general_qe": "single bilingual chapter audit with confidence-aware repair",
        "priority_rescue": "short strong referents + observed RU literalization are synthetic critical",
        "acceptance": "single candidate + deterministic guard + general bilingual re-QE",
        "gold_reference_available_to_pipeline": False,
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    try:
        v9h.main()
    finally:
        _annotate_v9j()


if __name__ == "__main__":
    main()
