from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9am_numeric_fast as v9am
from bookai.address_fidelity import normalize_numbered_address_literals


_BASE_CONTRACT_ROUTE = v9am._contract_route_v9am
_BASE_HARD_ROWS = v9am._contract_hard_rows
_ADDRESS_STATS: dict[str, Any] = {
    "normalization_passes": 0,
    "segments_changed": 0,
    "replacements": 0,
    "changed_ids": [],
}


def _normalize_address_literals(targets, translated) -> dict[str, Any]:
    _ADDRESS_STATS["normalization_passes"] = int(_ADDRESS_STATS.get("normalization_passes") or 0) + 1
    changed_ids = set(str(x) for x in _ADDRESS_STATS.get("changed_ids") or [])
    replacements = 0
    newly_changed = 0
    for segment in targets:
        sid = str(segment.id)
        current = str(translated.get(sid) or "")
        if not current:
            continue
        fixed, count = normalize_numbered_address_literals(str(segment.text or ""), current)
        if count <= 0 or fixed == current:
            continue
        translated[sid] = fixed
        replacements += int(count)
        if sid not in changed_ids:
            newly_changed += 1
            changed_ids.add(sid)
    _ADDRESS_STATS["replacements"] = int(_ADDRESS_STATS.get("replacements") or 0) + replacements
    _ADDRESS_STATS["segments_changed"] = len(changed_ids)
    _ADDRESS_STATS["changed_ids"] = sorted(changed_ids)
    if replacements:
        print(
            "[v9an-address-normalize] "
            + json.dumps(
                {
                    "replacements_this_pass": replacements,
                    "new_segments_this_pass": newly_changed,
                    "changed_ids": sorted(changed_ids),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return dict(_ADDRESS_STATS)


def _address_aware_route(targets, translated, memory):
    # Clean source-proven untranslated numeric address labels BEFORE routing so
    # they never consume a DeepSeek specialist slot merely to translate a street
    # number that deterministic code can prove and render safely.
    _normalize_address_literals(targets, translated)
    return _BASE_CONTRACT_ROUTE(targets, translated, memory)


def _address_aware_hard_rows(targets, translated, memory):
    # Run the same narrow postcondition after sanitizer/repair in case an upstream
    # edit reintroduced the English label. This pass is deterministic and free.
    _normalize_address_literals(targets, translated)
    return _BASE_HARD_ROWS(targets, translated, memory)


def _annotate_report() -> None:
    v3 = v9am.v9al.v9ah.v9ag.v9ad.v9ac.v9ab.v3
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
            "experiment": "v9an-fast-deepseek-address-contracts",
            "numbered_address_cleanup": (
                "deterministic source-proven English ordinal/cardinal address labels are rendered in Russian before specialist routing"
            ),
            "gigachat_ultra_used": False,
        }
    )
    data["architecture"] = architecture
    data["v9an_address_fidelity"] = dict(_ADDRESS_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_route = v9am._contract_route_v9am
    old_rows = v9am._contract_hard_rows
    v9am._contract_route_v9am = _address_aware_route
    v9am._contract_hard_rows = _address_aware_hard_rows
    try:
        v9am.main()
    finally:
        v9am._contract_route_v9am = old_route
        v9am._contract_hard_rows = old_rows
        _annotate_report()


if __name__ == "__main__":
    main()
