from __future__ import annotations

import re

import chapter_reference_translation_v9 as v9


# Regression hardening for the exact v8c failure:
# source: "brass ... not brass but bronze"
# bad RU:  "bronze ... not brass but bronze"
# A bag-of-materials presence check cannot detect this, so compare the ordered
# material mentions whenever the source explicitly contrasts materials.

_BASE_DETERMINISTIC_FINDINGS = v9._deterministic_findings


def _source_material_sequence(text: str) -> list[str]:
    low = str(text or "").casefold()
    rows: list[tuple[int, str]] = []
    for material in v9._MATERIALS:
        for match in re.finditer(rf"\b{re.escape(material)}\b", low):
            rows.append((match.start(), material))
    rows.sort()
    return [material for _, material in rows]


def _target_material_sequence(text: str) -> list[str]:
    low = str(text or "").casefold()
    rows: list[tuple[int, str]] = []
    for material, stems in v9._MATERIALS.items():
        for stem in stems:
            for match in re.finditer(re.escape(stem.casefold()), low):
                rows.append((match.start(), material))
    rows.sort()
    # Multiple stems for the same word could theoretically overlap; collapse
    # identical adjacent hits without collapsing genuine repeated mentions.
    out: list[str] = []
    last_pos = -1
    last_material = ""
    for pos, material in rows:
        if pos == last_pos and material == last_material:
            continue
        out.append(material)
        last_pos, last_material = pos, material
    return out


def _deterministic_findings(segment, candidate: str, memory):
    rows = list(_BASE_DETERMINISTIC_FINDINGS(segment, candidate, memory))
    inv = v9._extract_invariants(segment.text)
    if not inv.get("material_contrast"):
        return rows

    source_seq = _source_material_sequence(segment.text)
    target_seq = _target_material_sequence(candidate)
    if source_seq and target_seq[: len(source_seq)] != source_seq:
        key = ("material_order", "explicit material contrast/order changed")
        existing = {(str(row.get("code")), str(row.get("reason"))) for row in rows}
        if key not in existing:
            rows.append({
                "id": segment.id,
                "severity": "critical",
                "confidence": 0.995,
                "code": "material_order",
                "source_span": " → ".join(source_seq),
                "target_span": " → ".join(target_seq[: max(len(source_seq), 1)]),
                "reason": "explicit material contrast/order changed",
                "repairability": "local",
            })
    return rows


# Patch the v9 module itself: every v9 quality/repair/finalization function resolves
# this module-global function at runtime, so the production path and tests use the
# hardened invariant without duplicating the full architecture.
v9._deterministic_findings = _deterministic_findings


def main() -> None:
    v9.main()


if __name__ == "__main__":
    main()
