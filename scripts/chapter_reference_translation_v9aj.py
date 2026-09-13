from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v9ah as v9ah


_BASE_AB_ROUTE = v9ah._BASE_AB_ROUTE
_DUPLICATE_STATS: dict[str, Any] = {}


def _norm(value: str) -> str:
    value = str(value or "").casefold().replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9]+", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def _similarity(a: str, b: str) -> float:
    a_n, b_n = _norm(a), _norm(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n, autojunk=False).ratio()


def _cross_segment_duplicate_routes(targets, translated) -> list[dict[str, Any]]:
    """Detect likely JSON/id-alignment mistakes without a model call.

    A long literary translation can legitimately repeat short formulaic phrases, so
    this intentionally ignores short rows. A pair is suspicious only when the two
    Russian outputs are almost the same while the two English sources are clearly
    different. Both rows are routed: the independent bilingual specialist decides
    which one is wrong and keeps the genuinely correct one unchanged.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    n = len(targets)
    for i in range(n):
        left = targets[i]
        left_ru = str(translated.get(left.id) or "").strip()
        if len(left_ru) < 180 or len(str(left.text or "")) < 100:
            continue
        for j in range(i + 1, n):
            right = targets[j]
            right_ru = str(translated.get(right.id) or "").strip()
            if len(right_ru) < 180 or len(str(right.text or "")) < 100:
                continue

            # Cheap length guard avoids running SequenceMatcher on obviously
            # unrelated translations.
            length_ratio = min(len(left_ru), len(right_ru)) / max(len(left_ru), len(right_ru))
            if length_ratio < 0.74:
                continue
            ru_similarity = _similarity(left_ru, right_ru)
            if ru_similarity < 0.90:
                continue
            source_similarity = _similarity(left.text, right.text)
            if source_similarity > 0.58:
                continue

            key = tuple(sorted((left.id, right.id)))
            if key in seen:
                continue
            seen.add(key)
            reason = (
                "possible cross-segment JSON/id alignment error: Russian outputs for "
                f"{left.id} and {right.id} are {ru_similarity:.3f} similar while English "
                f"sources are only {source_similarity:.3f} similar. Re-derive this row "
                "from its own SOURCE and neighboring English context; do not copy the peer."
            )
            for index, segment, peer in ((i, left, right.id), (j, right, left.id)):
                rows.append(
                    {
                        "id": segment.id,
                        "index": index,
                        "code": "cross_segment_duplicate",
                        "priority": 24,
                        "reason": reason + f" peer_id={peer}",
                        "glossary_hits": [],
                    }
                )
    # One route per id even if a pathological repeated output matched >1 peer.
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        old = by_id.get(row["id"])
        if old is None or int(row["priority"]) > int(old.get("priority") or 0):
            by_id[row["id"]] = row
    return list(by_id.values())


def _route_v9aj(targets, translated, memory):
    selected0, ranked0 = _BASE_AB_ROUTE(targets, translated, memory)
    ranked_by_id = {str(row.get("id") or ""): dict(row) for row in ranked0}
    duplicates = _cross_segment_duplicate_routes(targets, translated)

    for row in duplicates:
        sid = row["id"]
        current = ranked_by_id.get(sid)
        if current is None:
            ranked_by_id[sid] = dict(row)
            continue
        # A duplicate/alignment signal is independent evidence. Preserve any
        # existing risk explanation but promote it to a mandatory route.
        current["priority"] = max(int(current.get("priority") or 0), int(row["priority"]))
        current["reason"] = (
            str(current.get("reason") or "").strip()
            + "; "
            + str(row.get("reason") or "").strip()
        ).strip("; ")
        current["code"] = "cross_segment_duplicate"
        ranked_by_id[sid] = current

    ranked = sorted(
        ranked_by_id.values(),
        key=lambda r: (-int(r.get("priority") or 0), int(r.get("index") or 0)),
    )
    duplicate_ids = [row["id"] for row in duplicates]
    _DUPLICATE_STATS.clear()
    _DUPLICATE_STATS.update(
        {
            "candidates": len(duplicate_ids),
            "ids": duplicate_ids,
            "mandatory_priority": 24,
            "detector": "long-RU-near-duplicate + dissimilar-EN-source",
        }
    )
    if duplicate_ids:
        print("[v9aj-cross-segment-duplicate] " + json.dumps(_DUPLICATE_STATS, ensure_ascii=False), flush=True)
    return selected0, ranked


def _annotate() -> None:
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
            "version": "quality-v9aj-cross-segment-alignment-guard",
            "base": "v9ah adaptive sparse DeepSeek + validated source sanitizer",
            "cross_segment_guard": (
                "model-free long target near-duplicate detection with dissimilar English sources; "
                "both rows become mandatory bilingual repair routes"
            ),
            "gold_reference_available_to_pipeline": False,
        }
    )
    data["architecture"] = architecture
    data["v9aj_cross_segment_duplicate"] = dict(_DUPLICATE_STATS)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    old_base = v9ah._BASE_AB_ROUTE
    v9ah._BASE_AB_ROUTE = _route_v9aj
    try:
        v9ah.main()
    finally:
        v9ah._BASE_AB_ROUTE = old_base
        _annotate()


if __name__ == "__main__":
    main()
