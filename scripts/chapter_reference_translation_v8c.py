from __future__ import annotations

import json
import re
from pathlib import Path

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v6 as v6
import chapter_reference_translation_v8 as v8
import chapter_reference_translation_v8b as v8b
from bookai.pipeline import _chapter_groups, _should_translate


# v8c fixes two benchmark/production defects exposed by the first-three run:
# 1) internal titled sections (e.g. "SEEMED LIKE A GOOD IDEA AT THE TIME")
#    belong to the enclosing Chapter Two and must not truncate the chapter;
# 2) deterministic dialogue cleanup must remove mixed ASCII/guillemets artifacts
#    left after model edits.  The quality architecture itself remains v8b.

_V8C_STATS: dict = {}
_ORIGINAL_V8_NORMALIZE = v8._normalize_dialogue_v8

_CHAPTER_HEADING_RE = re.compile(
    r"^chapter\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty(?:[- ](?:one|two|three|four))?|\d+)\s*$",
    re.I,
)


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _select_complete_chapter(document):
    """Select a real book chapter, including any internally titled sub-sections."""
    targets = [segment for segment in document.segments if _should_translate(segment.text)]
    v6._SOURCE_SEGMENTS = list(targets)
    groups = _chapter_groups(targets)
    wanted = _norm(v3.CHAPTER_NAME)
    matches = [i for i, (name, _) in enumerate(groups) if _norm(name) == wanted]
    if not matches:
        matches = [i for i, (name, _) in enumerate(groups) if wanted and wanted in _norm(name)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one top-level chapter matching {wanted!r}; found {len(matches)}. "
            f"Available={[name for name, _ in groups]}"
        )

    start = matches[0]
    chapter_name = groups[start][0]
    rows = list(groups[start][1])
    included_groups = [chapter_name]
    for name, group_rows in groups[start + 1 :]:
        # The next actual 'Chapter N' title ends this chapter. Any other title is
        # a scene/part/sub-section title and remains inside the parent chapter.
        if _CHAPTER_HEADING_RE.match(str(name or "").strip()):
            break
        rows.extend(group_rows)
        included_groups.append(name)

    print(
        "[v8c-complete-chapter] "
        + json.dumps(
            {
                "chapter": chapter_name,
                "included_groups": included_groups,
                "segments": len(rows),
                "source_chars": sum(len(segment.text) for segment in rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return chapter_name, rows


def _normalize_dialogue_v8c(segment, text: str):
    value, _ = _ORIGINAL_V8_NORMALIZE(segment, text)
    before = value

    # Models occasionally return hybrid English/Russian dialogue punctuation after
    # an otherwise good edit. Repair only punctuation marks; never rewrite words.
    value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"([,!?…])\s*[«»]\s*(?=[а-яё])", r"\1 — ", value)
    value = re.sub(r"\.\s*[«»]\s*,?\s*—\s*(?=[а-яё])", r", — ", value)
    value = re.sub(r"—\s*[«»'\"]\s*(?=[А-ЯЁ])", "— ", value)
    value = re.sub(r"(?<=[А-Яа-яЁё0-9,.!?…])['\"](?=\s*(?:—|[,.!?…]|$))", "", value)
    value = re.sub(r"(?:(?<=\s)|^)['\"](?=[А-ЯЁ])", "", value)
    value = re.sub(r"(?<=[А-Яа-яЁё])['\"](?=[,.!?…])", "", value)
    value = re.sub(r"\s+([,.!?…])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value).strip()

    return value, int(value != before)


def _configure_v8c() -> None:
    v8b._configure_v8b()
    slug = re.sub(r"[^a-z0-9]+", "-", str(v3.CHAPTER_NAME or "chapter").casefold()).strip("-") or "chapter"
    v3._select_chapter = _select_complete_chapter
    v8._normalize_dialogue_v8 = _normalize_dialogue_v8c
    v3.OUTPUT = Path(f"Devices_and_Desires_RU_{slug}_EVAL_V8C.fb2")
    v3.PROGRESS = Path(f"chapter-v8c-{slug}-progress.json")
    v3.PROBE = Path(f"chapter-v8c-{slug}-probe.json")
    v3.ROUTING = Path(f"chapter-v8c-{slug}-routing.json")
    v3.REPORT = Path(f"chapter-v8c-{slug}.json")
    v3.SOURCE_TXT = Path(f"chapter-v8c-{slug}-source.txt")
    v3.TRANSLATED_TXT = Path(f"chapter-v8c-{slug}-translated.txt")
    v3.MAP_JSON = Path(f"chapter-v8c-{slug}-translation-map.json")


def _annotate_v8c() -> None:
    if not v3.REPORT.exists():
        return
    try:
        report = json.loads(v3.REPORT.read_text("utf-8"))
    except Exception:
        return
    report["architecture"] = {
        **dict(report.get("architecture") or {}),
        "version": "quality-v8c-complete-chapters",
        "chapter_selection": "top-level Chapter N span including internal titled sections",
        "dialogue_cleanup": "deterministic mixed-quote cleanup",
        "gold_reference_available_to_pipeline": False,
    }
    report["v8c_stats"] = dict(_V8C_STATS)
    v3.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    _configure_v8c()
    try:
        v3.main()
    finally:
        v6._annotate_report()
        v8._annotate_v8()
        v8b._annotate_v8b()
        _annotate_v8c()
        v8._normalize_dialogue_v8 = _ORIGINAL_V8_NORMALIZE


if __name__ == "__main__":
    main()
