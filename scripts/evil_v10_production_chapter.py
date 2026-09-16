from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bookai.pipeline import _chapter_groups, _should_translate
from bookai.v10 import DeterministicQA, V10Issue
from bookai.v10_source_bible import _TITLE_WORDS
from bookai.v10_production_runtime import install_v10_production_runtime_guard
import bookai.v10_general_release as general_release
import bookai.release_final as release_final
import bookai.v10_release as v10_release


_BASE_SELECT = v10_release.select_numbered_chapter
_BASE_CANDIDATES = release_final.FinalBookBibleBuilder._candidate_records
_BASE_CLEAN_QUANTITY = general_release.FinalV10QualityQA._clean_quantity
_BARE_NUMBER = re.compile(r"^\d+$")
_PRODUCTION_HARDENING = "v10-production-hardening-3"


def _select_numeric_or_named_chapter(document, chapter_name: str):
    """Select one complete chapter for both `Chapter One` and bare numeric FB2 books."""
    wanted = " ".join(str(chapter_name or "").casefold().split())
    if not _BARE_NUMBER.fullmatch(wanted):
        return _BASE_SELECT(document, chapter_name)

    all_targets = [segment for segment in document.segments if _should_translate(segment.text)]
    groups = _chapter_groups(all_targets)
    exact = [
        i for i, (name, _rows) in enumerate(groups)
        if " ".join(str(name or "").casefold().split()) == wanted
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected one bare numeric chapter matching {chapter_name!r}; found {len(exact)}; "
            f"available={[name for name, _ in groups]}"
        )

    start = exact[0]
    end = len(groups)
    next_chapter = ""
    for i in range(start + 1, len(groups)):
        label = " ".join(str(groups[i][0] or "").casefold().split())
        if _BARE_NUMBER.fullmatch(label):
            end = i
            next_chapter = groups[i][0]
            break

    selected_groups = groups[start:end]
    targets = [segment for _name, rows in selected_groups for segment in rows]
    if not targets:
        raise RuntimeError(f"Chapter {chapter_name!r} selected no translatable segments")

    position = {segment.id: i for i, segment in enumerate(all_targets)}
    boundary_complete = True
    if end < len(groups):
        next_rows = groups[end][1]
        boundary_complete = bool(next_rows) and position[targets[-1].id] + 1 == position[next_rows[0].id]
    if not boundary_complete:
        raise RuntimeError(f"Chapter {chapter_name!r} is not contiguous with next numeric boundary")

    return all_targets, groups[start][0], targets, {
        "requested": chapter_name,
        "matched": groups[start][0],
        "included_groups": [name for name, _rows in selected_groups],
        "next_numbered_chapter": next_chapter or None,
        "boundary_complete": True,
        "segments": len(targets),
        "selector": "production-bare-numeric-v3",
    }


def _harden_source_only_candidates(segments) -> list[dict[str, Any]]:
    """Reject ordinary words/contraction prefixes and normalize title+name candidates."""
    rows = [dict(row) for row in _BASE_CANDIDATES(segments)]
    joined = "\n".join(str(segment.text or "") for segment in segments)
    lowercase_counts: Counter[str] = Counter(
        token.casefold()
        for token in re.findall(r"(?<![A-Za-z])([a-z][a-z'-]{2,})(?![A-Za-z])", joined)
    )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_common = 0
    rejected_contraction_prefix = 0
    stripped_titles = 0
    for row in rows:
        kind = str(row.get("kind_hint") or "")
        candidate = " ".join(str(row.get("candidate") or "").split())
        if not candidate:
            continue

        if kind == "proper":
            parts = candidate.split()
            if len(parts) >= 2 and parts[0].casefold() in _TITLE_WORDS:
                remainder = " ".join(parts[1:])
                full_count = len(re.findall(
                    rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z])", joined
                ))
                remainder_count = len(re.findall(
                    rf"(?<![A-Za-z]){re.escape(remainder)}(?![A-Za-z])", joined
                ))
                if remainder_count > full_count:
                    candidate = remainder
                    row["candidate"] = candidate
                    row["title_evidence"] = max(1, int(row.get("title_evidence") or 0))
                    stripped_titles += 1

            if " " not in candidate:
                # Do not let the `Don` part of `Don't`, `Can` of `Can't`, etc. become
                # a book-wide entity merely because a contraction starts a sentence.
                standalone_count = len(re.findall(
                    rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z'])", joined,
                    re.I,
                ))
                if standalone_count == 0:
                    rejected_contraction_prefix += 1
                    continue
                if lowercase_counts[candidate.casefold()] > 0:
                    rejected_common += 1
                    continue

        key = (kind, candidate.casefold())
        row["candidate"] = candidate
        if key not in merged:
            merged[key] = row
            continue
        current = merged[key]
        current["frequency"] = max(
            int(current.get("frequency") or 0), int(row.get("frequency") or 0)
        )
        current["mid_frequency"] = max(
            int(current.get("mid_frequency") or 0), int(row.get("mid_frequency") or 0)
        )
        current["title_evidence"] = max(
            int(current.get("title_evidence") or 0), int(row.get("title_evidence") or 0)
        )
        contexts = list(current.get("contexts") or [])
        seen = {
            (str(x.get("chapter") or ""), str(x.get("text") or ""))
            for x in contexts if isinstance(x, dict)
        }
        for ctx in row.get("contexts") or []:
            if not isinstance(ctx, dict):
                continue
            ckey = (str(ctx.get("chapter") or ""), str(ctx.get("text") or ""))
            if ckey not in seen and len(contexts) < 3:
                contexts.append(ctx)
                seen.add(ckey)
        current["contexts"] = contexts

    out = list(merged.values())
    print(
        f"[v10-production-entity-filter] input={len(rows)} output={len(out)} "
        f"rejected_lowercase_common={rejected_common} "
        f"rejected_contraction_prefix={rejected_contraction_prefix} stripped_titles={stripped_titles}",
        flush=True,
    )
    return out


def _safe_gender_issues(segment, target: str, memory) -> list[V10Issue]:
    """Hard-fail gender only when the named character is explicitly the source actor."""
    source = str(segment.text or "")
    low = str(target or "").casefold().replace("ё", "е")
    out: list[V10Issue] = []
    verbs = r"said|asked|replied|answered|thought|remarked|observed|whispered|shouted|called|added|continued"
    for name, desc in memory.characters.items():
        name_s = str(name or "").strip()
        if not name_s or not re.search(rf"(?<![A-Za-z]){re.escape(name_s)}(?![A-Za-z])", source, re.I):
            continue
        direct_actor = bool(
            re.search(rf"(?<![A-Za-z]){re.escape(name_s)}(?![A-Za-z])[^.!?]{{0,42}}\b(?:{verbs})\b", source, re.I)
            or re.search(rf"\b(?:{verbs})\b[^.!?]{{0,24}}(?<![A-Za-z]){re.escape(name_s)}(?![A-Za-z])", source, re.I)
        )
        if not direct_actor:
            continue
        gender = re.search(r"gender=(male|female)", str(desc), re.I)
        ru = re.search(r"ru=([^;]+)", str(desc), re.I)
        if not gender:
            continue
        if ru:
            canon_word = re.findall(r"[А-Яа-яЁё]+", ru.group(1))
            if canon_word:
                stem = canon_word[0].casefold().replace("ё", "е")[:max(3, len(canon_word[0]) - 2)]
                if stem and stem not in low:
                    continue
        value = gender.group(1).casefold()
        if value == "male" and re.search(r"\b(?:сказала|говорила|ответила|спросила|подумала|заметила|добавила)\b", low):
            out.append(V10Issue(segment.id, "character_gender", "local", "hard", f"{name_s} is male but direct Russian attribution is feminine"))
        if value == "female" and re.search(r"\b(?:сказал|говорил|ответил|спросил|подумал|заметил|добавил)\b", low):
            out.append(V10Issue(segment.id, "character_gender", "local", "hard", f"{name_s} is female but direct Russian attribution is masculine"))
    return out


def _clean_quantity_with_compounds(cls, source: str, target: str) -> dict[str, Any]:
    quantity = dict(_BASE_CLEAN_QUANTITY(source, target))
    low = str(target or "").casefold().replace("ё", "е")
    dimensions = {
        "two": (2, r"\b(?:двумерн|двухмерн)\w*\b"),
        "three": (3, r"\bтрехмерн\w*\b"),
        "four": (4, r"\bчетырехмерн\w*\b"),
        "five": (5, r"\bпятимерн\w*\b"),
        "six": (6, r"\bшестимерн\w*\b"),
        "seven": (7, r"\bсемимерн\w*\b"),
        "eight": (8, r"\bвосьмимерн\w*\b"),
        "nine": (9, r"\bдевятимерн\w*\b"),
        "ten": (10, r"\bдесятимерн\w*\b"),
    }
    suppress: set[int] = set()
    for en, (value, ru_pattern) in dimensions.items():
        if re.search(rf"\b{en}[- ]dimensional\b", source, re.I) and re.search(ru_pattern, low):
            suppress.add(value)
    if suppress:
        quantity["base_missing"] = [v for v in quantity.get("base_missing") or [] if v not in suppress]
        quantity["missing_mentions"] = [
            row for row in quantity.get("missing_mentions") or [] if row.get("value") not in suppress
        ]
        quantity["ok"] = not quantity["base_missing"] and not quantity["missing_mentions"] and not quantity.get("numbered_choice_missing")
    return quantity


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _install_production_hardening() -> None:
    install_v10_production_runtime_guard()
    v10_release.select_numbered_chapter = _select_numeric_or_named_chapter
    release_final.FinalBookBibleBuilder._candidate_records = staticmethod(_harden_source_only_candidates)
    DeterministicQA._gender_issues = staticmethod(_safe_gender_issues)
    general_release.FinalV10QualityQA._clean_quantity = classmethod(_clean_quantity_with_compounds)


def main() -> None:
    _install_production_hardening()
    import chapter_translation_v10_release9 as release9

    source = Path(os.getenv("BOOKAI_SOURCE") or "Evil for evil.fb2")
    if not source.exists() or source.stat().st_size < 10_000:
        raise RuntimeError(f"invalid/missing source: {source}")
    chapter = str(os.getenv("BOOKAI_CHAPTER_NAME") or "1")
    fp = _fingerprint(source)
    slug = re.sub(r"[^a-z0-9]+", "-", chapter.casefold()).strip("-") or "chapter"

    release9.SOURCE = source
    release9.MEMORY_SOURCE = source
    release9.CHAPTER_NAME = chapter
    release9.SLUG = slug
    release9.OUTPUT = Path(os.getenv("BOOKAI_V10_OUTPUT") or f"Evil_for_Evil_RU_V10_{slug}.fb2")
    release9.REPORT = Path(os.getenv("BOOKAI_V10_REPORT") or f"v10-evil-chapter-{slug}-report.json")
    release9.MAP = Path(os.getenv("BOOKAI_V10_MAP") or f"v10-evil-chapter-{slug}-map.json")
    release9.SOURCE_TXT = Path(os.getenv("BOOKAI_V10_SOURCE_TXT") or f"v10-evil-chapter-{slug}-source.txt")
    release9.TRANSLATED_TXT = Path(os.getenv("BOOKAI_V10_TRANSLATED_TXT") or f"v10-evil-chapter-{slug}-translated.txt")
    release9.BIBLE = Path(
        os.getenv("BOOKAI_V10_BIBLE_CACHE")
        or f".bookai-cache-{_PRODUCTION_HARDENING}-{fp}/book-bible.json"
    )
    release9.BIBLE.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[v10-production-wrapper] source={source} fingerprint={fp} chapter={chapter} "
        f"hardening={_PRODUCTION_HARDENING} bible={release9.BIBLE}",
        flush=True,
    )
    release9.main()


if __name__ == "__main__":
    main()
