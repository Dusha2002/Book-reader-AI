from __future__ import annotations

import json
import re
from pathlib import Path

from bookai.harness import TranslationHarness
from bookai.pipeline import _cache_path, _context_for, _memory_from_dict, _should_translate
from bookai.parsers.base import load_book, save_book

SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU.fb2")
CACHE_DIR = Path(".bookai-cache")


def looks_untranslated(original: str, candidate: str) -> bool:
    candidate = (candidate or "").strip()
    if not candidate:
        return True
    latin = len(re.findall(r"[A-Za-z]", candidate))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", candidate))
    if candidate == original.strip() and latin >= 3 and cyrillic == 0:
        return True
    if latin >= 6 and cyrillic < max(2, latin // 5):
        return True
    return False


def persist(path: Path, state: dict, translations: dict[str, str]) -> None:
    state["translations"] = translations
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    document = load_book(SOURCE)
    source_segments = [s for s in document.segments if _should_translate(s.text)]

    state_path = _cache_path(SOURCE, CACHE_DIR, "optimal")
    if not state_path.exists():
        raise RuntimeError(f"Recovery state not found: {state_path}")
    state = json.loads(state_path.read_text("utf-8"))
    memory = _memory_from_dict(state.get("memory") or {})
    translations: dict[str, str] = dict(state.get("translations") or {})

    remaining = [
        s for s in source_segments
        if s.id not in translations or looks_untranslated(s.text, translations.get(s.id, ""))
    ]
    print(f"[bookai-final-repair] initial_remaining={len(remaining)} total={len(source_segments)}", flush=True)

    harness = TranslationHarness.from_env()
    failed: list[str] = []

    for index, segment in enumerate(remaining, 1):
        candidate = ""
        before, after = _context_for(source_segments, [segment])

        # One segment at a time. Malformed model JSON must never abort the whole book.
        for attempt in range(2):
            try:
                result = harness.translate([segment], memory, context_before=before, context_after=after)
                candidate = (result.get(segment.id) or "").strip()
            except Exception as exc:
                print(
                    f"[bookai-final-repair] flash_error id={segment.id} attempt={attempt + 1}/2 "
                    f"error={type(exc).__name__}",
                    flush=True,
                )
                # Invalid JSON is unlikely to improve by hammering the same role; try once more,
                # then fall through to the stronger model.
                continue
            if not looks_untranslated(segment.text, candidate):
                break
            print(
                f"[bookai-final-repair] flash_retry id={segment.id} attempt={attempt + 1}/2",
                flush=True,
            )

        # Stubborn or malformed Flash responses go through V4 Pro one segment at a time.
        if looks_untranslated(segment.text, candidate):
            draft = {segment.id: candidate or segment.text}
            for attempt in range(3):
                try:
                    result = harness.hard_edit([segment], draft, memory)
                    candidate = (result.get(segment.id) or "").strip()
                except Exception as exc:
                    print(
                        f"[bookai-final-repair] pro_error id={segment.id} attempt={attempt + 1}/3 "
                        f"error={type(exc).__name__}",
                        flush=True,
                    )
                    continue
                if not looks_untranslated(segment.text, candidate):
                    break
                draft[segment.id] = candidate or segment.text
                print(
                    f"[bookai-final-repair] pro_retry id={segment.id} attempt={attempt + 1}/3",
                    flush=True,
                )

        if looks_untranslated(segment.text, candidate):
            failed.append(segment.id)
            print(
                f"[bookai-final-repair] unresolved={len(failed)} id={segment.id}; continuing",
                flush=True,
            )
            continue

        translations[segment.id] = candidate
        persist(state_path, state, translations)
        print(
            f"[bookai-final-repair] repaired={index}/{len(remaining)} id={segment.id}",
            flush=True,
        )

    final_remaining = [
        s for s in source_segments
        if s.id not in translations or looks_untranslated(s.text, translations.get(s.id, ""))
    ]
    persist(state_path, state, translations)
    if final_remaining:
        ids = ", ".join(s.id for s in final_remaining[:20])
        raise RuntimeError(f"Untranslated segments remain ({len(final_remaining)}): {ids}")

    save_book(document, translations, OUTPUT)
    print(
        f"[bookai-final-repair] done translated={len(source_segments)} output={OUTPUT} bytes={OUTPUT.stat().st_size}",
        flush=True,
    )
    print(f"[bookai-final-repair] usage={harness.usage}", flush=True)


if __name__ == "__main__":
    main()
