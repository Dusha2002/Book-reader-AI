from __future__ import annotations

import json
from pathlib import Path

from bookai.pipeline import translate_book
from bookai.reference_harness import build_reference_harness


SOURCE = Path("Devices_and_Desires.fb2")
OUTPUT = Path("Devices_and_Desires_RU_REFERENCE.fb2")
CACHE = Path(".bookai-cache-reference-v10")


def progress(event: dict) -> None:
    print("[bookai-progress] " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    harness = build_reference_harness()
    result = translate_book(
        SOURCE,
        OUTPUT,
        harness,
        mode="optimal",
        cache_dir=CACHE,
        progress=progress,
        memory_updates=True,
    )
    print(f"[full-reference] output={result} bytes={result.stat().st_size}", flush=True)
    print("[full-reference] usage=" + json.dumps(harness.usage, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
