from __future__ import annotations

import os

import chapter_reference_translation_v9am_numeric_fast as v9am
from bookai.frozen_draft import make_frozen_draft_backend


def main() -> None:
    path = os.getenv("BOOKAI_FROZEN_DRAFT_PATH") or "benchmark/frozen-primary-draft.json"
    old_backend = v9am.v9al.v9ak.GigaChatLightningV9AKBackend
    v9am.v9al.v9ak.GigaChatLightningV9AKBackend = make_frozen_draft_backend(
        old_backend,
        path=path,
        mode="replay",
    )
    try:
        v9am.main()
    finally:
        v9am.v9al.v9ak.GigaChatLightningV9AKBackend = old_backend


if __name__ == "__main__":
    main()
