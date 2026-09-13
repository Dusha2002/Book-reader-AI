from __future__ import annotations

import os

import chapter_reference_translation_v9ah as v9ah
from bookai.frozen_draft import make_frozen_draft_backend
from bookai.gigachat_rate_limit import make_rate_limit_resilient_backend


def main() -> None:
    mode = (os.getenv("BOOKAI_FROZEN_DRAFT_MODE") or "replay").strip().casefold()
    path = os.getenv("BOOKAI_FROZEN_DRAFT_PATH") or "benchmark/frozen-primary-draft.json"
    old_backend = v9ah.v9.GigaChatLightningV9Backend

    # Record mode is the only frozen-benchmark phase that reaches GigaChat.
    # Keep the whole batch intact through transient 429 windows; otherwise the
    # legacy recursive splitter amplifies request pressure (8→4→2→1).
    base_backend = (
        make_rate_limit_resilient_backend(old_backend)
        if mode == "record"
        else old_backend
    )
    v9ah.v9.GigaChatLightningV9Backend = make_frozen_draft_backend(
        base_backend,
        path=path,
        mode=mode,
    )
    try:
        v9ah.main()
    finally:
        v9ah.v9.GigaChatLightningV9Backend = old_backend


if __name__ == "__main__":
    main()
