from __future__ import annotations

import chapter_reference_translation_v9x as v9x
import bookai.reference_harness as reference_harness
from bookai.deepseek_direct import CONFIGURED_FLASH_MODEL, adapt_harness


def main() -> None:
    # v9x remains the translation architecture. v9y swaps transport to the
    # direct DeepSeek API and forces one exact model id end-to-end.
    reference_harness.FLASH_MODEL = CONFIGURED_FLASH_MODEL
    original_builder = v9x.v3.hybrid.build_reference_harness

    def build_direct_harness():
        return adapt_harness(original_builder())

    v9x.v3.hybrid.build_reference_harness = build_direct_harness
    try:
        v9x.main()
    finally:
        v9x.v3.hybrid.build_reference_harness = original_builder


if __name__ == "__main__":
    main()
