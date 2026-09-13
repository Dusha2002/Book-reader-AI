from __future__ import annotations

import chapter_reference_translation_v9x as v9x
from bookai.deepseek_direct import adapt_harness


def main() -> None:
    # v9x remains the translation architecture. v9y only swaps transport from
    # OpenRouter to DeepSeek's direct OpenAI-compatible API and normalizes any
    # legacy model alias to the exact direct model id: deepseek-v4.1-flash.
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
