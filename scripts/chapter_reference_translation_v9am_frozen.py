from __future__ import annotations

# Compatibility entrypoint kept for the existing frozen benchmark workflow.
# The active experiment is v9an: v9am fidelity contracts plus deterministic
# numbered-address cleanup. Keeping this shim avoids touching the long benchmark
# workflow just to advance the experimental implementation.

import chapter_reference_translation_v9an_frozen as v9an_frozen


def main() -> None:
    v9an_frozen.main()


if __name__ == "__main__":
    main()
