from __future__ import annotations

"""Compatibility entrypoint.

Production no longer evolves through version-suffixed wrappers. Keep this file so
old commands remain valid, but delegate all current behavior to the stable strategy.
"""

from production_literary_translation import main


if __name__ == "__main__":
    main()
