from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _source() -> Path:
    source = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
    if not source.exists():
        raise FileNotFoundError(source)
    return source


def _source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def configure_production_source():
    """Bind the legacy kernel to one explicit source book before production runs.

    The stable production facade still reuses the proven v9ah kernel. Older modules
    were historically benchmark-specific, so this wrapper makes their source and
    shared book-intelligence caches source-specific without changing translation
    policy or adding any model layer.
    """
    source = _source()
    fingerprint = _source_fingerprint(source)

    cache_base = (
        os.getenv("BOOKAI_V8_SHARED_CACHE_BASE")
        or os.getenv("BOOKAI_V8_SHARED_CACHE")
        or ".bookai-cache-production-literary-v1"
    )
    cache_root = Path(f"{cache_base}-{fingerprint}")
    os.environ["BOOKAI_V8_SHARED_CACHE"] = str(cache_root)

    # Import only after the source-specific cache namespace is established.
    import production_literary_translation as production

    v9ab = production.legacy_kernel.v9ag.v9ad.v9ac.v9ab
    v3 = v9ab.v3

    v3.SOURCE = source
    v3.CACHE = Path(
        f".bookai-cache-production-chapter-{fingerprint}-{v3.CHAPTER_SLUG}"
    )
    v3.OUTPUT = Path(
        os.getenv("BOOKAI_CHAPTER_OUTPUT")
        or f"{source.stem}_RU_CHAPTER_EVAL{source.suffix.lower()}"
    )
    # Propagate the new source/output/cache into hybrid/full-reference aliases.
    v3._configure_modules()

    # v9aa/v9ab use BOOKAI_V8_SHARED_CACHE for whole-book name canon and source
    # intelligence. Since that root now contains the source fingerprint, cache from
    # one novel can never contaminate another novel.
    print(
        f"[production-source] source={source} fingerprint={fingerprint} "
        f"cache={cache_root}",
        flush=True,
    )
    return production, source, fingerprint


def main() -> None:
    production, _, _ = configure_production_source()
    production.LiteraryTranslationStrategy().run()


if __name__ == "__main__":
    main()
