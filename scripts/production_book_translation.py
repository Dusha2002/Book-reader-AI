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


def _stratified_source_sample(segments, limit: int = 32000) -> str:
    """Sample the whole novel under the same fixed character budget.

    One source-only book-intelligence call is retained. The opening receives 25% of
    the budget; the rest is split across ordered windows through the book, including
    a guaranteed final window. Separator bytes are budgeted explicitly, so the tail
    can never disappear because of a final hard slice.
    """
    texts = [str(getattr(segment, "text", "") or "").strip() for segment in segments]
    texts = [text for text in texts if text]
    if not texts or limit <= 0:
        return ""

    complete = "\n".join(texts)
    if len(complete) <= limit:
        return complete

    separator = "\n\n--- SOURCE WINDOW ---\n\n"
    window_count = max(4, min(12, int(os.getenv("BOOKAI_SOURCE_SAMPLE_WINDOWS") or "8")))
    spread_windows = max(1, window_count - 1)

    # Reserve separator bytes and every spread window before spending on the head.
    separator_budget = len(separator) * spread_windows
    usable = max(0, limit - separator_budget)
    head_budget = max(1600, usable // 4)
    head_budget = min(head_budget, usable)
    spread_budget = max(0, usable - head_budget)
    per_window = spread_budget // spread_windows if spread_windows else 0

    def take_from(start: int, budget: int, used: set[int]) -> tuple[str, set[int]]:
        if budget <= 0:
            return "", used
        parts: list[str] = []
        chars = 0
        index = max(0, min(len(texts) - 1, start))
        while index < len(texts) and chars < budget:
            if index in used:
                index += 1
                continue
            text = texts[index]
            prefix = 1 if parts else 0
            remaining = budget - chars - prefix
            if remaining <= 0:
                break
            if len(text) <= remaining:
                if parts:
                    chars += 1
                parts.append(text)
                chars += len(text)
                used.add(index)
            elif remaining >= 160:
                if parts:
                    chars += 1
                parts.append(text[:remaining])
                chars += remaining
                used.add(index)
                break
            else:
                break
            index += 1
        return "\n".join(parts), used

    used: set[int] = set()
    head, used = take_from(0, head_budget, used)
    pieces = [head] if head else []
    n = len(texts)

    # Anchors are monotonic and the last one is always the final source paragraph.
    for window in range(spread_windows):
        if spread_windows == 1:
            anchor = n - 1
        else:
            fraction = (window + 1) / spread_windows
            anchor = min(n - 1, max(0, round((n - 1) * fraction)))
        start = max(0, anchor - 2)
        chunk, used = take_from(start, per_window, used)
        if not chunk and anchor not in used:
            # Very long neighboring paragraphs may consume a prior window. Starting
            # exactly at the anchor preserves coverage without increasing the budget.
            chunk, used = take_from(anchor, per_window, used)
        pieces.append(chunk)

    # A window may be empty only because its anchor was already consumed. Preserve
    # separator positions so the pre-budgeted length invariant remains conservative.
    sample = separator.join(pieces)
    if len(sample) > limit:
        raise RuntimeError(f"stratified source sample exceeded budget: {len(sample)} > {limit}")
    return sample


def configure_production_source():
    """Bind the proven production kernel to one explicit, source-isolated book."""
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
    v3._configure_modules()

    # Replace only v9ab's sampling policy: same model, one call and the same fixed
    # character budget, but context now covers the complete novel.
    def source_sample(limit: int = 32000) -> str:
        sample = _stratified_source_sample(
            list(getattr(v9ab.v6, "_SOURCE_SEGMENTS", []) or []),
            limit=limit,
        )
        print(
            f"[production-source-sample] mode=stratified windows="
            f"{os.getenv('BOOKAI_SOURCE_SAMPLE_WINDOWS') or '8'} chars={len(sample)} limit={limit}",
            flush=True,
        )
        return sample

    v9ab._source_sample = source_sample

    # v9aa/v9ab use BOOKAI_V8_SHARED_CACHE for whole-book name canon and source
    # intelligence. The source fingerprint prevents cross-book contamination.
    print(
        f"[production-source] source={source} fingerprint={fingerprint} "
        f"cache={cache_root} source_sample=stratified",
        flush=True,
    )
    return production, source, fingerprint


def main() -> None:
    production, _, _ = configure_production_source()
    production.LiteraryTranslationStrategy().run()


if __name__ == "__main__":
    main()
