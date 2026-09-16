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
    """Use the same token budget, but sample the whole novel instead of only its start.

    Book intelligence is intentionally one cheap source-only model call. For large
    novels, feeding only the first ~32k characters gives it excellent opening context
    but poor awareness of late characters, factions and specialist vocabulary. This
    sampler keeps the character budget unchanged while reserving 25% for the opening
    and spreading the rest across ordered windows through the entire source.
    """
    texts = [str(getattr(segment, "text", "") or "").strip() for segment in segments]
    texts = [text for text in texts if text]
    if not texts or limit <= 0:
        return ""

    complete = "\n".join(texts)
    if len(complete) <= limit:
        return complete

    window_count = max(4, min(12, int(os.getenv("BOOKAI_SOURCE_SAMPLE_WINDOWS") or "8")))
    head_budget = max(3000, limit // 4)
    head_parts: list[str] = []
    head_chars = 0
    head_end = 0
    for index, text in enumerate(texts):
        if head_parts and head_chars + len(text) + 1 > head_budget:
            break
        remaining = head_budget - head_chars
        if len(text) > remaining and not head_parts:
            head_parts.append(text[:remaining])
            head_chars += remaining
            head_end = index + 1
            break
        head_parts.append(text)
        head_chars += len(text) + 1
        head_end = index + 1
        if head_chars >= head_budget:
            break

    pieces = ["\n".join(head_parts)] if head_parts else []
    used_indexes = set(range(head_end))
    remaining_budget = max(0, limit - sum(len(piece) for piece in pieces))
    spread_windows = max(1, window_count - 1)
    per_window = max(1200, remaining_budget // spread_windows)
    n = len(texts)

    for window in range(spread_windows):
        fraction = (window + 1) / spread_windows
        anchor = min(n - 1, max(head_end, round((n - 1) * fraction)))
        start = max(head_end, anchor - 2)
        chunk: list[str] = []
        chars = 0
        index = start
        while index < n and chars < per_window:
            if index not in used_indexes:
                text = texts[index]
                remaining = per_window - chars
                if len(text) <= remaining:
                    chunk.append(text)
                    chars += len(text) + 1
                elif remaining > 160:
                    chunk.append(text[:remaining])
                    chars += remaining
                used_indexes.add(index)
            index += 1
        if chunk:
            pieces.append("\n".join(chunk))

    sample = "\n\n--- SOURCE WINDOW ---\n\n".join(pieces)
    return sample[:limit]


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

    # Replace only v9ab's sampling policy: same model, one call and same character
    # budget, but context now covers the complete novel rather than only its opening.
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
