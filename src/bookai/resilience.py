from __future__ import annotations

from collections.abc import Callable

from .models import GateFinding, Segment


MapCall = Callable[[list[Segment]], dict[str, str]]
FindingCall = Callable[[list[Segment]], list[GateFinding]]


def _structured_failure(exc: BaseException | None) -> bool:
    # JSON/schema/id-contract errors are deterministic response-shape failures.
    # TypeError is included because older critic parsers may encounter model
    # variants such as {"issues": 0}; retries/splitting are safe and bounded.
    return isinstance(exc, (ValueError, TypeError))


def resilient_segment_map(
    segments: list[Segment],
    call: MapCall,
    *,
    label: str,
    attempts: int = 2,
) -> dict[str, str]:
    """Recover strict id→text calls by retrying, then recursively splitting."""
    if not segments:
        return {}

    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return call(segments)
        except BaseException as exc:
            last_error = exc
            print(
                f"[bookai-structured-retry] role={label} size={len(segments)} "
                f"attempt={attempt + 1}/{max(1, attempts)} error={type(exc).__name__}",
                flush=True,
            )

    if not _structured_failure(last_error):
        raise RuntimeError(f"{label} failed due to provider/transport error") from last_error

    if len(segments) == 1:
        raise RuntimeError(f"{label} failed strict structured output for {segments[0].id}") from last_error

    mid = len(segments) // 2
    left = segments[:mid]
    right = segments[mid:]
    print(
        f"[bookai-batch-split] role={label} size={len(segments)} -> {len(left)}+{len(right)}",
        flush=True,
    )
    out = resilient_segment_map(left, call, label=label, attempts=attempts)
    out.update(resilient_segment_map(right, call, label=label, attempts=attempts))
    return out


def resilient_findings(
    segments: list[Segment],
    call: FindingCall,
    *,
    label: str,
    attempts: int = 2,
) -> list[GateFinding]:
    """Recover critic JSON/schema failures without ever silently skipping QA.

    A malformed critic response is retried on the same batch, then the batch is
    split so the cheap model has a simpler contract. A persistent single-segment
    failure is fatal: quality mode fails closed instead of assuming "no issues".
    Transport/provider failures also propagate rather than fan out.
    """
    if not segments:
        return []

    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return call(segments)
        except BaseException as exc:
            last_error = exc
            print(
                f"[bookai-critic-retry] role={label} size={len(segments)} "
                f"attempt={attempt + 1}/{max(1, attempts)} error={type(exc).__name__}",
                flush=True,
            )

    if not _structured_failure(last_error):
        raise RuntimeError(f"{label} failed due to provider/transport error") from last_error

    if len(segments) == 1:
        raise RuntimeError(f"{label} failed strict QA contract for {segments[0].id}") from last_error

    mid = len(segments) // 2
    left = segments[:mid]
    right = segments[mid:]
    print(
        f"[bookai-critic-split] role={label} size={len(segments)} -> {len(left)}+{len(right)}",
        flush=True,
    )
    return (
        resilient_findings(left, call, label=label, attempts=attempts)
        + resilient_findings(right, call, label=label, attempts=attempts)
    )
