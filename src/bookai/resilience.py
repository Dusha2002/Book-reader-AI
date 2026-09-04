from __future__ import annotations

from collections.abc import Callable

from .models import Segment


MapCall = Callable[[list[Segment]], dict[str, str]]


def resilient_segment_map(
    segments: list[Segment],
    call: MapCall,
    *,
    label: str,
    attempts: int = 2,
) -> dict[str, str]:
    """Recover strict id→text calls by retrying, then recursively splitting.

    Cheap models are much more likely to violate JSON/id contracts on larger
    batches than on one or two targets. We never guess missing ids. Only
    structured-output/validation failures (ValueError, including JSON decode
    errors) trigger splitting; network/rate-limit failures propagate normally so
    a provider outage cannot explode into dozens of requests.
    """
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

    if not isinstance(last_error, ValueError):
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
