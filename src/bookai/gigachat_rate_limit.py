from __future__ import annotations

import os
import time
from typing import TypeVar


T = TypeVar("T")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(token in text for token in ("ratelimit", "rate limit", "too many requests", " 429 ", "status\":429", "status': 429"))


def make_rate_limit_resilient_backend(base_cls: type[T]) -> type[T]:
    """Retry the SAME GigaChat request on 429 before any recursive batch split.

    Older translation backends treat every exception as evidence that a batch is
    too large and immediately split 12→6→3→1. That is correct for malformed or
    size-sensitive responses, but actively harmful for 429: it multiplies request
    pressure while the upstream quota window is still closed.

    This wrapper keeps batch semantics unchanged and intercepts only rate-limit
    failures at the transport boundary. Non-429 failures still flow to the base
    backend's existing recovery/split logic.
    """

    class RateLimitResilientBackend(base_cls):  # type: ignore[misc, valid-type]
        name = f"{getattr(base_cls, 'name', base_cls.__name__)}-rate-limit-resilient"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rate_limit_failed_calls = 0
            self._rate_limit_exhausted_batches = 0

        def _chat(self, payload, batch, *, strict):
            attempts = max(1, int(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_ATTEMPTS") or "6"))
            base_sleep = max(0.0, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF") or "2.0"))
            max_sleep = max(base_sleep, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_MAX_SLEEP") or "15.0"))

            for attempt in range(1, attempts + 1):
                try:
                    return super()._chat(payload, batch, strict=strict)
                except Exception as exc:
                    if not _is_rate_limit_error(exc):
                        raise
                    self._rate_limit_failed_calls += 1
                    if attempt >= attempts:
                        # The base backend will count this failed logical batch
                        # once when the exception propagates into its split logic.
                        self._rate_limit_exhausted_batches += 1
                        print(
                            f"[gigachat-rate-limit] exhausted=true attempts={attempts} "
                            f"segments={len(batch)} action=delegate_to_existing_recovery",
                            flush=True,
                        )
                        raise
                    sleep_seconds = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                    print(
                        f"[gigachat-rate-limit] attempt={attempt}/{attempts} "
                        f"segments={len(batch)} action=retry_same_batch sleep={sleep_seconds:.2f}s",
                        flush=True,
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

            raise RuntimeError("unreachable rate-limit retry state")

        def translate_many(self, *args, **kwargs):
            before_failed = int(self._rate_limit_failed_calls)
            before_exhausted = int(self._rate_limit_exhausted_batches)
            try:
                return super().translate_many(*args, **kwargs)
            finally:
                # Successful logical batches already count their final successful
                # physical call in the base usage counter, so every preceding 429
                # is extra. Exhausted logical batches are themselves counted once
                # by the base recovery path; subtract that final failed call to
                # avoid double-counting physical traffic.
                failed_delta = int(self._rate_limit_failed_calls) - before_failed
                exhausted_delta = int(self._rate_limit_exhausted_batches) - before_exhausted
                extra = max(0, failed_delta - exhausted_delta)
                usage = getattr(self, "usage", None)
                if extra > 0 and usage is not None and hasattr(usage, "api_calls"):
                    usage.api_calls += extra

    RateLimitResilientBackend.__name__ = f"RateLimitResilient{base_cls.__name__}"
    RateLimitResilientBackend.__qualname__ = RateLimitResilientBackend.__name__
    return RateLimitResilientBackend
