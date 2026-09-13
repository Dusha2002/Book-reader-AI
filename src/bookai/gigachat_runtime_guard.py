from __future__ import annotations

import os
import random
import time
from typing import Any

from .gigachat_mt import GigaChatLightningBackend


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True only for GigaChat throttling / HTTP 429 failures."""
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        token in text
        for token in (
            "ratelimiterror",
            "rate_limit",
            "rate limit",
            "too many requests",
            "http 429",
            "status code: 429",
            "status_code=429",
            " 429 ",
        )
    )


def install_gigachat_runtime_guard() -> None:
    """Protect the v9ah GigaChat primary path from transient PERS throttling.

    GigaChat PERS effectively gives us one request stream. The legacy resilient
    translator treats every exception as a malformed batch and recursively splits
    it. That is useful for bad JSON, but disastrous for 429: one throttled 12-item
    request becomes 6+6, then 3+3, and creates a request storm.

    This runtime guard keeps the v9ah implementation itself frozen. It retries the
    exact same request with bounded exponential backoff, never mistakes 429 for a
    response-schema incompatibility, and refuses to recursively split a batch when
    throttling is the only failure.
    """
    cls = GigaChatLightningBackend
    if getattr(cls, "_bookai_rate_guard_installed", False):
        return

    original_chat = cls._chat
    original_schema_incompatibility = cls._schema_incompatibility

    def guarded_chat(self, payload: dict, batch: list[Any], *, strict: bool) -> object:
        attempts = max(2, min(10, int(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_ATTEMPTS") or "7")))
        base = max(0.5, min(10.0, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF") or "2.0")))
        cap = max(base, min(30.0, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_MAX_SLEEP") or "15.0")))

        for attempt in range(1, attempts + 1):
            try:
                return original_chat(self, payload, batch, strict=strict)
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= attempts:
                    raise
                delay = min(cap, base * (2 ** (attempt - 1))) + random.uniform(0.05, 0.35)
                print(
                    f"[gigachat-throttle] retry_same_batch={attempt}/{attempts - 1} "
                    f"segments={len(batch)} sleep={delay:.2f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def guarded_schema_incompatibility(exc: Exception) -> bool:
        if is_rate_limit_error(exc):
            return False
        return original_schema_incompatibility(exc)

    def guarded_translate_resilient(
        self,
        batch,
        memory,
        *,
        source_segments,
        depth: int = 0,
    ):
        try:
            rows, usage = self._translate_batch(batch, memory, source_segments=source_segments)
            self.usage.add(usage, calls=1)
        except Exception as exc:
            self.usage.add({}, calls=1)
            if is_rate_limit_error(exc):
                print(
                    f"[gigachat-throttle] exhausted_without_split segments={len(batch)} depth={depth}",
                    flush=True,
                )
                return {}, {segment.id: f"{type(exc).__name__}: {exc}" for segment in batch}
            if len(batch) == 1:
                return self._single_strict_recovery(batch[0], memory, source_segments, exc)
            if depth < self.max_split_depth:
                mid = max(1, len(batch) // 2)
                left_rows, left_errors = guarded_translate_resilient(
                    self,
                    batch[:mid],
                    memory,
                    source_segments=source_segments,
                    depth=depth + 1,
                )
                right_rows, right_errors = guarded_translate_resilient(
                    self,
                    batch[mid:],
                    memory,
                    source_segments=source_segments,
                    depth=depth + 1,
                )
                return {**left_rows, **right_rows}, {**left_errors, **right_errors}
            return {}, {segment.id: f"{type(exc).__name__}: {exc}" for segment in batch}

        missing = [segment for segment in batch if segment.id not in rows]
        if not missing:
            return rows, {}
        if len(batch) == 1:
            retry_rows, retry_errors = self._single_strict_recovery(batch[0], memory, source_segments)
            rows.update(retry_rows)
            return rows, retry_errors
        if depth < self.max_split_depth:
            retry_rows, retry_errors = guarded_translate_resilient(
                self,
                missing,
                memory,
                source_segments=source_segments,
                depth=depth + 1,
            )
            rows.update(retry_rows)
            return rows, retry_errors
        return rows, {segment.id: "missing from GigaChat batch response" for segment in missing}

    cls._chat = guarded_chat
    cls._schema_incompatibility = staticmethod(guarded_schema_incompatibility)
    cls._translate_resilient = guarded_translate_resilient
    cls._bookai_rate_guard_installed = True
    print("[gigachat-throttle] runtime_guard=installed", flush=True)
