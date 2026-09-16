from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

from .gigachat_runtime_guard import is_rate_limit_error
from .gigachat_v3 import GigaChatLightningV3Backend


_LOCK = threading.Lock()
_LAST_FINISHED = 0.0
_BASE_ENSURE = GigaChatLightningV3Backend._ensure_client


class _GuardedClient:
    """Serialize PERS chat calls and retry the exact same request on HTTP 429.

    Clean v10 uses tagged/JSON calls that bypass the legacy `_chat` method, so the
    older v9 runtime guard does not protect them. This proxy sits at the client
    boundary and therefore covers Book Bible, primary translation and every Giga
    repair stage without changing their semantic behavior.
    """

    def __init__(self, raw: Any):
        self._raw = raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def chat(self, request: Any) -> Any:
        global _LAST_FINISHED
        attempts = max(2, min(10, int(os.getenv("BOOKAI_V10_GIGA_RATE_ATTEMPTS") or "7")))
        base = max(0.5, min(8.0, float(os.getenv("BOOKAI_V10_GIGA_RATE_BACKOFF") or "1.8")))
        cap = max(base, min(30.0, float(os.getenv("BOOKAI_V10_GIGA_RATE_MAX_SLEEP") or "15")))
        min_interval = max(0.0, min(5.0, float(os.getenv("BOOKAI_V10_GIGA_MIN_INTERVAL") or "0.7")))

        with _LOCK:
            now = time.monotonic()
            delay = min_interval - (now - _LAST_FINISHED)
            if delay > 0:
                time.sleep(delay)
            for attempt in range(1, attempts + 1):
                try:
                    result = self._raw.chat(request)
                    _LAST_FINISHED = time.monotonic()
                    return result
                except Exception as exc:
                    if not is_rate_limit_error(exc) or attempt >= attempts:
                        _LAST_FINISHED = time.monotonic()
                        raise
                    sleep_for = min(cap, base * (2 ** (attempt - 1))) + random.uniform(0.05, 0.25)
                    print(
                        f"[v10-giga-throttle] retry_same_request={attempt}/{attempts - 1} "
                        f"sleep={sleep_for:.2f}s",
                        flush=True,
                    )
                    time.sleep(sleep_for)
        raise AssertionError("unreachable")


def install_v10_production_runtime_guard() -> None:
    cls = GigaChatLightningV3Backend
    if getattr(cls, "_bookai_v10_production_guard_installed", False):
        return

    def guarded_ensure(self):
        proxy = getattr(self, "_bookai_v10_guarded_client", None)
        if proxy is not None:
            return proxy
        raw = _BASE_ENSURE(self)
        proxy = _GuardedClient(raw)
        setattr(self, "_bookai_v10_guarded_client", proxy)
        return proxy

    cls._ensure_client = guarded_ensure
    cls._bookai_v10_production_guard_installed = True
    print("[v10-giga-throttle] production_runtime_guard=installed", flush=True)


__all__ = ["install_v10_production_runtime_guard"]
