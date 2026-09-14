from __future__ import annotations

import os
import random
import threading
import time

from .gigachat_runtime_guard import is_rate_limit_error


class GigaChatUltraProvider:
    """LLMProvider-compatible GigaChat 3 Ultra client for sparse specialist work.

    Ultra is the existing sparse semantic specialist. Transient 429 responses are
    retried in place with bounded exponential backoff so throttling never creates a
    new translation branch/layer or silently downgrades semantic verification.
    """

    def __init__(self, *, role: str = "gate") -> None:
        self.credentials = (os.getenv("GIGACHAT_AUTH_KEY") or "").strip()
        self.scope = (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
        self.base_url = (os.getenv("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1").strip()
        self.model = (os.getenv("BOOKAI_GIGACHAT_ULTRA_MODEL") or "GigaChat-3-Ultra").strip()
        self.role = role
        self.timeout_seconds = max(20, min(120, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_TIMEOUT") or "70")))
        self.max_tokens = max(1200, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_MAX_TOKENS") or "7000"))
        self.max_attempts = max(1, min(10, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_ATTEMPTS") or "5")))
        self.rate_limit_backoff = max(
            0.5,
            min(10.0, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF") or "2.0")),
        )
        self.rate_limit_max_sleep = max(
            self.rate_limit_backoff,
            min(30.0, float(os.getenv("BOOKAI_GIGACHAT_RATE_LIMIT_MAX_SLEEP") or "15.0")),
        )
        self._client = None
        self._lock = threading.Lock()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "cost": 0.0,
        }
        if not self.credentials:
            raise ValueError("GIGACHAT_AUTH_KEY is required for GigaChat-3-Ultra")

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from gigachat import GigaChat

        client = GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            base_url=self.base_url,
            verify_ssl_certs=False,
            timeout=self.timeout_seconds,
            max_retries=1,
            retry_backoff_factor=0.8,
        )
        token = client.get_token()
        if not str(getattr(token, "access_token", "") or ""):
            raise RuntimeError("GigaChat Ultra OAuth succeeded but access_token is empty")
        self._client = client
        return client

    @staticmethod
    def _response_usage(response) -> dict[str, int]:
        obj = getattr(response, "usage", None)
        return {
            "prompt_tokens": int(getattr(obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(obj, "total_tokens", 0) or 0),
        }

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        client = self._ensure_client()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            try:
                # Physical-person GigaChat API effectively has one request stream.
                # Serialize specialist calls instead of producing our own 429 burst.
                with self._lock:
                    response = client.chat(payload)
                content = str(response.choices[0].message.content or "").strip()
                usage = self._response_usage(response)
                with self._lock:
                    self.usage["prompt_tokens"] += usage["prompt_tokens"]
                    self.usage["completion_tokens"] += usage["completion_tokens"]
                    self.usage["total_tokens"] += usage["total_tokens"]
                    self.usage["requests"] += 1
                print(
                    f"[gigachat-ultra] role={self.role} attempt={attempt + 1} "
                    f"elapsed={time.perf_counter()-started:.2f}s tokens={usage['total_tokens']} chars={len(content)}",
                    flush=True,
                )
                if len(content) <= 2:
                    raise ValueError("GigaChat Ultra returned near-empty content")
                return content
            except Exception as exc:
                last_error = exc
                limited = is_rate_limit_error(exc)
                print(
                    f"[gigachat-ultra] role={self.role} attempt={attempt + 1}/{self.max_attempts} "
                    f"rate_limited={str(limited).lower()} error={type(exc).__name__}: {str(exc)[:220]}",
                    flush=True,
                )
                if attempt + 1 >= self.max_attempts:
                    break
                if limited:
                    delay = min(
                        self.rate_limit_max_sleep,
                        self.rate_limit_backoff * (2 ** attempt),
                    ) + random.uniform(0.05, 0.35)
                else:
                    delay = min(4.0, 0.8 * (2**attempt)) + random.uniform(0.05, 0.25)
                print(
                    f"[gigachat-ultra] role={self.role} retry_same_request sleep={delay:.2f}s",
                    flush=True,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error
