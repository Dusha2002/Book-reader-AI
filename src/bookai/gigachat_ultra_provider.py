from __future__ import annotations

import os
import random
import threading
import time


class GigaChatUltraProvider:
    """LLMProvider-compatible GigaChat 3 Ultra client for sparse specialist work.

    The production v9ah baseline remains unchanged. This provider is used only by
    the Ultra A/B wrapper, where it replaces the DeepSeek gate/specialist role.
    Prompts already demand JSON, so we intentionally avoid relying on structured
    response_format support in Freemium and parse the returned text upstream.
    """

    def __init__(self, *, role: str = "gate") -> None:
        self.credentials = (os.getenv("GIGACHAT_AUTH_KEY") or "").strip()
        self.scope = (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
        self.base_url = (os.getenv("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1").strip()
        self.model = (os.getenv("BOOKAI_GIGACHAT_ULTRA_MODEL") or "GigaChat-3-Ultra").strip()
        self.role = role
        self.timeout_seconds = max(20, min(120, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_TIMEOUT") or "70")))
        self.max_tokens = max(1200, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_MAX_TOKENS") or "7000"))
        self.max_attempts = max(1, min(3, int(os.getenv("BOOKAI_GIGACHAT_ULTRA_ATTEMPTS") or "2")))
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
                # Physical-person GigaChat API has one stream. Serialize Ultra
                # specialist calls explicitly instead of creating accidental 429s.
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
                # Treat the same pathological near-empty answer class that hurt
                # Lightning batching as retryable before upstream JSON parsing.
                if len(content) <= 2:
                    raise ValueError("GigaChat Ultra returned near-empty content")
                return content
            except Exception as exc:
                last_error = exc
                print(
                    f"[gigachat-ultra] role={self.role} attempt={attempt + 1}/{self.max_attempts} "
                    f"error={type(exc).__name__}: {str(exc)[:220]}",
                    flush=True,
                )
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(4.0, 0.8 * (2**attempt)) + random.uniform(0.05, 0.25))
        assert last_error is not None
        raise last_error
