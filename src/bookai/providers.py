from __future__ import annotations

import os
import random
import time

import httpx

from .llm import OpenAICompatibleProvider


class CappedOpenAICompatibleProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for compact control-plane roles.

    Analyzer/gate/memory responses should be small structured JSON. Besides an
    output-token cap, these roles need their own retry/time budget: nesting an
    outer semantic retry around the global 3x180s transport retry can otherwise
    stall a chapter for tens of minutes. Translation/polish providers are not
    affected by these compact-role limits.
    """

    _DEFAULT_ATTEMPTS = {"analyzer": 1, "gate": 2, "memory": 2}
    _DEFAULT_TIMEOUTS = {"analyzer": 75.0, "gate": 90.0, "memory": 75.0}

    def __init__(
        self,
        *args,
        max_tokens: int = 4096,
        max_attempts: int | None = None,
        request_timeout: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_tokens = max(256, int(max_tokens))
        role_key = str(getattr(self, "role", "llm")).upper()
        role_attempts = os.getenv(f"BOOKAI_{role_key}_RETRY_ATTEMPTS")
        role_timeout = os.getenv(f"BOOKAI_{role_key}_REQUEST_TIMEOUT")
        default_attempts = self._DEFAULT_ATTEMPTS.get(self.role, 2)
        default_timeout = self._DEFAULT_TIMEOUTS.get(self.role, 90.0)
        self.max_attempts = max(
            1,
            int(
                max_attempts
                if max_attempts is not None
                else (role_attempts or default_attempts)
            ),
        )
        self.request_timeout = max(
            15.0,
            float(
                request_timeout
                if request_timeout is not None
                else (role_timeout or default_timeout)
            ),
        )

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if "openrouter.ai" in self.base_url:
            payload["reasoning"] = {"effort": self.reasoning_effort}
            payload["usage"] = {"include": True}
            headers["X-Title"] = "Book Reader AI"
        elif "deepseek.com" in self.base_url:
            payload["thinking"] = {"type": "enabled" if self.reasoning_effort != "none" else "disabled"}

        if self.reasoning_effort == "none":
            payload["temperature"] = temperature

        response: httpx.Response | None = None

        for attempt in range(self.max_attempts):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
                if not retryable:
                    response.raise_for_status()
                    break
                if attempt + 1 >= self.max_attempts:
                    response.raise_for_status()
                raw_retry_after = response.headers.get("Retry-After")
                try:
                    server_delay = float(raw_retry_after) if raw_retry_after else 0.0
                except ValueError:
                    server_delay = 0.0
                delay = min(20.0, max(server_delay, min(8.0, 2.0**attempt)))
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{self.max_attempts} status={response.status_code} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.6))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 >= self.max_attempts:
                    raise
                delay = min(8.0, 2.0**attempt)
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{self.max_attempts} error={type(exc).__name__} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.6))

        if response is None:
            raise RuntimeError("LLM request did not produce a response")
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        with self._usage_lock:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.usage["total_tokens"] += int(usage.get("total_tokens") or 0)
            self.usage["requests"] += 1
            self.usage["cost"] += float(usage.get("cost") or 0.0)
        return data["choices"][0]["message"]["content"]
