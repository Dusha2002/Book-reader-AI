from __future__ import annotations

import os
import random
import time

import httpx

from .llm import OpenAICompatibleProvider


class CappedOpenAICompatibleProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider with a hard output-token budget.

    Analyzer/gate/memory roles should return compact structured JSON. A small
    model occasionally ignores that contract and emits hundreds of thousands of
    characters. Capping those roles prevents one malformed response from
    consuming minutes and a large amount of quota.
    """

    def __init__(self, *args, max_tokens: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max(256, int(max_tokens))

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

        max_attempts = max(1, int(os.getenv("BOOKAI_RETRY_ATTEMPTS") or "3"))
        request_timeout = max(30.0, float(os.getenv("BOOKAI_REQUEST_TIMEOUT") or "180"))
        response: httpx.Response | None = None

        for attempt in range(max_attempts):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
                retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
                if not retryable:
                    response.raise_for_status()
                    break
                if attempt + 1 >= max_attempts:
                    response.raise_for_status()
                raw_retry_after = response.headers.get("Retry-After")
                try:
                    server_delay = float(raw_retry_after) if raw_retry_after else 0.0
                except ValueError:
                    server_delay = 0.0
                delay = min(45.0, max(server_delay, min(20.0, 2.0**attempt)))
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{max_attempts} status={response.status_code} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.8))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt + 1 >= max_attempts:
                    raise
                delay = min(20.0, 2.0**attempt)
                print(
                    f"[bookai-retry] role={self.role} model={self.model} "
                    f"attempt={attempt + 1}/{max_attempts} error={type(exc).__name__} sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay + random.uniform(0.15, 0.8))

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
