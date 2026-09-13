from __future__ import annotations

from dataclasses import dataclass

from bookai.gigachat_rate_limit import make_rate_limit_resilient_backend


class FakeRateLimitError(RuntimeError):
    pass


@dataclass
class Usage:
    api_calls: int = 0


class DummyBackend:
    name = "dummy"

    def __init__(self):
        self.chat_calls = 0
        self.usage = Usage()

    def _chat(self, payload, batch, *, strict):
        self.chat_calls += 1
        if self.chat_calls <= 2:
            raise FakeRateLimitError('429 Too Many Requests')
        return "ok"

    def translate_many(self, *args, **kwargs):
        try:
            value = self._chat({}, [1, 2, 3], strict=True)
            self.usage.api_calls += 1
            return {"value": value}, {}
        except Exception:
            self.usage.api_calls += 1
            raise


def test_rate_limit_retries_same_request_before_returning(monkeypatch):
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_ATTEMPTS", "4")
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF", "0")
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_MAX_SLEEP", "0")
    Backend = make_rate_limit_resilient_backend(DummyBackend)
    backend = Backend()

    rows, errors = backend.translate_many()

    assert errors == {}
    assert rows == {"value": "ok"}
    assert backend.chat_calls == 3
    assert backend.usage.api_calls == 3


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    class Broken(DummyBackend):
        def _chat(self, payload, batch, *, strict):
            self.chat_calls += 1
            raise ValueError("bad schema")

    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_ATTEMPTS", "5")
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF", "0")
    Backend = make_rate_limit_resilient_backend(Broken)
    backend = Backend()

    try:
        backend.translate_many()
    except ValueError:
        pass
    else:
        raise AssertionError("non-rate-limit error must propagate")

    assert backend.chat_calls == 1
    assert backend.usage.api_calls == 1


def test_rate_limit_exhaustion_delegates_after_configured_attempts(monkeypatch):
    class AlwaysLimited(DummyBackend):
        def _chat(self, payload, batch, *, strict):
            self.chat_calls += 1
            raise FakeRateLimitError("status:429")

    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_ATTEMPTS", "3")
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_BACKOFF", "0")
    monkeypatch.setenv("BOOKAI_GIGACHAT_RATE_LIMIT_MAX_SLEEP", "0")
    Backend = make_rate_limit_resilient_backend(AlwaysLimited)
    backend = Backend()

    try:
        backend.translate_many()
    except FakeRateLimitError:
        pass
    else:
        raise AssertionError("exhausted rate limit must propagate")

    assert backend.chat_calls == 3
    assert backend.usage.api_calls == 3
