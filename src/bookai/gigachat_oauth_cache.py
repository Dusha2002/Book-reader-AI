from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_CLIENT_LOCK = threading.Lock()
_SHARED_CLIENTS: dict[tuple[str, str, str, str], Any] = {}
_FAILURE_UNTIL: dict[tuple[str, str, str, str], float] = {}


def _credential_fingerprint(credentials: str) -> str:
    return hashlib.sha256(credentials.encode("utf-8")).hexdigest()


def _cache_path() -> Path | None:
    raw = (os.getenv("BOOKAI_GIGACHAT_TOKEN_CACHE") or "").strip()
    return Path(raw) if raw else None


def _load_cached_token(path: Path | None, fingerprint: str) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("credential_fingerprint") != fingerprint:
            return None
        saved_at = float(payload.get("saved_at") or 0.0)
        ttl = max(60.0, min(1740.0, float(os.getenv("BOOKAI_GIGACHAT_TOKEN_CACHE_TTL") or "1440")))
        if time.time() - saved_at >= ttl:
            return None
        token = str(payload.get("access_token") or "").strip()
        return token or None
    except Exception:
        return None


def _save_cached_token(path: Path | None, fingerprint: str, access_token: str) -> None:
    if path is None or not access_token:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "access_token": access_token,
                    "credential_fingerprint": fingerprint,
                    "saved_at": time.time(),
                },
                ensure_ascii=False,
            ),
            "utf-8",
        )
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception as exc:
        print(f"[gigachat-client] token_cache_write_skipped error={type(exc).__name__}", flush=True)


def ensure_cached_client(backend: Any) -> Any:
    """Return one shared GigaChat client and reuse OAuth across chapter processes.

    The SDK accepts both an access_token and OAuth credentials. A cached token skips
    the initial OAuth round-trip; if the token has expired, the SDK can still refresh
    through credentials on a 401. No token value is ever printed.
    """
    existing = getattr(backend, "_client", None)
    if existing is not None:
        return existing

    credentials = str(getattr(backend, "credentials", "") or "").strip()
    if not credentials:
        raise RuntimeError("GIGACHAT_AUTH_KEY is not configured")

    scope = str(getattr(backend, "scope", "GIGACHAT_API_PERS") or "GIGACHAT_API_PERS")
    base_url = str(getattr(backend, "base_url", "https://api.giga.chat/v1") or "https://api.giga.chat/v1")
    model = str(getattr(backend, "model", "GigaChat-3-Lightning") or "GigaChat-3-Lightning")
    fingerprint = _credential_fingerprint(credentials)
    key = (scope, base_url, model, fingerprint)

    with _CLIENT_LOCK:
        shared = _SHARED_CLIENTS.get(key)
        if shared is not None:
            backend._client = shared
            print("[gigachat-client] shared_client_hit", flush=True)
            return shared

        wait_until = _FAILURE_UNTIL.get(key, 0.0)
        remaining = wait_until - time.monotonic()
        if remaining > 0:
            print(f"[gigachat-client] oauth_backoff wait={remaining:.2f}s", flush=True)
            time.sleep(remaining)

        from gigachat import GigaChat

        timeout_seconds = int(getattr(backend, "timeout_seconds", 55) or 55)
        max_retries = int(getattr(backend, "max_retries", 1) or 0)
        retry_backoff = float(getattr(backend, "retry_backoff", 0.8) or 0.8)
        cached_token = _load_cached_token(_cache_path(), fingerprint)

        client_kwargs = dict(
            credentials=credentials,
            scope=scope,
            base_url=base_url,
            verify_ssl_certs=False,
            timeout=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff,
        )
        if cached_token:
            client_kwargs["access_token"] = cached_token
            client = GigaChat(**client_kwargs)
            print(f"[gigachat-client] token_cache_hit model={model}", flush=True)
        else:
            attempts = max(1, min(3, int(os.getenv("BOOKAI_GIGACHAT_OAUTH_ATTEMPTS") or "2")))
            last_error: Exception | None = None
            client = None
            for attempt in range(1, attempts + 1):
                print(
                    f"[gigachat-client] oauth model={model} timeout={timeout_seconds}s attempt={attempt}/{attempts}",
                    flush=True,
                )
                started = time.perf_counter()
                candidate = GigaChat(**client_kwargs)
                try:
                    token = candidate.get_token()
                    access_token = str(getattr(token, "access_token", "") or "").strip()
                    if not access_token:
                        raise RuntimeError("GigaChat OAuth succeeded but access_token is empty")
                    _save_cached_token(_cache_path(), fingerprint, access_token)
                    client = candidate
                    print(f"[gigachat-client] oauth_ok elapsed={time.perf_counter()-started:.2f}s", flush=True)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[gigachat-client] oauth_error attempt={attempt}/{attempts} error={type(exc).__name__}",
                        flush=True,
                    )
                    if attempt < attempts:
                        time.sleep(min(5.0, retry_backoff * (2 ** (attempt - 1))))
            if client is None:
                _FAILURE_UNTIL[key] = time.monotonic() + max(
                    2.0,
                    min(15.0, float(os.getenv("BOOKAI_GIGACHAT_OAUTH_FAILURE_BACKOFF") or "5")),
                )
                assert last_error is not None
                raise last_error

        if bool(getattr(backend, "validate_model", False)):
            available: list[str] = []
            try:
                models = client.get_models()
                for row in getattr(models, "data", []) or []:
                    name = getattr(row, "id_", None) or getattr(row, "id", None) or getattr(row, "name", None)
                    if name:
                        available.append(str(name))
            except Exception as exc:
                print(f"[gigachat-client] model_list_skipped error={type(exc).__name__}", flush=True)
            if available and model not in available:
                raise RuntimeError(f"Requested GigaChat model {model!r} is unavailable; available={available}")

        _FAILURE_UNTIL.pop(key, None)
        _SHARED_CLIENTS[key] = client
        backend._client = client
        return client
