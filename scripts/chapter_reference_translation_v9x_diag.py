from __future__ import annotations

import json
import re

import httpx

import chapter_reference_translation_v9x as v9x
from bookai.llm import OpenAICompatibleProvider

_ORIGINAL_COMPLETE = OpenAICompatibleProvider.complete
_SECRET_RE = re.compile(r"(?i)(?:bearer\s+)?(?:sk-|or-)[A-Za-z0-9_\-]{12,}")


def _safe_detail(response: httpx.Response) -> str:
    parts = []
    try:
        data = response.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            for key in ("code", "message"):
                value = err.get(key)
                if value not in (None, ""):
                    parts.append(f"{key}={value}")
            meta = err.get("metadata")
            if isinstance(meta, dict):
                for key in ("provider_name", "provider", "raw"):
                    value = meta.get(key)
                    if value not in (None, ""):
                        parts.append(f"{key}={value}")
        elif isinstance(data, dict):
            for key in ("code", "message", "detail"):
                value = data.get(key)
                if value not in (None, ""):
                    parts.append(f"{key}={value}")
    except Exception:
        try:
            parts.append(response.text)
        except Exception:
            pass
    value = " | ".join(str(x) for x in parts) or "no parseable error body"
    value = _SECRET_RE.sub("[redacted]", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:600]


def _diagnostic_complete(self, system: str, user: str, *, temperature: float = 0.2):
    try:
        return _ORIGINAL_COMPLETE(self, system, user, temperature=temperature)
    except httpx.HTTPStatusError as exc:
        response = exc.response
        print(
            f"[bookai-http-error] role={getattr(self, 'role', 'llm')} "
            f"model={getattr(self, 'model', '')} status={response.status_code} "
            f"detail={_safe_detail(response)}",
            flush=True,
        )
        raise


OpenAICompatibleProvider.complete = _diagnostic_complete


if __name__ == "__main__":
    v9x.main()
