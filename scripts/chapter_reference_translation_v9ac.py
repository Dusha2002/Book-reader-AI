from __future__ import annotations

import json
from typing import Any

import chapter_reference_translation_v9ab as v9ab

# Freeze the pre-v9ab router before v9ab.main() monkey-patches the v9aa symbol.
_BASE_AA_ROUTE = v9ab.v9aa._route_v9aa
_ORIGINAL_AB_ROUTE = v9ab._route_v9ab


def _giga_json_v9ac(system: str, payload: Any, *, max_tokens: int = 5000) -> dict[str, Any]:
    """Use GigaChat JSON mode for cheap analysis and repair malformed JSON once with GigaChat itself."""
    backend = v9ab._giga()
    client = backend._ensure_client()

    def call(messages, *, json_mode: bool = True):
        request = {
            "model": backend.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": max(1200, min(7000, int(max_tokens))),
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        try:
            response = client.chat(request)
        except Exception:
            if not json_mode:
                raise
            request.pop("response_format", None)
            response = client.chat(request)
        usage = backend._usage(response)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v9ab._GIGA_ANALYST_USAGE[key] += int(usage.get(key) or 0)
        v9ab._GIGA_ANALYST_USAGE["api_calls"] += 1
        return str(response.choices[0].message.content or "")

    messages = [
        {"role": "system", "content": system + "\nReturn one valid JSON object and no prose outside it."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw = call(messages, json_mode=True)
    try:
        return v9ab._parse_json(raw)
    except Exception:
        # One cheap same-provider syntax-repair call. It must only repair JSON syntax,
        # never reinterpret the source task, so no DeepSeek tokens are spent here.
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Repair the following malformed model output into ONE valid JSON object. "
                    "Preserve every key/value and semantic decision; only fix JSON syntax. Return JSON only."
                ),
            },
            {"role": "user", "content": raw[:28000]},
        ]
        repaired = call(repair_messages, json_mode=True)
        return v9ab._parse_json(repaired)


def _route_v9ac(targets, translated, memory):
    """Call v9ab routing while exposing the frozen v9aa baseline to it.

    v9ab intentionally replaces v9aa._route_v9aa at runtime; without this guard the
    wrapper recursively called itself. Quality processing is sequential at this layer.
    """
    current = v9ab.v9aa._route_v9aa
    v9ab.v9aa._route_v9aa = _BASE_AA_ROUTE
    try:
        return _ORIGINAL_AB_ROUTE(targets, translated, memory)
    finally:
        v9ab.v9aa._route_v9aa = current


def _parse_source_memory_v9ac(provider, sample: str):
    """Build the cheap GigaChat book bible during the normal analysis phase.

    This keeps DeepSeek out of book intelligence while making the canon/glossary
    available to Chapter One's draft instead of waiting until after it.
    """
    memory = v9ab.v6.BookMemory()
    stats = v9ab._ensure_giga_book_intelligence(memory)
    print(
        "[v9ac-book-intelligence] provider=GigaChat-3-Lightning "
        f"cache_hit={str(bool(stats.get('cache_hit'))).lower()} "
        f"glossary={stats.get('glossary', 0)} canon={stats.get('canon', 0)} "
        "deepseek_skipped=true",
        flush=True,
    )
    return memory


def main() -> None:
    # Patch module-global lookup points used by v9ab.main().
    v9ab._giga_json = _giga_json_v9ac
    v9ab._route_v9ab = _route_v9ac
    v9ab._parse_source_memory_v9ab = _parse_source_memory_v9ac
    v9ab.main()


if __name__ == "__main__":
    main()
