from __future__ import annotations

from typing import Any


# Official model id accepted by https://api.deepseek.com.
CONFIGURED_FLASH_MODEL = "deepseek-v4-flash"
DIRECT_FLASH_MODEL = "deepseek-v4-flash"


def direct_model_id(configured_model: str, base_url: str) -> str:
    """Normalize legacy/OpenRouter-style aliases to the official direct DeepSeek id."""
    if "api.deepseek.com" not in (base_url or ""):
        return configured_model
    aliases = {
        "deepseek-v4-flash": DIRECT_FLASH_MODEL,
        "deepseek-v4.1-flash": DIRECT_FLASH_MODEL,
        "deepseek/deepseek-v4.1-flash": DIRECT_FLASH_MODEL,
        "deepseek/deepseek-v4-flash-0731": DIRECT_FLASH_MODEL,
        "deepseek/deepseek-v4.1-flash-0731": DIRECT_FLASH_MODEL,
    }
    return aliases.get(configured_model, configured_model)


def adapt_provider(provider: Any) -> Any:
    """Adapt one OpenAI-compatible provider for direct DeepSeek transport."""
    if provider is None:
        return provider
    configured = str(getattr(provider, "model", "") or "")
    base_url = str(getattr(provider, "base_url", "") or "")
    wire = direct_model_id(configured, base_url)
    if wire != configured:
        provider.configured_model = configured
        provider.model = wire
    return provider


def adapt_harness(harness: Any) -> Any:
    """Normalize every API role in a TranslationHarness to direct DeepSeek."""
    providers = [
        getattr(harness, "analyzer", None),
        getattr(harness, "gate", None),
        getattr(harness, "editor", None),
        getattr(harness, "hard_editor", None),
        getattr(harness, "memory_model", None),
    ]
    translator = getattr(harness, "translator", None)
    translator_provider = getattr(translator, "provider", None)
    if translator_provider is not None:
        providers.append(translator_provider)

    seen: set[int] = set()
    for provider in providers:
        if provider is None or id(provider) in seen:
            continue
        seen.add(id(provider))
        adapt_provider(provider)

    if translator is not None and translator_provider is not None:
        translator.name = str(getattr(translator_provider, "model", CONFIGURED_FLASH_MODEL))
    return harness
