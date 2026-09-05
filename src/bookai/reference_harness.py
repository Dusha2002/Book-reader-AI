from __future__ import annotations

import os

from .harness import LLMTranslator, TranslationHarness
from .llm import OpenAICompatibleProvider
from .models import GateFinding
from .providers import CappedOpenAICompatibleProvider
from .reference_profile import apply_reference_profile


FLASH_MODEL = "deepseek/deepseek-v4-flash-0731"
PRO_MODEL = "deepseek/deepseek-v4-pro-0813"


class ReferenceTranslationHarness(TranslationHarness):
    """Quality-first harness calibrated against the user-provided literary reference.

    Flash handles cheap/high-volume roles. V4 Pro is reserved for mandatory
    literary polish and difficult repairs. Literary-critic style judgements are
    repair signals, not publication vetoes; semantic/deterministic hard failures
    remain blockers, and the independent reference judge decides benchmark fit.
    """

    _ADVISORY_LITERARY_PREFIXES = (
        "calque:",
        "rhythm:",
        "voice:",
        "irony:",
        "idiom:",
        "dialogue:",
    )

    def analyze(self, sample: str):
        return apply_reference_profile(super().analyze(sample))

    def update_memory(self, originals, translated, memory):
        return apply_reference_profile(super().update_memory(originals, translated, memory))

    def gate_findings(self, originals, draft, memory):
        findings = super().gate_findings(originals, draft, memory)
        out: list[GateFinding] = []
        for finding in findings:
            reason = (finding.reason or "").strip().lower()
            if finding.severity == "hard" and reason.startswith(self._ADVISORY_LITERARY_PREFIXES):
                out.append(GateFinding(finding.id, "medium", finding.reason))
            else:
                out.append(finding)
        return out


def build_reference_harness() -> ReferenceTranslationHarness:
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("BOOKAI_API_KEY")
    base = os.getenv("BOOKAI_BASE_URL") or "https://openrouter.ai/api/v1"
    reasoning = os.getenv("BOOKAI_REASONING") or "none"
    if not key:
        raise ValueError("OPENROUTER_API_KEY or BOOKAI_API_KEY is required")

    flash = (os.getenv("BOOKAI_TRANSLATOR_MODEL") or FLASH_MODEL).strip()
    analyzer_model = (os.getenv("BOOKAI_ANALYZER_MODEL") or flash).strip()
    gate_model = (os.getenv("BOOKAI_GATE_MODEL") or flash).strip()
    memory_model = (os.getenv("BOOKAI_MEMORY_MODEL") or flash).strip()
    editor_model = (os.getenv("BOOKAI_EDITOR_MODEL") or PRO_MODEL).strip()
    hard_model = (os.getenv("BOOKAI_HARD_MODEL") or editor_model).strip()

    def normal(model: str, role: str) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(key, base, model, reasoning, role=role)

    def capped(model: str, role: str, default_cap: int) -> CappedOpenAICompatibleProvider:
        cap = int(os.getenv(f"BOOKAI_{role.upper()}_MAX_TOKENS") or default_cap)
        return CappedOpenAICompatibleProvider(
            key,
            base,
            model,
            reasoning,
            role=role,
            max_tokens=cap,
        )

    translator_provider = normal(flash, "translator")
    return ReferenceTranslationHarness(
        analyzer=capped(analyzer_model, "analyzer", 4096),
        translator=LLMTranslator(translator_provider),
        gate=capped(gate_model, "gate", 8192),
        editor=normal(editor_model, "editor"),
        hard_editor=normal(hard_model, "hard_editor"),
        memory_model=capped(memory_model, "memory", 4096),
        use_llm_gate=(os.getenv("BOOKAI_LLM_GATE", "true").lower() not in {"0", "false", "no"}),
    )
