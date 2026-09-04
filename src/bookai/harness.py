from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .llm import (
    OpenAICompatibleProvider,
    analyze_memory,
    edit_batch,
    qa_batch,
    quality_gate_batch,
    translate_batch,
    update_memory,
)
from .models import BookMemory, GateFinding, LLMProvider, Segment, SegmentTranslator


class LLMTranslator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.name = getattr(provider, "model", "llm")

    def translate(self, segments, memory, *, context_before=None, context_after=None):
        return translate_batch(
            self.provider,
            segments,
            memory,
            context_before=context_before,
            context_after=context_after,
        )


class MadladTranslator:
    """Optional local MADLAD-400 translator. Install with: pip install -e '.[local]'"""

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int = 8):
        self.model_name = model_name or os.getenv("BOOKAI_MADLAD_MODEL") or "google/madlad400-3b-mt"
        self.device = device or os.getenv("BOOKAI_LOCAL_DEVICE") or "auto"
        self.batch_size = batch_size
        self.name = f"local:{self.model_name}"
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Local MADLAD requires the 'local' extra: pip install -e '.[local]'") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = {"torch_dtype": dtype}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs)
        if self.device not in {"auto", "cpu"} and not hasattr(self._model, "hf_device_map"):
            self._model.to(self.device)

    def translate(self, segments, memory, *, context_before=None, context_after=None):
        self._load()
        assert self._tokenizer is not None and self._model is not None
        out: dict[str, str] = {}
        for start in range(0, len(segments), self.batch_size):
            chunk = segments[start : start + self.batch_size]
            texts = [f"<2ru> {s.text}" for s in chunk]
            encoded = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            device = next(self._model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            generated = self._model.generate(**encoded, max_new_tokens=1536, num_beams=1)
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            out.update({s.id: text.strip() for s, text in zip(chunk, decoded)})
        return out


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def heuristic_findings(originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> list[GateFinding]:
    findings: dict[str, GateFinding] = {}
    for s in originals:
        ru = draft.get(s.id, "").strip()
        en = s.text.strip()
        reason = ""
        severity = "medium"
        if not ru or ru == en:
            reason = "empty or untranslated output"
            severity = "hard"
        elif len(en) >= 40 and (len(ru) / max(len(en), 1) < 0.35 or len(ru) / max(len(en), 1) > 2.8):
            reason = "suspicious length ratio"
        elif _numbers(en) != _numbers(ru):
            reason = "numbers changed or disappeared"
            severity = "hard"
        else:
            for src_term, preferred in memory.glossary.items():
                if src_term and preferred and src_term.lower() in en.lower() and preferred.lower() not in ru.lower():
                    reason = f"glossary mismatch: {src_term}"
                    break
        if reason:
            findings[s.id] = GateFinding(s.id, severity, reason)
    return list(findings.values())


@dataclass
class TranslationHarness:
    analyzer: LLMProvider
    translator: SegmentTranslator
    gate: LLMProvider
    editor: LLMProvider
    hard_editor: LLMProvider
    memory_model: LLMProvider
    use_llm_gate: bool = True

    @classmethod
    def from_env(cls) -> "TranslationHarness":
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("BOOKAI_API_KEY")
        base = os.getenv("BOOKAI_BASE_URL") or "https://openrouter.ai/api/v1"
        reasoning = os.getenv("BOOKAI_REASONING") or "none"

        def p(env_name: str, default: str, role: str) -> OpenAICompatibleProvider:
            return OpenAICompatibleProvider(key, base, os.getenv(env_name) or default, reasoning, role=role)

        translator_provider = p("BOOKAI_TRANSLATOR_MODEL", "deepseek/deepseek-v4-flash-0731", "translator")
        backend = (os.getenv("BOOKAI_TRANSLATOR_BACKEND") or "api").lower()
        translator: SegmentTranslator
        if backend == "madlad":
            translator = MadladTranslator()
        elif backend == "api":
            translator = LLMTranslator(translator_provider)
        else:
            raise ValueError("BOOKAI_TRANSLATOR_BACKEND must be 'api' or 'madlad'")

        return cls(
            analyzer=p("BOOKAI_ANALYZER_MODEL", "deepseek/deepseek-v4-flash-0731", "analyzer"),
            translator=translator,
            gate=p("BOOKAI_GATE_MODEL", "deepseek/deepseek-v4-flash-0731", "gate"),
            editor=p("BOOKAI_EDITOR_MODEL", "qwen/qwen3.8-flash", "editor"),
            hard_editor=p("BOOKAI_HARD_MODEL", "deepseek/deepseek-v4-pro-0813", "hard_editor"),
            memory_model=p("BOOKAI_MEMORY_MODEL", "deepseek/deepseek-v4-flash-0731", "memory"),
            use_llm_gate=(os.getenv("BOOKAI_LLM_GATE", "true").lower() not in {"0", "false", "no"}),
        )

    @classmethod
    def single_provider(cls, provider: LLMProvider) -> "TranslationHarness":
        return cls(provider, LLMTranslator(provider), provider, provider, provider, provider)

    @property
    def usage(self) -> dict:
        providers = {
            "analyzer": self.analyzer,
            "gate": self.gate,
            "editor": self.editor,
            "hard_editor": self.hard_editor,
            "memory": self.memory_model,
        }
        if isinstance(self.translator, LLMTranslator):
            providers["translator"] = self.translator.provider
        roles: dict[str, dict] = {}
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
        seen: set[int] = set()
        for role, provider in providers.items():
            usage = dict(getattr(provider, "usage", {}) or {})
            roles[role] = {"model": getattr(provider, "model", "unknown"), **usage}
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            for key in total:
                total[key] += int(usage.get(key) or 0)
        return {"total": total, "roles": roles, "translator": self.translator.name}

    def analyze(self, sample: str) -> BookMemory:
        return analyze_memory(self.analyzer, sample)

    def translate(self, segments, memory, *, context_before=None, context_after=None):
        return self.translator.translate(segments, memory, context_before=context_before, context_after=context_after)

    def gate_findings(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> list[GateFinding]:
        merged = {f.id: f for f in heuristic_findings(originals, draft, memory)}
        if self.use_llm_gate:
            for finding in quality_gate_batch(self.gate, originals, draft, memory):
                old = merged.get(finding.id)
                if old is None or finding.severity == "hard":
                    merged[finding.id] = finding
        return list(merged.values())

    def edit(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> dict[str, str]:
        return edit_batch(self.editor, originals, draft, memory)

    def hard_edit(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> dict[str, str]:
        return qa_batch(self.hard_editor, originals, draft, memory)

    def update_memory(self, originals: list[Segment], translated: dict[str, str], memory: BookMemory) -> BookMemory:
        return update_memory(self.memory_model, originals, translated, memory)
