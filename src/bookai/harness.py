from __future__ import annotations

import os
from dataclasses import dataclass

from .audit import safe_analyze_memory, safe_chapter_brief, safe_update_memory, semantic_gate_batch
from .llm import (
    OpenAICompatibleProvider,
    alternative_batch,
    choose_candidate_batch,
    edit_batch,
    qa_batch,
    quality_gate_batch,
    translate_batch,
)
from .models import BookMemory, GateFinding, LLMProvider, Segment, SegmentTranslator
from .polish import literary_polish_batch
from .providers import CappedOpenAICompatibleProvider
from .quality import batch_issues


FLASH_MODEL = "deepseek/deepseek-v4-flash-0731"


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
    """Optional local MADLAD-400 draft translator. Flash still performs polish/QA."""

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

        # This is a HARD ceiling, not a configurable default. The project may use
        # a weaker/local draft backend, but no API role may exceed V4 Flash.
        requested_ceiling = (os.getenv("BOOKAI_MAX_MODEL") or FLASH_MODEL).strip()
        if requested_ceiling != FLASH_MODEL:
            print(
                f"[bookai-model-ceiling] requested_ceiling={requested_ceiling} forced={FLASH_MODEL}",
                flush=True,
            )
        ceiling = FLASH_MODEL

        def p(env_name: str, role: str) -> OpenAICompatibleProvider:
            requested = (os.getenv(env_name) or FLASH_MODEL).strip()
            if requested != ceiling:
                print(
                    f"[bookai-model-ceiling] role={role} requested={requested} forced={ceiling}",
                    flush=True,
                )
            compact_caps = {"analyzer": 4096, "gate": 8192, "memory": 4096}
            if role in compact_caps:
                cap = int(os.getenv(f"BOOKAI_{role.upper()}_MAX_TOKENS") or compact_caps[role])
                return CappedOpenAICompatibleProvider(
                    key,
                    base,
                    ceiling,
                    reasoning,
                    role=role,
                    max_tokens=cap,
                )
            return OpenAICompatibleProvider(key, base, ceiling, reasoning, role=role)

        translator_provider = p("BOOKAI_TRANSLATOR_MODEL", "translator")
        backend = (os.getenv("BOOKAI_TRANSLATOR_BACKEND") or "api").lower()
        translator: SegmentTranslator
        if backend == "madlad":
            translator = MadladTranslator()
        elif backend == "api":
            translator = LLMTranslator(translator_provider)
        else:
            raise ValueError("BOOKAI_TRANSLATOR_BACKEND must be 'api' or 'madlad'")

        return cls(
            analyzer=p("BOOKAI_ANALYZER_MODEL", "analyzer"),
            translator=translator,
            gate=p("BOOKAI_GATE_MODEL", "gate"),
            editor=p("BOOKAI_EDITOR_MODEL", "editor"),
            hard_editor=p("BOOKAI_HARD_MODEL", "hard_editor"),
            memory_model=p("BOOKAI_MEMORY_MODEL", "memory"),
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
        total: dict[str, int | float] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "cost": 0.0,
        }
        seen: set[int] = set()
        for role, provider in providers.items():
            usage = dict(getattr(provider, "usage", {}) or {})
            roles[role] = {"model": getattr(provider, "model", "unknown"), **usage}
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "requests"):
                total[key] = int(total[key]) + int(usage.get(key) or 0)
            total["cost"] = float(total["cost"]) + float(usage.get("cost") or 0.0)
        return {"total": total, "roles": roles, "translator": self.translator.name}

    def analyze(self, sample: str) -> BookMemory:
        return safe_analyze_memory(self.analyzer, sample)

    def chapter_brief(self, originals: list[Segment], memory: BookMemory) -> str:
        return safe_chapter_brief(self.analyzer, originals, memory)

    def translate(self, segments, memory, *, context_before=None, context_after=None):
        return self.translator.translate(segments, memory, context_before=context_before, context_after=context_after)

    def polish(
        self,
        originals: list[Segment],
        draft: dict[str, str],
        memory: BookMemory,
        *,
        context: list[dict] | None = None,
    ) -> dict[str, str]:
        return literary_polish_batch(self.editor, originals, draft, memory, context=context)

    @staticmethod
    def _merge_finding(merged: dict[str, GateFinding], finding: GateFinding) -> None:
        old = merged.get(finding.id)
        if old is None:
            merged[finding.id] = finding
            return
        if finding.severity == "hard" and old.severity != "hard":
            old.severity = "hard"
        if finding.reason and finding.reason not in old.reason:
            old.reason = old.reason + "; " + finding.reason

    def gate_findings(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> list[GateFinding]:
        merged: dict[str, GateFinding] = {}
        for issue in batch_issues(originals, draft, memory):
            self._merge_finding(
                merged,
                GateFinding(issue.id, issue.severity, f"{issue.code}: {issue.reason}"),
            )
        if self.use_llm_gate:
            # Two independent cheap-model critics instead of one overloaded judge:
            # first semantic completeness, then literary Russian/voice.
            for finding in semantic_gate_batch(self.gate, originals, draft, memory):
                self._merge_finding(merged, finding)
            for finding in quality_gate_batch(self.gate, originals, draft, memory):
                self._merge_finding(merged, finding)
        return list(merged.values())

    def edit(
        self,
        originals: list[Segment],
        draft: dict[str, str],
        memory: BookMemory,
        *,
        reasons: dict[str, str] | None = None,
        context: list[dict] | None = None,
    ) -> dict[str, str]:
        return edit_batch(self.editor, originals, draft, memory, reasons=reasons, context=context)

    def alternative(self, originals: list[Segment], memory: BookMemory) -> dict[str, str]:
        # Alternative generation is an optional quality booster, not a single
        # point of failure. Invalid JSON/empty ids must fall through to the
        # targeted semantic repair pass that follows.
        last_error: BaseException | None = None
        for attempt in range(2):
            try:
                return alternative_batch(self.editor, originals, memory)
            except BaseException as exc:
                last_error = exc
                print(
                    f"[bookai-alternative-retry] attempt={attempt + 1}/2 error={type(exc).__name__}",
                    flush=True,
                )
        print(
            f"[bookai-alternative-skip] error={type(last_error).__name__ if last_error else 'unknown'}",
            flush=True,
        )
        # English source is deliberately returned as a poisoned sentinel. The
        # deterministic QA immediately rejects it, so it can never replace the
        # valid current translation; targeted repair still runs afterwards.
        return {s.id: s.text for s in originals}

    def choose(self, originals: list[Segment], first: dict[str, str], second: dict[str, str], memory: BookMemory) -> dict[str, str]:
        try:
            return choose_candidate_batch(self.gate, originals, first, second, memory)
        except BaseException as exc:
            # A/B selection is also optional. Preserve the known-good candidate
            # and let the explicit defect-driven editor repair it next.
            print(f"[bookai-judge-skip] error={type(exc).__name__}", flush=True)
            return dict(first)

    def hard_edit(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> dict[str, str]:
        # Compatibility surface: this is still forced to Flash under the ceiling.
        return qa_batch(self.hard_editor, originals, draft, memory)

    def update_memory(self, originals: list[Segment], translated: dict[str, str], memory: BookMemory) -> BookMemory:
        return safe_update_memory(self.memory_model, originals, translated, memory)
