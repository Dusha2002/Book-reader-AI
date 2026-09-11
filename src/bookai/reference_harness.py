from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

from .critics import literary_gate_batch
from .harness import LLMTranslator, TranslationHarness
from .llm import OpenAICompatibleProvider, extract_json
from .models import BookMemory, GateFinding, Segment
from .providers import CappedOpenAICompatibleProvider
from .quality import assert_exact_ids, batch_issues, hard_ids, is_heading
from .reference_profile import apply_reference_profile
from .resilience import resilient_findings


FLASH_MODEL = "deepseek/deepseek-v4.1-flash"
# Compatibility constant for the independent benchmark judge only.
# ReferenceTranslationHarness itself hard-forces every operational role to FLASH_MODEL.
PRO_MODEL = "deepseek/deepseek-v4-pro-0813"

_CLOSERS = "\"'”’»)]}"
_OPENERS = "\"'“‘«([{"
_COMMON_ABBREVIATIONS = (
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "no.",
)


def _memory_prompt(memory: BookMemory) -> str:
    return json.dumps(asdict(memory), ensure_ascii=False, separators=(",", ":"))


def _sentence_units(text: str) -> list[str]:
    """Conservative English sentence splitter used by the v10 micro-translation path."""
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in ".!?":
            i += 1
            continue
        j = i + 1
        while j < n and text[j] == "." and ch == ".":
            j += 1
        while j < n and text[j] in _CLOSERS:
            j += 1
        if j < n and not text[j].isspace():
            i += 1
            continue
        prefix = text[start:j].rstrip()
        lower = prefix.casefold()
        if ch == "." and any(lower.endswith(abbr) for abbr in _COMMON_ABBREVIATIONS):
            i += 1
            continue
        if ch == "." and re.search(r"(?:^|\s)[A-Z]\.$", prefix):
            i += 1
            continue
        k = j
        while k < n and text[k].isspace():
            k += 1
        if k < n:
            probe = k
            while probe < n and text[probe] in _OPENERS:
                probe += 1
            if probe < n and not (text[probe].isupper() or text[probe].isdigit()):
                i += 1
                continue
        unit = text[start:j].strip()
        if unit:
            out.append(unit)
        start = k
        i = k
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


class ReferenceTranslationHarness(TranslationHarness):
    """Flash-only v10 harness calibrated against the user literary reference."""

    _ADVISORY_LITERARY_PREFIXES = (
        "calque:", "rhythm:", "voice:", "irony:", "idiom:", "dialogue:",
    )

    def analyze(self, sample: str):
        return apply_reference_profile(super().analyze(sample))

    def update_memory(self, originals, translated, memory):
        return apply_reference_profile(super().update_memory(originals, translated, memory))

    @staticmethod
    def _needs_sentence_decomposition(segment: Segment) -> bool:
        if is_heading(segment):
            return False
        units = _sentence_units(segment.text)
        if not units:
            return False
        char_threshold = max(420, int(os.getenv("BOOKAI_V10_DECOMPOSE_CHARS") or "650"))
        sentence_threshold = max(3, int(os.getenv("BOOKAI_V10_DECOMPOSE_SENTENCES") or "4"))
        isolate_sentence_chars = max(280, int(os.getenv("BOOKAI_V10_ISOLATE_SENTENCE_CHARS") or "420"))
        max_sentence_chars = max(len(unit) for unit in units)
        clause_pressure = segment.text.count(",") + 2 * segment.text.count(";") + segment.text.count(":") + segment.text.count("—") + segment.text.count(" - ")
        if len(units) == 1:
            return max_sentence_chars >= isolate_sentence_chars or clause_pressure >= 10
        return len(segment.text) >= char_threshold or len(units) >= sentence_threshold or max_sentence_chars >= isolate_sentence_chars or clause_pressure >= 10

    def _translate_decomposed(self, segment: Segment, memory: BookMemory, *, context_before: list[Segment] | None = None, context_after: list[Segment] | None = None) -> dict[str, str]:
        if not isinstance(self.translator, LLMTranslator):
            return super().translate([segment], memory, context_before=context_before, context_after=context_after)
        units = _sentence_units(segment.text)
        if len(units) < 2:
            return super().translate([segment], memory, context_before=context_before, context_after=context_after)
        sentence_segments = [Segment(id=f"{segment.id}__v10_{index:02d}", text=text, locator=segment.locator, chapter=segment.chapter) for index, text in enumerate(units, 1)]
        before = [{"id": s.id, "text": s.text} for s in (context_before or [])]
        after = [{"id": s.id, "text": s.text} for s in (context_after or [])]
        provider = self.translator.provider
        batch_size = max(1, min(4, int(os.getenv("BOOKAI_V10_SENTENCE_BATCH") or "1")))
        sentence_system = """You are the fidelity stage of a Flash-only EN→RU literary translation harness.
The hard paragraph has been decomposed into sentence obligations to keep each generation task small.
Translate ONLY the TARGET_SENTENCES supplied in this call. Use the COMPLETE SOURCE PARAGRAPH,
the full sentence map, neighboring source paragraphs, previously accepted Russian sentence translations
and the translation bible to resolve pronouns, ellipsis, irony, terminology, causality and subtext.
Each target sentence must be complete: preserve every proposition, relation, number, negation,
qualification, image, joke premise and technical detail. Natural Russian wording is required; literal
English syntax is not. Do not merge ids, omit clauses, add explanations, or translate context passages.
Return ONLY one JSON object mapping EXACTLY every target sentence id to its Russian translation."""
        sentence_ru: dict[str, str] = {}
        source_map = {s.id: s.text for s in sentence_segments}
        for start in range(0, len(sentence_segments), batch_size):
            batch = sentence_segments[start:start + batch_size]
            previous = [{"id": prior.id, "source": prior.text, "translation": sentence_ru[prior.id]} for prior in sentence_segments[:start] if prior.id in sentence_ru][-3:]
            upcoming = [{"id": item.id, "text": item.text} for item in sentence_segments[start + len(batch):start + len(batch) + 3]]
            sentence_user = (
                f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\n"
                f"CONTEXT_BEFORE:{json.dumps(before, ensure_ascii=False)}\n"
                f"FULL_SOURCE_PARAGRAPH:{json.dumps(segment.text, ensure_ascii=False)}\n"
                f"PARAGRAPH_SENTENCE_MAP:{json.dumps(source_map, ensure_ascii=False)}\n"
                f"PREVIOUS_ACCEPTED_RUSSIAN:{json.dumps(previous, ensure_ascii=False)}\n"
                f"TARGET_SENTENCES:{json.dumps({s.id: s.text for s in batch}, ensure_ascii=False)}\n"
                f"UPCOMING_SOURCE_SENTENCES:{json.dumps(upcoming, ensure_ascii=False)}\n"
                f"CONTEXT_AFTER:{json.dumps(after, ensure_ascii=False)}"
            )
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    sentence_obj = extract_json(provider.complete(sentence_system, sentence_user, temperature=0.1 if attempt == 0 else 0.0))
                    if not isinstance(sentence_obj, dict):
                        raise ValueError("v10_sentence_translator returned non-object JSON")
                    expected_ids = {s.id for s in batch}
                    extra_ids = sorted({str(k) for k in sentence_obj} - expected_ids)
                    if extra_ids:
                        print(f"[bookai-v10] sentence_context_spillover id={segment.id} start={start} ignored={extra_ids[:6]}", flush=True)
                    # For a single target, keep the original object so assert_exact_ids
                    # can safely recover a sole mislabeled value. For multi-target
                    # calls, retain strict exact-id filtering to prevent cross-wiring.
                    target_obj = sentence_obj if len(batch) == 1 else {sid: sentence_obj[sid] for sid in expected_ids if sid in sentence_obj}
                    translated_part = assert_exact_ids(batch, target_obj, "v10_sentence_translator")
                    sentence_ru.update(translated_part)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[bookai-v10] sentence_batch_retry id={segment.id} start={start} attempt={attempt + 1} error={type(exc).__name__}", flush=True)
            if last_error is not None:
                raise last_error
        reassembly_system = """You are the literary reassembly stage of a Flash-only EN→RU translation harness.
Rebuild ONE finished Russian literary paragraph from the already faithful sentence translations.
The sentence translations are semantic anchors, not a demand to preserve their exact syntax.
Keep every source fact, relation, number, negation, qualification, joke premise and image, while restoring
the author's paragraph-level cadence: restrained, precise, dryly ironic, natural Russian prose.
Preserve accumulation and punch-line placement when functional. You may change Russian sentence boundaries,
connectives and word order only to improve idiomatic flow and voice. Never summarize, omit, add, explain,
intensify emotion, or flatten technical/physical meaning. Return ONLY one JSON object mapping the exact
paragraph id to the complete Russian paragraph."""
        sentence_pairs = [{"id": s.id, "source": s.text, "translation": sentence_ru[s.id]} for s in sentence_segments]
        reassembly_user = (
            f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\n"
            f"CONTEXT_BEFORE:{json.dumps(before, ensure_ascii=False)}\n"
            f"FULL_SOURCE_PARAGRAPH:{json.dumps(segment.text, ensure_ascii=False)}\n"
            f"SENTENCE_TRANSLATIONS:{json.dumps(sentence_pairs, ensure_ascii=False)}\n"
            f"CONTEXT_AFTER:{json.dumps(after, ensure_ascii=False)}\n"
            f"PARAGRAPH_ID:{segment.id}"
        )
        paragraph_obj = extract_json(self.editor.complete(reassembly_system, reassembly_user, temperature=0.1))
        result = assert_exact_ids([segment], paragraph_obj, "v10_paragraph_reassembly")
        hard = hard_ids(batch_issues([segment], result, memory))
        if not hard:
            return result
        joined = " ".join(sentence_ru[s.id] for s in sentence_segments).strip()
        fallback = {segment.id: joined}
        fallback_hard = hard_ids(batch_issues([segment], fallback, memory))
        if fallback_hard:
            raise ValueError("v10 decomposed paragraph failed deterministic QA for: " + ", ".join(sorted(fallback_hard)))
        print(f"[bookai-v10] reassembly_fallback id={segment.id}", flush=True)
        return fallback

    def translate(self, segments, memory, *, context_before=None, context_after=None):
        chunk = list(segments)
        if not chunk:
            return {}
        complex_segments = [s for s in chunk if self._needs_sentence_decomposition(s)]
        if not complex_segments:
            return super().translate(chunk, memory, context_before=context_before, context_after=context_after)
        complex_ids = {s.id for s in complex_segments}
        normal = [s for s in chunk if s.id not in complex_ids]
        out: dict[str, str] = {}
        if normal:
            out.update(super().translate(normal, memory, context_before=context_before, context_after=context_after))
        for segment in complex_segments:
            units = _sentence_units(segment.text)
            mode = "sentence_micro_batches" if len(units) >= 2 else "isolated_long_sentence"
            print(f"[bookai-v10] micro_translate id={segment.id} mode={mode} chars={len(segment.text)} sentences={len(units)}", flush=True)
            out.update(self._translate_decomposed(segment, memory, context_before=context_before, context_after=context_after))
        return out

    def _semantic_confirmation_batch(self, originals: list[Segment], draft: dict[str, str], memory: BookMemory) -> list[GateFinding]:
        if not originals:
            return []
        pairs = {s.id: {"en": s.text, "ru": draft.get(s.id, "")} for s in originals}
        system = """You are the semantic confirmation stage of a Flash-only EN→RU literary translation harness.
For EVERY supplied paragraph id, compare English and Russian proposition by proposition.
You MUST return one explicit check row for EVERY id, including paragraphs with no defect.
Judge only semantic coverage/fidelity: omissions/additions; actor-action-object relations; pronoun referents;
enumeration members; numbers/quantities; chronology/cause/effect; negation/modality/uncertainty; concrete images;
technical referents; joke premises, contrasts and qualifications. Harmless nonliteral Russian wording is fine.
Return ONLY JSON exactly in this shape:
{"checks":{"id":{"ok":true,"reason":""}}}
The checks object must contain EXACTLY all supplied ids and no others.
Use ok=false for any genuine semantic corruption or omission and give a concise specific reason."""
        user = f"TRANSLATION_BIBLE:{_memory_prompt(memory)}\nPAIRS:{json.dumps(pairs, ensure_ascii=False)}"
        obj = extract_json(self.gate.complete(system, user, temperature=0.0))
        if not isinstance(obj, dict) or not isinstance(obj.get("checks"), dict):
            raise ValueError("semantic confirmation returned invalid checks object")
        checks = obj["checks"]
        expected = [s.id for s in originals]
        expected_set = set(expected)
        actual_set = {str(k) for k in checks}
        if actual_set != expected_set:
            raise ValueError("semantic confirmation id contract violation: " f"missing={sorted(expected_set - actual_set)[:12]} " f"extra={sorted(actual_set - expected_set)[:12]}")
        findings: list[GateFinding] = []
        for sid in expected:
            row = checks[sid]
            if not isinstance(row, dict) or type(row.get("ok")) is not bool:
                raise ValueError(f"semantic confirmation invalid row for {sid}")
            reason = str(row.get("reason") or "").strip()
            if row["ok"]:
                continue
            if not reason:
                raise ValueError(f"semantic confirmation missing reason for {sid}")
            findings.append(GateFinding(sid, "hard", f"semantic_confirmation: {reason}"))
        return findings

    def gate_findings(self, originals, draft, memory):
        merged: dict[str, GateFinding] = {}
        for issue in batch_issues(originals, draft, memory):
            self._merge_finding(merged, GateFinding(issue.id, issue.severity, f"{issue.code}: {issue.reason}"))
        if self.use_llm_gate:
            semantic = resilient_findings(list(originals), lambda part: self._semantic_confirmation_batch(part, draft, memory), label="semantic_confirmation", attempts=2)
            literary = resilient_findings(list(originals), lambda part: literary_gate_batch(self.gate, part, draft, memory), label="literary_gate", attempts=2)
            for finding in semantic + literary:
                self._merge_finding(merged, finding)
        out: list[GateFinding] = []
        for finding in merged.values():
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
    def forced_model(env_name: str, role: str) -> str:
        requested = (os.getenv(env_name) or FLASH_MODEL).strip()
        if requested != FLASH_MODEL:
            print(f"[bookai-v10-model-ceiling] role={role} requested={requested} forced={FLASH_MODEL}", flush=True)
        return FLASH_MODEL
    translator_model = forced_model("BOOKAI_TRANSLATOR_MODEL", "translator")
    analyzer_model = forced_model("BOOKAI_ANALYZER_MODEL", "analyzer")
    gate_model = forced_model("BOOKAI_GATE_MODEL", "gate")
    memory_model = forced_model("BOOKAI_MEMORY_MODEL", "memory")
    editor_model = forced_model("BOOKAI_EDITOR_MODEL", "editor")
    hard_model = forced_model("BOOKAI_HARD_MODEL", "hard_editor")
    def normal(model: str, role: str) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(key, base, model, reasoning, role=role)
    def capped(model: str, role: str, default_cap: int) -> CappedOpenAICompatibleProvider:
        cap = int(os.getenv(f"BOOKAI_{role.upper()}_MAX_TOKENS") or default_cap)
        return CappedOpenAICompatibleProvider(key, base, model, reasoning, role=role, max_tokens=cap)
    print("[bookai-v10] strategy=v4.1-flash-only+contextual-sentence-micro-batches+isolated-long-sentences+explicit-semantic-confirmation external_reference_judge=separate", flush=True)
    translator_provider = normal(translator_model, "translator")
    return ReferenceTranslationHarness(
        analyzer=capped(analyzer_model, "analyzer", 4096),
        translator=LLMTranslator(translator_provider),
        gate=capped(gate_model, "gate", 8192),
        editor=normal(editor_model, "editor"),
        hard_editor=normal(hard_model, "hard_editor"),
        memory_model=capped(memory_model, "memory", 4096),
        use_llm_gate=(os.getenv("BOOKAI_LLM_GATE", "true").lower() not in {"0", "false", "no"}),
    )