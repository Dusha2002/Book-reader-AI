from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from .llm import extract_json
from .models import BookMemory, Segment
from .reference_profile import REFERENCE_GLOSSARY_SEED

STYLE_DIMENSIONS = {
    "register",
    "narrative_voice",
    "sentence_rhythm",
    "lexicon",
    "imagery",
    "dialogue",
    "punctuation",
    "formatting",
    "other",
}

_STOPWORDS = {
    "about", "after", "again", "against", "almost", "also", "among", "another", "around",
    "because", "before", "being", "between", "could", "didn", "does", "doing", "down",
    "during", "each", "even", "every", "first", "from", "hadn", "hasn", "have", "having",
    "into", "itself", "just", "might", "more", "most", "much", "must", "never", "other",
    "over", "same", "should", "since", "some", "still", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "under", "very",
    "wasn", "were", "what", "when", "where", "which", "while", "with", "would", "your",
}


def lint_style_rule(instruction: str) -> list[str]:
    """Reject rules that would turn a style card into copied vocabulary or plot lore."""
    text = " ".join(str(instruction or "").split())
    flags: list[str] = []
    if not text:
        return ["empty"]
    if len(text) < 25:
        flags.append("too_specific")
    if re.search(r'["“”«»][^"“”«»]*[A-Za-zА-Яа-яЁё][^"“”«»]*["“”«»]', text):
        flags.append("quoted_example")
    if re.search(r"\b(?:e\.g\.|i\.e\.|for example|for instance|such as|words? like|phrases? like)\b", text, re.I):
        flags.append("example_marker")
    if re.search(r"(?:\b[\w'-]{2,}\b\s*[,/]\s*){2,}\b[\w'-]{2,}\b", text):
        flags.append("word_list")
    body = re.sub(r"^[A-ZА-ЯЁ][a-zа-яё]+\b", "", text)
    if re.search(r"\b(?:[A-ZА-ЯЁ][a-zа-яё]+\s+){1,}[A-ZА-ЯЁ][a-zа-яё]+\b|\b[A-ZА-ЯЁ]{3,}\b|[a-zа-яё][A-ZА-ЯЁ][a-zа-яё]", body):
        flags.append("proper_noun")
    return flags


def _fallback_style_rules(memory: BookMemory) -> list[dict[str, str]]:
    candidates = [
        ("narrative_voice", memory.style.narrative_voice),
        ("sentence_rhythm", memory.style.rhythm),
        ("dialogue", memory.style.dialogue),
        ("register", memory.style.humor),
    ]
    rules: list[dict[str, str]] = []
    for dimension, value in candidates:
        clean = " ".join(str(value or "").split())
        if clean:
            rules.append({"dimension": dimension, "instruction": clean[:900]})
    return rules[:8]


def extract_style_card(provider, sample: str, memory: BookMemory) -> dict:
    """One-call abstract author-style extraction inspired by TBL's style cards."""
    max_chars = max(6000, min(16000, int(os.getenv("BOOKAI_STYLE_SAMPLE_CHARS") or "12000")))
    text = sample[:max_chars]
    system = """Extract a compact SOURCE-AUTHOR STYLE CARD for EN→RU literary translation.
Describe reusable tendencies, never plot facts and never characteristic words to copy. The translation
must preserve meaning exactly; style rules only shape register, cadence and expression.

Return ONLY JSON:
{"context":"1-3 short sentences about era/technology/social frame, no plot summary",
 "rules":[{"dimension":"register|narrative_voice|sentence_rhythm|lexicon|imagery|dialogue|punctuation|formatting|other",
           "instruction":"abstract reusable instruction"}]}

Rules:
- 6-12 rules maximum; each must be independently useful.
- No quotations from the book, examples, word lists, character/place names, author names or plot summary.
- Capture dry/warm/ironic distance, sentence architecture, dialogue behavior, imagery density,
  lexical age/register and punctuation only when supported by the sample.
- Treat tendencies as tendencies: do not force repeated tics or fixed vocabulary."""
    user = (
        f"KNOWN_TRANSLATION_MEMORY:{json.dumps(asdict(memory), ensure_ascii=False)}\n"
        f"SOURCE_SAMPLE:{text}"
    )
    try:
        obj = extract_json(provider.complete(system, user, temperature=0.0))
    except Exception as exc:
        return {
            "context": "",
            "rules": _fallback_style_rules(memory),
            "rejected": [],
            "fallback": type(exc).__name__,
        }
    if not isinstance(obj, dict):
        return {"context": "", "rules": _fallback_style_rules(memory), "rejected": [], "fallback": "non_object"}

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, object]] = []
    for raw in obj.get("rules") or []:
        if not isinstance(raw, dict):
            continue
        dimension = str(raw.get("dimension") or "other").strip().lower()
        if dimension not in STYLE_DIMENSIONS:
            dimension = "other"
        instruction = " ".join(str(raw.get("instruction") or "").split())
        flags = lint_style_rule(instruction)
        row = {"dimension": dimension, "instruction": instruction}
        if flags:
            rejected.append({**row, "flags": flags})
            continue
        accepted.append(row)
        if len(accepted) >= 12:
            break
    if len(accepted) < 4:
        accepted = _fallback_style_rules(memory)
    context = " ".join(str(obj.get("context") or "").split())[:500]
    return {"context": context, "rules": accepted, "rejected": rejected}


def assemble_style_card(card: dict) -> str:
    rules = [
        f"- {str(row.get('instruction') or '').strip()}"
        for row in card.get("rules") or []
        if isinstance(row, dict) and str(row.get("instruction") or "").strip()
    ]
    context = str(card.get("context") or "").strip()
    parts = [
        "SOURCE AUTHOR STYLE CARD — tendencies only; never copy fixed wording:",
        *rules,
        "Anti-tic guard: vary wording naturally; style rules never authorize adding, omitting, or changing source facts.",
    ]
    if context:
        parts.extend([
            "SETTING FRAME:",
            context,
            "Use period-appropriate vocabulary; do not import a later technological or social register.",
        ])
    return "\n".join(parts)


def apply_style_card(memory: BookMemory, card: dict) -> BookMemory:
    block = assemble_style_card(card)
    if not block.strip():
        return memory
    style = replace(
        memory.style,
        narrative_voice=(memory.style.narrative_voice + "\n" + block)[-12000:],
    )
    return replace(memory, style=style)


def ensure_chapter_digests(harness, chapters, memory: BookMemory, existing: dict | None = None) -> dict[str, str]:
    """Parallel, cached chapter prescan as used by long-form translation agents."""
    digests = {str(k): str(v) for k, v in dict(existing or {}).items() if str(v).strip()}
    missing = [(name, chapter) for name, chapter in chapters if not digests.get(name)]
    if not missing:
        return digests
    workers = max(2, min(8, int(os.getenv("BOOKAI_CONTEXT_WORKERS") or "6")))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bookai-prescan") as pool:
        futures = {pool.submit(harness.chapter_brief, chapter, memory): name for name, chapter in missing}
        for future in as_completed(futures):
            name = futures[future]
            try:
                value = " ".join(str(future.result() or "").split())[:5000]
            except Exception as exc:
                print(f"[bookai-context] chapter_digest_skip chapter={name!r} error={type(exc).__name__}", flush=True)
                value = ""
            if value:
                digests[name] = value
    return digests


def build_book_synopsis(provider, chapter_digests: dict[str, str], existing: str = "") -> str:
    if str(existing or "").strip():
        return str(existing).strip()
    payload = {name: digest for name, digest in chapter_digests.items() if digest}
    if not payload:
        return ""
    system = """Build a compact continuity synopsis for an EN→RU literary translator from chapter digests.
Keep only facts that help later translation: identities/roles, relationships, chronology, recurring objects,
technical/world rules and unresolved referents. Do not prescribe prose style. Return ONLY {"synopsis":"..."}.
Maximum 900 Russian words."""
    try:
        obj = extract_json(provider.complete(system, json.dumps(payload, ensure_ascii=False), temperature=0.0))
        if isinstance(obj, dict):
            return " ".join(str(obj.get("synopsis") or "").split())[:9000]
    except Exception as exc:
        print(f"[bookai-context] synopsis_skip error={type(exc).__name__}", flush=True)
    return ""


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text.casefold())
        if token not in _STOPWORDS
    }


class SourceContextIndex:
    """Lightweight IDF retrieval for sparse distant context; no embedding dependency."""

    def __init__(self, segments: list[Segment]):
        self.segments = list(segments)
        self.positions = {segment.id: index for index, segment in enumerate(self.segments)}
        self.term_sets = [_terms(segment.text) for segment in self.segments]
        df = Counter(term for terms in self.term_sets for term in terms)
        n = max(1, len(self.segments))
        self.idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    def retrieve(self, batch: list[Segment], k: int | None = None) -> list[Segment]:
        if not batch:
            return []
        k = max(0, min(4, int(k if k is not None else os.getenv("BOOKAI_RETRIEVAL_K") or "2")))
        if k == 0:
            return []
        batch_ids = {segment.id for segment in batch}
        query = set().union(*(_terms(segment.text) for segment in batch))
        if not query:
            return []
        batch_positions = [self.positions.get(segment.id, 0) for segment in batch]
        lo, hi = min(batch_positions), max(batch_positions)
        scored: list[tuple[float, int]] = []
        for index, (segment, terms) in enumerate(zip(self.segments, self.term_sets)):
            if segment.id in batch_ids or lo - 5 <= index <= hi + 5:
                continue
            overlap = query & terms
            if len(overlap) < 2:
                continue
            score = sum(self.idf.get(term, 1.0) for term in overlap) / math.sqrt(max(1, len(terms)))
            if segment.chapter == batch[0].chapter:
                score *= 1.12
            scored.append((score, index))
        scored.sort(reverse=True)
        return [self.segments[index] for _, index in scored[:k]]


def memory_with_context(
    memory: BookMemory,
    *,
    book_synopsis: str = "",
    chapter_digest: str = "",
    distant: list[Segment] | None = None,
) -> BookMemory:
    parts: list[str] = []
    if book_synopsis:
        parts.append("BOOK CONTINUITY SYNOPSIS:\n" + book_synopsis[:6000])
    if chapter_digest:
        parts.append("CURRENT CHAPTER DIGEST:\n" + chapter_digest[:4500])
    if distant:
        snippets = [
            {"id": segment.id, "chapter": segment.chapter, "source": segment.text[:1200]}
            for segment in distant
        ]
        parts.append(
            "RETRIEVED DISTANT SOURCE CONTEXT — read only; translate only target ids:\n"
            + json.dumps(snippets, ensure_ascii=False)
        )
    if not parts:
        return memory
    rolling = (memory.rolling_summary + "\n" + "\n".join(parts))[-14000:]
    return replace(memory, rolling_summary=rolling)


def _ru_stem(word: str) -> str:
    token = word.casefold()
    for suffix in ("иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "ей", "ой", "ая", "яя", "ий", "ый", "ое", "ее", "ов", "ев", "ам", "ям", "ах", "ях", "ы", "и", "а", "я", "у", "ю", "е", "ом", "ем"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _target_present(target: str, candidate: str) -> bool:
    candidate_words = [_ru_stem(x) for x in re.findall(r"[А-Яа-яЁё-]+", candidate)]
    target_words = [_ru_stem(x) for x in re.findall(r"[А-Яа-яЁё-]+", target)]
    if not target_words:
        return target.casefold() in candidate.casefold()
    return all(any(c.startswith(t) or t.startswith(c) for c in candidate_words) for t in target_words)


def locked_glossary_violations(
    originals: list[Segment],
    translations: dict[str, str],
    locked: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Mechanical validator for high-confidence locked terms, inspired by epublate."""
    locked = dict(locked or REFERENCE_GLOSSARY_SEED)
    out: dict[str, list[str]] = {}
    for segment in originals:
        source = segment.text.casefold()
        candidate = str(translations.get(segment.id) or "")
        missing: list[str] = []
        for source_term, target_term in locked.items():
            if source_term.casefold() in source and not _target_present(target_term, candidate):
                missing.append(source_term)
        if missing:
            out[segment.id] = missing
    return out


def risk_score(segment: Segment, translation: str, locked_violation: bool = False) -> float:
    """Cheap uncertainty proxy used to select a bounded segment-level refinement set."""
    text = segment.text
    score = 0.0
    score += min(4.0, len(text) / 700.0)
    score += min(2.0, text.count(";") * 0.45 + text.count(":") * 0.25)
    score += min(2.0, (text.count("—") + text.count(" - ")) * 0.35)
    score += 1.0 if ('"' in text or "“" in text or "‘" in text) else 0.0
    score += 1.2 if len(re.findall(r"\b(?:but|although|however|unless|except|rather|while)\b", text, re.I)) >= 2 else 0.0
    score += 4.0 if locked_violation else 0.0
    if translation:
        src_len = max(1, len(text))
        ratio = len(translation) / src_len
        if ratio < 0.45 or ratio > 1.65:
            score += 2.0
    return score


def select_refinement_targets(
    targets: list[Segment],
    translations: dict[str, str],
    *,
    limit: int | None = None,
    locked_violations: dict[str, list[str]] | None = None,
) -> list[Segment]:
    limit = max(0, int(limit if limit is not None else os.getenv("BOOKAI_RAPID_REFINE_MAX") or "220"))
    if limit == 0:
        return []
    locked_ids = set((locked_violations or {}).keys())
    ranked = sorted(
        (
            (risk_score(segment, str(translations.get(segment.id) or ""), segment.id in locked_ids), index, segment)
            for index, segment in enumerate(targets)
            if segment.id in translations
        ),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    return [segment for score, _, segment in ranked if score >= 2.5][:limit]


def atomic_persist(path: Path, state: dict, translations: dict[str, str], memory: BookMemory) -> None:
    """Crash-safe checkpoint: write+fsync a sibling temp file, then atomic replace."""
    state["pipeline_version"] = state.get("pipeline_version") or "literary-harness-v2.1"
    state["translations"] = translations
    state["memory"] = asdict(memory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
