from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gigachat_v3 import GigaChatLightningV3Backend
from .llm import extract_json
from .models import BookMemory, Segment
from .numeric_fidelity import compare_numeric_fidelity
from .reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile
from .semantic_fidelity import (
    compare_material_fidelity,
    compare_order_fidelity,
    compare_question_fidelity,
    compare_short_omission_fidelity,
)


# v10 deliberately does not import any chapter_reference_translation_v9* module.
# It is a clean orchestration layer over stable parser/model/provider primitives:
# whole-book source intelligence -> Giga primary -> deterministic QA -> Giga spans
# -> one DeepSeek semantic batch -> deterministic publication gate.


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _usage(response) -> dict[str, int]:
    usage_obj = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }


def _json_from_text(text: str) -> dict[str, Any]:
    obj = extract_json(str(text or ""))
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def _giga_json(backend: GigaChatLightningV3Backend, system: str, payload: Any, *, max_tokens: int = 5000) -> dict[str, Any]:
    client = backend._ensure_client()
    request = {
        "model": backend.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": max(1200, min(7000, int(max_tokens))),
    }
    last: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat(request)
            backend.usage.add(_usage(response), calls=1)
            return _json_from_text(str(response.choices[0].message.content or ""))
        except Exception as exc:
            last = exc
            if attempt == 0 and any(token in f"{type(exc).__name__}: {exc}".casefold() for token in ("429", "rate", "timeout", "json")):
                time.sleep(1.2)
                continue
            raise
    raise last or RuntimeError("GigaChat JSON call failed")


_COMMON_CAPITALIZED = {
    "The", "A", "An", "And", "But", "Or", "If", "When", "While", "Then", "There", "This", "That",
    "He", "She", "It", "They", "We", "I", "You", "His", "Her", "Their", "Chapter", "One", "Two",
    "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
    "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
}
_COMMON_TERMS = {
    "because", "before", "after", "although", "through", "without", "between", "another", "something",
    "anything", "nothing", "everything", "himself", "herself", "themselves", "however", "therefore",
    "perhaps", "already", "against", "thought", "looked", "little", "rather", "should", "would", "could",
    "people", "really", "enough", "always", "almost", "around", "having", "being", "where", "which",
}


class BookBibleBuilder:
    """Build compact source-only intelligence from candidates sampled across the whole book.

    Unlike v9ab's contiguous 32k sample, candidate extraction scans every source segment.
    Only compact candidate+context records are sent to GigaChat, so the one-time book
    analysis stays cheap enough to cache and reuse for every chapter.
    """

    def __init__(self, backend: GigaChatLightningV3Backend, cache_path: Path):
        self.backend = backend
        self.cache_path = Path(cache_path)
        self.stats: dict[str, Any] = {"cache_hit": False, "analysis_calls": 0}

    @staticmethod
    def _candidate_records(segments: list[Segment]) -> list[dict[str, Any]]:
        proper_counts: Counter[str] = Counter()
        term_counts: Counter[str] = Counter()
        contexts: dict[str, list[dict[str, str]]] = {}
        proper_re = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){0,2}\b")
        word_re = re.compile(r"\b[a-z][a-z'-]{5,24}\b")
        technical_phrase_re = re.compile(
            r"\b(?:[a-z][a-z-]+\s+){0,2}(?:armou?r|cuisses?|gorget|brigandine|saw|file|plate|lock|locks|gear|gears|"
            r"blade|tool|tools|machine|screw|spring|steel|iron|brass|bronze|tinplate|stock|treadle|scorpion)\b",
            re.I,
        )

        def remember(key: str, segment: Segment) -> None:
            rows = contexts.setdefault(key, [])
            if len(rows) < 2:
                rows.append({"chapter": str(segment.chapter or ""), "text": _norm(segment.text)[:430]})

        for segment in segments:
            text = str(segment.text or "")
            for match in proper_re.finditer(text):
                value = _norm(match.group(0))
                if value.split()[0] in _COMMON_CAPITALIZED:
                    continue
                proper_counts[value] += 1
                remember(value, segment)
            low = text.casefold()
            for match in word_re.finditer(low):
                value = match.group(0)
                if value in _COMMON_TERMS:
                    continue
                term_counts[value] += 1
            for match in technical_phrase_re.finditer(low):
                value = _norm(match.group(0))
                term_counts[value] += 5
                remember(value, segment)

        proper = [name for name, count in proper_counts.most_common(120) if count >= 2]
        # Favor rare/specialized vocabulary but keep enough recurring terms for titles/institutions.
        term_scored = sorted(
            ((value, count) for value, count in term_counts.items() if 1 <= count <= 45),
            key=lambda pair: (" " in pair[0] or "-" in pair[0], min(pair[1], 12), len(pair[0])),
            reverse=True,
        )[:160]
        records: list[dict[str, Any]] = []
        for value in proper:
            records.append({"candidate": value, "kind_hint": "proper", "frequency": proper_counts[value], "contexts": contexts.get(value, [])})
        for value, count in term_scored:
            if value in proper_counts:
                continue
            if value not in contexts:
                for segment in segments:
                    if re.search(rf"\b{re.escape(value)}\b", str(segment.text or ""), re.I):
                        remember(value, segment)
                        if len(contexts.get(value, [])) >= 1:
                            break
            records.append({"candidate": value, "kind_hint": "term", "frequency": count, "contexts": contexts.get(value, [])})
        return records

    @staticmethod
    def _memory_from_data(data: dict[str, Any]) -> BookMemory:
        memory = apply_reference_profile(BookMemory(glossary=dict(REFERENCE_GLOSSARY_SEED)))
        style = dict(data.get("style") or {})
        for attr in ("narrative_voice", "rhythm", "dialogue", "humor"):
            if style.get(attr):
                setattr(memory.style, attr, str(style[attr]))
        canon = {str(k): _norm(v) for k, v in dict(data.get("canonicals") or {}).items() if _norm(v)}
        glossary = {str(k): _norm(v) for k, v in dict(data.get("glossary") or {}).items() if _norm(v)}
        memory.glossary.update(glossary)
        memory.glossary.update(canon)
        for name, desc in dict(data.get("characters") or {}).items():
            clean = _norm(desc)
            ru = canon.get(str(name), "")
            memory.characters[str(name)] = (f"ru={ru};" if ru else "") + clean
        memory.rolling_summary = _norm(data.get("summary") or "")[:7000]
        return memory

    def build(self, segments: list[Segment]) -> tuple[BookMemory, dict[str, Any]]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text("utf-8"))
                if isinstance(data, dict) and data.get("version") == 10:
                    self.stats.update({"cache_hit": True, "glossary": len(data.get("glossary") or {}), "canon": len(data.get("canonicals") or {}), "characters": len(data.get("characters") or {})})
                    return self._memory_from_data(data), dict(self.stats)
            except Exception:
                pass

        records = self._candidate_records(segments)
        aggregate: dict[str, Any] = {"version": 10, "glossary": {}, "canonicals": {}, "characters": {}, "style": {}, "summary": ""}
        system = """Build SOURCE-ONLY EN→RU translation intelligence for a literary novel.
You receive candidates extracted across the ENTIRE book with source contexts. Classify only genuine recurring proper names,
characters, institutions, titles, technical objects/materials and rare high-impact terms. Omit ordinary vocabulary.
For proper names choose one stable Russian spelling. For characters infer gender only when source context supports it.
For technical terms choose the precise Russian denotation appropriate to the shown scene, not a generic dictionary gloss.
Return ONLY JSON {"glossary":{"EN":"RU"},"canonicals":{"EN name":"RU"},"characters":{"EN name":"gender=male|female|unknown;role=...;voice=..."}}."""
        batch_size = max(30, int(os.getenv("BOOKAI_V10_BIBLE_BATCH") or "45"))
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            try:
                obj = _giga_json(self.backend, system, {"candidates": batch}, max_tokens=5200)
                self.stats["analysis_calls"] += 1
            except Exception as exc:
                print(f"[v10-bible] candidate_batch={start // batch_size + 1} error={type(exc).__name__}", flush=True)
                continue
            for key in ("glossary", "canonicals", "characters"):
                rows = obj.get(key)
                if isinstance(rows, dict):
                    aggregate[key].update({str(k): _norm(v) for k, v in rows.items() if str(k).strip() and _norm(v)})

        # Distributed style sample: evenly spaced excerpts from beginning to end, never one contiguous 32k slice.
        sample_count = min(28, len(segments))
        style_sample = []
        if sample_count:
            for n in range(sample_count):
                index = round(n * (len(segments) - 1) / max(1, sample_count - 1))
                seg = segments[index]
                style_sample.append({"chapter": seg.chapter, "text": _norm(seg.text)[:650]})
        style_system = """Infer a compact EN→RU literary style bible from excerpts distributed across a whole novel.
Return ONLY JSON {"style":{"narrative_voice":"...","rhythm":"...","dialogue":"...","humor":"..."},"summary":"..."}.
Describe translation instructions, not plot invention. Preserve restraint, irony, technical precision and character voice."""
        try:
            obj = _giga_json(self.backend, style_system, {"excerpts": style_sample}, max_tokens=2600)
            self.stats["analysis_calls"] += 1
            if isinstance(obj.get("style"), dict):
                aggregate["style"] = obj["style"]
            aggregate["summary"] = _norm(obj.get("summary") or "")
        except Exception as exc:
            print(f"[v10-bible] style error={type(exc).__name__}", flush=True)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), "utf-8")
        self.stats.update({
            "candidate_records": len(records),
            "glossary": len(aggregate["glossary"]),
            "canon": len(aggregate["canonicals"]),
            "characters": len(aggregate["characters"]),
            "whole_book_segments_scanned": len(segments),
        })
        print("[v10-bible] " + json.dumps(self.stats, ensure_ascii=False), flush=True)
        return self._memory_from_data(aggregate), dict(self.stats)


_TAG_RE = re.compile(r"<s\s+id=[\"']?(s\d{6})[\"']?\s*>(.*?)</s>", re.I | re.S)


class GigaPrimaryTransport(GigaChatLightningV3Backend):
    """Plain tagged transport: one logical batch is normally one API call.

    No JSON Schema. Missing ids are retried once as a compact missing-only batch;
    there is no recursive 12->6->3->1 split cascade.
    """

    name = "gigachat-3-lightning-v10-tagged"

    @staticmethod
    def parse_tagged(text: str, expected: set[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for sid, value in _TAG_RE.findall(str(text or "")):
            sid = sid.casefold()
            clean = str(value or "").strip()
            if sid in expected and clean:
                out[sid] = clean
        return out

    def _tag_prompt(self, batch: list[Segment], memory: BookMemory, source_segments: list[Segment] | None, *, retry: bool = False) -> str:
        style = memory.style
        glossary = self._relevant_glossary(batch, memory) or "нет"
        characters = self._relevant_characters(batch, memory) or "нет"
        context = self._context_for_batch(batch, source_segments) or "нет"
        targets = "\n".join(f'<src id="{s.id}">{s.text}</src>' for s in batch)
        retry_note = "Это повтор только пропущенных ID; верни КАЖДЫЙ указанный ID." if retry else ""
        return f"""Профессиональный литературный перевод EN→RU. Переведи только SRC-блоки.
Не сокращай, не пересказывай, не добавляй факты. Сохраняй субъект/объект действия, числа, отрицания, причинность,
хронологию, технический смысл, пол персонажей и все смысловые части. Русский должен быть естественной опубликованной прозой.

VOICE: {style.narrative_voice}
RHYTHM: {style.rhythm}
DIALOGUE: {style.dialogue}
HUMOR: {style.humor}
CHARACTERS: {characters}
GLOSSARY: {glossary}
CONTEXT_ONLY: {context}
{retry_note}

TARGETS:
{targets}

ФОРМАТ: ровно по одному блоку на каждый id, без JSON и комментариев:
<s id="s000001">полный русский перевод</s>"""

    def _call_tagged(self, batch: list[Segment], memory: BookMemory, source_segments: list[Segment] | None, *, retry: bool = False) -> dict[str, str]:
        client = self._ensure_client()
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты точный литературный переводчик. Не выводи ничего кроме требуемых <s id=...>...</s> блоков."},
                {"role": "user", "content": self._tag_prompt(batch, memory, source_segments, retry=retry)},
            ],
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
        }
        response = client.chat(request)
        self.usage.add(_usage(response), calls=1)
        expected = {s.id for s in batch}
        return self.parse_tagged(str(response.choices[0].message.content or ""), expected)

    def translate_many(self, segments: list[Segment], memory: BookMemory, *, source_segments: list[Segment] | None = None) -> tuple[dict[str, str], dict[str, str]]:
        result: dict[str, str] = {}
        errors: dict[str, str] = {}
        regular: list[Segment] = []
        for segment in segments:
            heading = self._deterministic_heading(segment)
            if heading:
                result[segment.id] = heading
            else:
                regular.append(segment)
        for batch in self._batches(regular):
            try:
                rows = self._call_tagged(batch, memory, source_segments, retry=False)
            except Exception as exc:
                rows = {}
                print(f"[v10-primary] batch_error={type(exc).__name__} segments={len(batch)}", flush=True)
            missing = [s for s in batch if s.id not in rows]
            if missing:
                try:
                    retry_rows = self._call_tagged(missing, memory, source_segments, retry=True)
                    rows.update(retry_rows)
                except Exception as exc:
                    print(f"[v10-primary] missing_retry_error={type(exc).__name__} segments={len(missing)}", flush=True)
            result.update(rows)
            for segment in batch:
                if segment.id not in result:
                    errors[segment.id] = "missing after one compact tagged retry"
        return result, errors


@dataclass(frozen=True)
class V10Issue:
    id: str
    code: str
    mode: str  # local | semantic
    severity: str
    reason: str


class DeterministicQA:
    _LATIN = re.compile(r"\b[A-Za-z]{2,}\b")
    _BEAT = re.compile(r"[.!?](?:['\"»”])?(?:\s|$)")

    @staticmethod
    def _beats(text: str) -> int:
        return len(DeterministicQA._BEAT.findall(str(text or "")))

    @staticmethod
    def _gender_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        low = str(target or "").casefold().replace("ё", "е")
        out: list[V10Issue] = []
        for name, desc in memory.characters.items():
            if not re.search(rf"\b{re.escape(name)}\b", source, re.I):
                continue
            gender = re.search(r"gender=(male|female)", str(desc), re.I)
            ru = re.search(r"ru=([^;]+)", str(desc), re.I)
            if not gender:
                continue
            if ru and ru.group(1).strip().casefold().replace("ё", "е") not in low:
                continue
            value = gender.group(1).casefold()
            if value == "male" and re.search(r"\b(?:сказала|говорила|ответила|спросила|подумала|заметила)\b", low):
                out.append(V10Issue(segment.id, "character_gender", "local", "hard", f"{name} is male but Russian agreement is feminine"))
            if value == "female" and re.search(r"\b(?:сказал|говорил|ответил|спросил|подумал|заметил)\b", low):
                out.append(V10Issue(segment.id, "character_gender", "local", "hard", f"{name} is female but Russian agreement is masculine"))
            if re.search(r"\bking\b", source, re.I) and value == "male" and "королева" in low:
                out.append(V10Issue(segment.id, "character_gender", "local", "hard", "male King rendered as queen"))
        return out

    @staticmethod
    def _glossary_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        low_target = str(target or "").casefold().replace("ё", "е")
        out: list[V10Issue] = []
        for en, ru in memory.glossary.items():
            en_s = str(en or "").strip()
            ru_s = _norm(ru)
            if not en_s or not ru_s or en_s[:1].isupper():
                continue
            if not re.search(rf"\b{re.escape(en_s)}\b", source, re.I):
                continue
            first = re.findall(r"[А-Яа-яЁё]+", ru_s)
            if not first:
                continue
            stem = first[0].casefold().replace("ё", "е")[: max(4, min(7, len(first[0]) - 1))]
            if stem and stem not in low_target:
                out.append(V10Issue(segment.id, "glossary_term", "local", "major", f"source term {en_s!r} should preserve book-bible term {ru_s!r}"))
        return out

    def scan_segment(self, segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
        source = str(segment.text or "")
        ru = str(target or "")
        out: list[V10Issue] = []
        numeric = compare_numeric_fidelity(source, ru)
        if not numeric.get("ok", True):
            out.append(V10Issue(segment.id, "numeric", "local", "hard", f"missing source numeric values {numeric.get('missing')}"))
        # Quantities not represented by the legacy numeric parser.
        low = ru.casefold().replace("ё", "е")
        if re.search(r"\b(?:a|one)\s+dozen\b", source, re.I) and not re.search(r"\b(?:двенадцат|12|дюжин)\w*\b", low):
            out.append(V10Issue(segment.id, "quantity_dozen", "local", "hard", "dozen=12 is not preserved"))
        if re.search(r"\b(?:a|one)\s+dozen\b", source, re.I) and "полдюжин" in low:
            out.append(V10Issue(segment.id, "quantity_dozen", "local", "hard", "dozen=12 was reduced to half-dozen=6"))
        if re.search(r"\bquarter[- ]inch\b", source, re.I) and not (re.search(r"четверт\w*\s+дюйм", low) or re.search(r"6[,.]35\s*мм", low)):
            out.append(V10Issue(segment.id, "quarter_inch", "local", "hard", "quarter-inch quantity not preserved"))
        for code, check in (
            ("question", compare_question_fidelity(source, ru)),
            ("material", compare_material_fidelity(source, ru)),
            ("order", compare_order_fidelity(source, ru)),
            ("short_omission", compare_short_omission_fidelity(source, ru)),
        ):
            if not check.get("ok", True):
                mode = "semantic" if code == "short_omission" else "local"
                out.append(V10Issue(segment.id, code, mode, "hard", json.dumps(check, ensure_ascii=False)))

        ratio = len(ru) / max(1, len(source)) if source else 1.0
        src_beats = self._beats(source)
        ru_beats = self._beats(ru)
        if len(source) >= 145 and src_beats >= 3 and ratio < 0.63 and src_beats - ru_beats >= 1:
            out.append(V10Issue(segment.id, "omission", "semantic", "hard", f"multi-beat coverage collapse ratio={ratio:.2f} beats={src_beats}->{ru_beats}"))
        elif len(source) >= 320 and src_beats >= 5 and ratio < 0.75 and src_beats - ru_beats >= 2:
            out.append(V10Issue(segment.id, "omission", "semantic", "hard", f"long paragraph lost multiple beats ratio={ratio:.2f} beats={src_beats}->{ru_beats}"))

        latin = [w for w in self._LATIN.findall(ru) if w.casefold() not in {"chapter"}]
        if latin:
            out.append(V10Issue(segment.id, "latin_leak", "local", "hard", "raw Latin remains: " + ", ".join(latin[:6])))
        if "лазер" in low and not re.search(r"\blaser\b", source, re.I):
            out.append(V10Issue(segment.id, "invented_specific", "semantic", "hard", "translation invented laser absent from source"))

        # High-confidence actor reversal: active English subject/object question became impersonal/plural Russian.
        actor = re.search(r"\bDid\s+(he|she)\s+[A-Za-z'-]+\s+(you|him|her|them)\s*\?", source, re.I)
        if actor and re.search(r"\b(?:тебя|его|ее|её|их)\b", low) and re.search(r"\b[а-я]+ли\b", low) and not re.search(r"\b(?:он|она)\b", low):
            out.append(V10Issue(segment.id, "actor_relation", "semantic", "hard", "active subject/object relation may have become impersonal/plural"))

        out.extend(self._gender_issues(segment, ru, memory))
        out.extend(self._glossary_issues(segment, ru, memory))
        # Deduplicate by code/mode/reason without hiding multiple categories.
        unique = {(row.code, row.mode, row.reason): row for row in out}
        return list(unique.values())

    def scan(self, segments: list[Segment], translated: dict[str, str], memory: BookMemory) -> list[V10Issue]:
        out: list[V10Issue] = []
        for segment in segments:
            target = str(translated.get(segment.id) or "")
            if not target:
                out.append(V10Issue(segment.id, "missing", "semantic", "hard", "segment has no translation"))
                continue
            out.extend(self.scan_segment(segment, target, memory))
        return out

    @staticmethod
    def semantic_risk(segment: Segment, memory: BookMemory) -> int:
        text = str(segment.text or "")
        score = 0
        rules = (
            (r"\b(?:either|neither|both|former|latter|the other)\b", 4),
            (r"\b(?:not|never|without|unless|hardly|scarcely|barely)\b", 2),
            (r"\b(?:because|therefore|although|whereas|since|until|by the time)\b", 2),
            (r"\b(?:outnumber|more than|less than|fewer than|twice|half|quarter|dozen)\b", 2),
            (r"\b(?:he|she|they|him|her|them|his|their)\b", 1),
        )
        for pattern, weight in rules:
            if re.search(pattern, text, re.I):
                score += weight
        if len(text) >= 420:
            score += 3
        if len(text) >= 700:
            score += 2
        for term in memory.glossary:
            if str(term).casefold() in text.casefold() and (" " in str(term) or "-" in str(term)):
                score += 2
                break
        return score


class GigaSpanPatcher:
    def __init__(self, backend: GigaChatLightningV3Backend, qa: DeterministicQA):
        self.backend = backend
        self.qa = qa
        self.stats = {"calls": 0, "requested": 0, "accepted": 0, "rejected": 0}

    def repair(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[str]:
        local_by_id: dict[str, list[V10Issue]] = {}
        for issue in issues:
            if issue.mode == "local":
                local_by_id.setdefault(issue.id, []).append(issue)
        if not local_by_id:
            return []
        by_id = {s.id: s for s in targets}
        rows = []
        for sid, defects in list(local_by_id.items())[:24]:
            segment = by_id.get(sid)
            if not segment:
                continue
            rows.append({
                "id": sid,
                "source": segment.text,
                "current_ru": translated.get(sid, ""),
                "defects": [{"code": d.code, "reason": d.reason} for d in defects],
            })
        self.stats["requested"] += len(rows)
        changed: list[str] = []
        system = """You are a FAST local EN→RU correction editor. Each row has a proven local defect.
Do NOT rewrite the paragraph. Return the smallest exact textual replacement needed in current_ru.
`find` MUST be an exact substring occurring exactly once in current_ru; `replace` is clean final Russian prose.
Preserve all unrelated wording. Fix numbers, units, materials, question punctuation, ordering, gender, glossary terms or raw Latin only.
Return ONLY JSON {"patches":[{"id":"...","find":"exact old substring","replace":"new substring"}]}.
If a safe local patch is impossible, omit that id; never return a full-paragraph rewrite."""
        batch_size = 10
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            try:
                obj = _giga_json(self.backend, system, {"items": batch}, max_tokens=3000)
                self.stats["calls"] += 1
            except Exception as exc:
                print(f"[v10-patcher] error={type(exc).__name__}", flush=True)
                continue
            patches = obj.get("patches") or []
            parsed = {str(p.get("id") or ""): p for p in patches if isinstance(p, dict)}
            for row in batch:
                sid = row["id"]
                patch = parsed.get(sid) or {}
                find = str(patch.get("find") or "")
                replace = str(patch.get("replace") or "")
                current = str(translated.get(sid) or "")
                if not find or not replace or current.count(find) != 1 or len(find) > max(220, len(current) * 0.55):
                    self.stats["rejected"] += 1
                    continue
                candidate = current.replace(find, replace, 1)
                segment = by_id[sid]
                before = self.qa.scan_segment(segment, current, memory)
                after = self.qa.scan_segment(segment, candidate, memory)
                before_local = sum(i.mode == "local" and i.severity == "hard" for i in before)
                after_local = sum(i.mode == "local" and i.severity == "hard" for i in after)
                after_semantic = sum(i.mode == "semantic" and i.severity == "hard" for i in after)
                before_semantic = sum(i.mode == "semantic" and i.severity == "hard" for i in before)
                if after_local < before_local and after_semantic <= before_semantic:
                    translated[sid] = candidate
                    changed.append(sid)
                    self.stats["accepted"] += 1
                else:
                    self.stats["rejected"] += 1
        return changed


class DeepSeekSemanticSpecialist:
    def __init__(self, provider: Any, qa: DeterministicQA, max_segments: int = 8):
        self.provider = provider
        self.qa = qa
        self.max_segments = max(1, max_segments)
        self.stats: dict[str, Any] = {"calls": 0, "selected": 0, "accepted": 0, "selected_ids": []}

    def _select(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[Segment]:
        by_id = {s.id: s for s in targets}
        forced = {i.id for i in issues if i.mode == "semantic" and i.severity == "hard"}
        ranked = sorted(
            targets,
            key=lambda s: ((s.id in forced), self.qa.semantic_risk(s, memory), len(s.text)),
            reverse=True,
        )
        selected: list[Segment] = []
        for segment in ranked:
            risk = self.qa.semantic_risk(segment, memory)
            if segment.id not in forced and risk < 4:
                continue
            selected.append(segment)
            if len(selected) >= self.max_segments:
                break
        return selected

    def repair(self, targets: list[Segment], translated: dict[str, str], memory: BookMemory, issues: list[V10Issue]) -> list[str]:
        selected = self._select(targets, translated, memory, issues)
        if not selected:
            return []
        index = {s.id: i for i, s in enumerate(targets)}
        issue_map: dict[str, list[str]] = {}
        for issue in issues:
            issue_map.setdefault(issue.id, []).append(f"{issue.code}: {issue.reason}")
        items = []
        for segment in selected:
            i = index[segment.id]
            items.append({
                "id": segment.id,
                "source": segment.text,
                "current_ru": translated.get(segment.id, ""),
                "known_defects": issue_map.get(segment.id, []),
                "before_en": [s.text for s in targets[max(0, i - 2):i]],
                "after_en": [s.text for s in targets[i + 1:i + 3]],
            })
        self.stats.update({"selected": len(selected), "selected_ids": [s.id for s in selected]})
        system = """You are the ONE expensive semantic specialist in a cost-sensitive EN→RU literary pipeline.
Compare each source, current Russian, neighboring English and known deterministic defects. Repair only genuine semantic/publication defects:
omissions, invented facts, actor/action/object reversals, antecedents, chronology, causality, negation/modality, difficult word sense or technical denotation.
Do not rewrite merely for taste. If change=true, corrected_ru MUST be the complete publication-ready Russian translation of exactly that source segment.
Preserve every fact, number, name and established term. Return exactly one row per id.
ONLY JSON {"items":[{"id":"...","change":true,"corrected_ru":"...","confidence":0.0,"reason":"..."}]}"""
        try:
            raw = self.provider.complete(system, json.dumps({"items": items}, ensure_ascii=False), temperature=0.0)
            obj = _json_from_text(raw)
            self.stats["calls"] = 1
        except Exception as exc:
            print(f"[v10-deepseek] error={type(exc).__name__}: {exc}", flush=True)
            return []
        parsed = {str(row.get("id") or ""): row for row in (obj.get("items") or []) if isinstance(row, dict)}
        changed: list[str] = []
        by_id = {s.id: s for s in selected}
        for sid, segment in by_id.items():
            row = parsed.get(sid) or {}
            if type(row.get("change")) is not bool or not row.get("change"):
                continue
            try:
                confidence = float(row.get("confidence") or 0)
            except Exception:
                confidence = 0.0
            candidate = _norm(row.get("corrected_ru") or "")
            if confidence < 0.62 or not candidate:
                continue
            current = str(translated.get(sid) or "")
            before = self.qa.scan_segment(segment, current, memory)
            after = self.qa.scan_segment(segment, candidate, memory)
            before_hard = sum(i.severity == "hard" for i in before)
            after_hard = sum(i.severity == "hard" for i in after)
            # Semantic specialist may fix a detector-blind issue; allow equal deterministic score, never worse.
            if after_hard <= before_hard:
                translated[sid] = candidate
                changed.append(sid)
        self.stats["accepted"] = len(changed)
        return changed


def issue_summary(issues: list[V10Issue]) -> dict[str, Any]:
    return {
        "count": len(issues),
        "hard": sum(i.severity == "hard" for i in issues),
        "by_code": dict(Counter(i.code for i in issues)),
        "ids": sorted({i.id for i in issues}),
    }
