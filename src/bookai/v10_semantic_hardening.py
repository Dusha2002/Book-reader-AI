from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import BookMemory, Segment
from .v10 import V10Issue, _giga_json


_SCHEMA = "v10-semantic-hardening-1"
_EN_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "from", "of", "for",
    "with", "without", "as", "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "their", "they", "we", "you", "he", "she", "not", "can", "could",
    "would", "should", "will", "have", "has", "had", "do", "does", "did", "there", "then", "than",
}
_COMMON_CAPS = {
    "The", "This", "That", "There", "Then", "When", "While", "After", "Before", "Because", "And", "But",
    "You", "Your", "He", "She", "They", "We", "I", "It", "His", "Her", "Their", "What", "Why", "How",
    "Yes", "No", "Well", "Now", "Maybe", "Perhaps", "Chapter",
}
_PRONOUNS_RU = {
    "i": ("я",),
    "you": ("ты", "вы"),
    "he": ("он",),
    "she": ("она",),
    "we": ("мы",),
    "they": ("они",),
}


def _norm_ru(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _source_fingerprint(segments: list[Segment]) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(str(segment.id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(segment.text or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _clean_cyrillic_word(value: str) -> str:
    words = re.findall(r"[А-Яа-яЁё]+", str(value or ""))
    return "".join(words).casefold().replace("ё", "е")


def canon_inflection_present(canon: str, target: str) -> bool:
    """Morphology-tolerant presence check for Russian proper-name canon.

    The old release gate understood many regular endings but missed soft-sign names
    such as Миль -> Миля. This intentionally checks only stems of source-grounded
    proper names; it is not a general fuzzy matcher.
    """
    low = _norm_ru(target)
    words = re.findall(r"[А-Яа-яЁё]+", str(canon or ""))
    if not words:
        return False
    for raw in words:
        word = raw.casefold().replace("ё", "е")
        if re.search(rf"\b{re.escape(word)}\b", low):
            continue
        stems: list[str] = [word]
        if len(word) >= 4 and word[-1] in "ьйая":
            stems.append(word[:-1])
        if len(word) >= 5 and word.endswith(("ий", "ый", "ой")):
            stems.append(word[:-2])
        found = False
        for stem in sorted(set(stems), key=len, reverse=True):
            if len(stem) < 3:
                continue
            if re.search(rf"\b{re.escape(stem)}[а-я]*\b", low):
                found = True
                break
        if not found:
            return False
    return True


def source_has_cross_clause_repeat(source: str) -> bool:
    """Return True when the source itself repeats a meaningful phrase anywhere.

    v10 previously exempted only repetitions inside one sentence, so a legitimate
    repeated phrase such as `pitched battle` in two neighbouring clauses was falsely
    classified as target hallucination. A repeated three-token predicate may contain
    one pronoun/determiner (for example `lifted its head`), so require two content
    words for trigrams while keeping bigrams strict.
    """
    words = [w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(source or ""))]
    if len(words) < 8:
        return False
    for n in (3, 2):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            content = [w for w in gram if w not in _EN_STOP and len(w) >= 4]
            required_content = 2 if n == 3 else n
            if len(content) < required_content:
                continue
            if len(" ".join(gram)) < 10:
                continue
            prev = seen.get(gram)
            if prev is not None and i - prev >= n:
                return True
            seen.setdefault(gram, i)
    return False


def _pronoun_pattern(actor: str) -> str:
    forms = _PRONOUNS_RU.get(actor.casefold(), ())
    if not forms:
        return r"(?!)"
    return r"\b(?:" + "|".join(map(re.escape, forms)) + r")\b"


def actor_polarity_issues(segment: Segment, target: str) -> list[V10Issue]:
    """Catch contrastive actor/negation reversals such as `You didn't X. I did`.

    These are catastrophic despite having all the same nouns and verbs. The check is
    deliberately narrow: it fires only when the English source explicitly contrasts
    two different personal-pronoun actors with `didn't/did not ... [other actor] did`.
    """
    source = str(segment.text or "")
    neg = re.search(
        r"\b(I|you|he|she|we|they)\s+(?:didn't|did\s+not)\s+([A-Za-z][A-Za-z'-]*)",
        source,
        re.I,
    )
    if not neg:
        return []
    tail = source[neg.end():neg.end() + 240]
    pos = re.search(r"\b(I|you|he|she|we|they)\s+did(?:\s+(?:that|it))?\b", tail, re.I)
    if not pos:
        return []
    actor_neg = neg.group(1).casefold()
    actor_pos = pos.group(1).casefold()
    if actor_neg == actor_pos:
        return []

    low = _norm_ru(target)
    neg_actor_re = _pronoun_pattern(actor_neg)
    pos_actor_re = _pronoun_pattern(actor_pos)
    neg_actor_hits = list(re.finditer(neg_actor_re, low, re.I))
    pos_actor_hits = list(re.finditer(pos_actor_re, low, re.I))

    if not neg_actor_hits or not pos_actor_hits:
        return [V10Issue(
            segment.id,
            "actor_polarity",
            "semantic",
            "hard",
            f"source explicitly contrasts {actor_neg!r} as NOT doing an action with {actor_pos!r} as doing it; target loses one contrasted actor",
        )]

    first_neg_actor = neg_actor_hits[0].start()
    later_pos = next((m.start() for m in pos_actor_hits if m.start() > first_neg_actor), None)
    if later_pos is None:
        return [V10Issue(
            segment.id,
            "actor_polarity",
            "semantic",
            "hard",
            f"source actor contrast order is {actor_neg!r} (negative) -> {actor_pos!r} (positive), but target does not preserve it",
        )]

    window = low[max(0, first_neg_actor - 28):min(len(low), first_neg_actor + 58)]
    if not re.search(r"\b(?:не|ни)\b", window):
        return [V10Issue(
            segment.id,
            "actor_polarity",
            "semantic",
            "hard",
            f"source explicitly negates the first actor {actor_neg!r}; target has no local Russian negation around that actor",
        )]
    return []


def measure_semantics_issues(segment: Segment, target: str) -> list[V10Issue]:
    """Guard a few high-risk English measure words whose literal senses are disastrous."""
    source = str(segment.text or "")
    low = _norm_ru(target)
    out: list[V10Issue] = []

    if re.search(r"\bhundredweight\b", source, re.I):
        suspicious = bool(re.search(r"\b(?:сто|сотн\w*)\s+вес\w*\b", low))
        mass_semantics = bool(re.search(r"\b(?:килограмм\w*|кг|фунт\w*|центнер\w*|пуд\w*|тонн\w*|весом)\b", low))
        if suspicious or not mass_semantics:
            out.append(V10Issue(
                segment.id,
                "measure_semantics",
                "semantic",
                "hard",
                "source `hundredweight` is a unit/amount of mass; target must express mass, not a literal 'hundred weights' reading",
            ))

    numeric_stone = re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+stone\b", source, re.I)
    if numeric_stone and re.search(r"\bкам(?:ень|ня|ней|ни)\b", low):
        out.append(V10Issue(
            segment.id,
            "measure_semantics",
            "semantic",
            "hard",
            "source uses `stone` as a weight unit after a number, but target appears to use the literal object 'камень'",
        ))

    if re.search(r"\ba\s+score\s+of\b", source, re.I) and re.search(r"\bсчет\w*\b", low):
        out.append(V10Issue(
            segment.id,
            "measure_semantics",
            "semantic",
            "hard",
            "source `a score of` means roughly twenty, not the noun 'счёт'",
        ))
    return out


def extract_entity_families(segments: list[Segment]) -> list[dict[str, Any]]:
    """Find recurring singular/plural capitalized entity families source-only."""
    counts: Counter[str] = Counter()
    contexts: dict[str, list[str]] = {}
    token_re = re.compile(r"(?<![A-Za-z'])\b[A-Z][a-z]{4,}\b(?!')")
    for segment in segments:
        text = str(segment.text or "")
        for match in token_re.finditer(text):
            token = match.group(0)
            if token in _COMMON_CAPS:
                continue
            counts[token] += 1
            rows = contexts.setdefault(token, [])
            if len(rows) < 3:
                rows.append(" ".join(text.split())[:520])

    families: list[dict[str, Any]] = []
    used: set[str] = set()
    for plural, pcount in counts.most_common():
        if not plural.endswith("s") or len(plural) < 6:
            continue
        singular = plural[:-1]
        scount = counts.get(singular, 0)
        if scount < 1 or pcount < 2 or scount + pcount < 4:
            continue
        if singular.casefold() in used:
            continue
        used.add(singular.casefold())
        families.append({
            "singular": singular,
            "plural": plural,
            "frequency": scount + pcount,
            "contexts": (contexts.get(singular, []) + contexts.get(plural, []))[:5],
        })
    return families[:24]


def build_entity_family_canon(backend: Any, segments: list[Segment], cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build/cache a source-only Russian stem canon for recurring demonym-like families."""
    cache_path = Path(cache_path)
    fingerprint = _source_fingerprint(segments)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text("utf-8"))
        except Exception:
            data = {}
        if data.get("schema") == _SCHEMA and data.get("source_fingerprint") == fingerprint:
            rows = list(data.get("families") or [])
            return rows, {"cache_hit": True, "candidates": int(data.get("candidates") or 0), "accepted": len(rows)}

    candidates = extract_entity_families(segments)
    if not candidates:
        return [], {"cache_hit": False, "candidates": 0, "accepted": 0}

    system = """Build a SOURCE-ONLY Russian morphology canon for recurring English proper-entity singular/plural families.
The candidates may be fictional nationalities, peoples, demonyms, factions, dynasties, or ordinary proper-name families.
Use ONLY the supplied English source contexts plus general linguistic knowledge; no Russian reference translation exists.
Accept a family only when the two forms clearly denote the SAME proper entity/people. For accepted families choose ONE stable
Russian Cyrillic root and natural Russian forms. For invented names preserve visible source spelling and do not replace them
with an unrelated real-world word. `ru_root` must be letters only and be the shared stem used in ALL returned forms.
Return ONLY JSON {"families":[{"singular":"...","plural":"...","ru_root":"...","ru_singular":"...","ru_plural":"...","ru_adjective":"...","confidence":0.0}]}.
"""
    allowed = {(row["singular"], row["plural"]) for row in candidates}
    accepted: list[dict[str, Any]] = []
    try:
        obj = _giga_json(backend, system, {"candidates": candidates}, max_tokens=3200)
    except Exception as exc:
        print(f"[v10-entity-family] error={type(exc).__name__}", flush=True)
        obj = {}

    for row in obj.get("families") or []:
        if not isinstance(row, dict):
            continue
        singular = str(row.get("singular") or "").strip()
        plural = str(row.get("plural") or "").strip()
        if (singular, plural) not in allowed:
            continue
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        root = _clean_cyrillic_word(row.get("ru_root") or "")
        ru_singular = str(row.get("ru_singular") or "").strip()
        ru_plural = str(row.get("ru_plural") or "").strip()
        ru_adjective = str(row.get("ru_adjective") or "").strip()
        forms = [ru_singular, ru_plural, ru_adjective]
        if confidence < 0.82 or len(root) < 4 or not all(re.search(r"[А-Яа-яЁё]", value) for value in forms):
            continue
        normalized_forms = [_clean_cyrillic_word(value) for value in forms]
        if not all(root[:max(4, len(root) - 1)] in value for value in normalized_forms):
            continue
        accepted.append({
            "singular": singular,
            "plural": plural,
            "ru_root": root,
            "ru_singular": ru_singular,
            "ru_plural": ru_plural,
            "ru_adjective": ru_adjective,
            "confidence": confidence,
        })

    payload = {
        "schema": _SCHEMA,
        "source_fingerprint": fingerprint,
        "candidates": len(candidates),
        "families": accepted,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"[v10-entity-family] candidates={len(candidates)} accepted={len(accepted)}", flush=True)
    return accepted, {"cache_hit": False, "candidates": len(candidates), "accepted": len(accepted)}


def install_entity_family_memory(memory: BookMemory, families: list[dict[str, Any]]) -> None:
    for row in families:
        singular = str(row.get("singular") or "").strip()
        plural = str(row.get("plural") or "").strip()
        root = _clean_cyrillic_word(row.get("ru_root") or "")
        forms = "/".join(str(row.get(key) or "").strip() for key in ("ru_singular", "ru_plural", "ru_adjective"))
        if not singular or not plural or not root:
            continue
        desc = f"gender=unknown;kind=demonym_family;ru_root={root};forms={forms};role=source-derived entity family"
        memory.characters[singular] = desc
        memory.characters[plural] = desc
        hint = f"Recurring proper entity family; keep one Russian stem {root!r} and choose the natural singular/plural/adjective morphology from context."
        memory.semantic_hints[singular] = hint
        memory.semantic_hints[plural] = hint


def entity_family_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
    source = str(segment.text or "")
    low = _norm_ru(target)
    roots_seen: set[str] = set()
    out: list[V10Issue] = []
    for source_form, desc in memory.characters.items():
        if "kind=demonym_family" not in str(desc):
            continue
        if not re.search(rf"(?<![A-Za-z]){re.escape(str(source_form))}(?![A-Za-z])", source, re.I):
            continue
        match = re.search(r"(?:^|;)ru_root=([^;]+)", str(desc), re.I)
        root = _clean_cyrillic_word(match.group(1) if match else "")
        if not root or root in roots_seen:
            continue
        roots_seen.add(root)
        if not re.search(rf"\b{re.escape(root)}[а-я]*\b", low):
            out.append(V10Issue(
                segment.id,
                "entity_family_canon",
                "semantic",
                "hard",
                f"source recurring entity family {source_form!r} must use the established Russian stem {root!r}; target uses a different or unrelated form",
            ))
    return out


def semantic_hardening_issues(segment: Segment, target: str, memory: BookMemory) -> list[V10Issue]:
    out: list[V10Issue] = []
    out.extend(actor_polarity_issues(segment, target))
    out.extend(measure_semantics_issues(segment, target))
    out.extend(entity_family_issues(segment, target, memory))
    return out


__all__ = [
    "actor_polarity_issues",
    "build_entity_family_canon",
    "canon_inflection_present",
    "entity_family_issues",
    "extract_entity_families",
    "install_entity_family_memory",
    "measure_semantics_issues",
    "semantic_hardening_issues",
    "source_has_cross_clause_repeat",
]
