from __future__ import annotations

import json

from bookai.models import BookMemory, Segment
from bookai.v10_semantic_profile import SourceSemanticProfileVerifier
from bookai.v10_term_verifier import _verification_candidates


class _Provider:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


def _segments() -> list[Segment]:
    return [
        Segment(
            id="s1",
            locator="test",
            chapter="Chapter One",
            text=(
                "I write these recollections for my children, recalling the real events "
                "of my youth and the surviving branches of our family."
            ),
        ),
        Segment(
            id="s2",
            locator="test",
            chapter="Chapter One",
            text="The old phrase kept its customary sense in that century, though a literal reading would mislead us.",
        ),
    ]


def _technical_segments() -> list[Segment]:
    return [
        Segment(
            id="t1",
            locator="test",
            chapter="Section One",
            text="The adaptive flux operator maps each observation into a compact response vector.",
        ),
        Segment(
            id="t2",
            locator="test",
            chapter="Section One",
            text="We then optimize the response vector with a constrained objective.",
        ),
    ]


def test_semantic_profile_can_correct_fiction_guess_to_narrative_nonfiction(tmp_path):
    provider = _Provider({
        "domain": "narrative_nonfiction",
        "domain_confidence": 0.96,
        "domain_reason": "first-person factual recollection addressed to descendants",
        "risks": [
            {
                "source": "surviving branches of our family",
                "meaning": "family members or family lines that remain, not physical remains",
                "kind": "polysemy",
                "confidence": 0.94,
            }
        ],
        "technical_terms": [],
    })
    memory = BookMemory(domain="literary_fiction")
    verifier = SourceSemanticProfileVerifier(provider, tmp_path / "book-bible.json")

    stats = verifier.analyze(_segments(), memory)

    assert memory.domain == "narrative_nonfiction"
    assert stats["domain_changed"] is True
    assert memory.semantic_hints["surviving branches of our family"].startswith("family members")
    assert stats["technical_terms"] == []
    assert provider.calls == 1


def test_semantic_profile_rejects_target_language_hint_and_reuses_cache(tmp_path):
    path = tmp_path / "book-bible.json"
    first = _Provider({
        "domain": "narrative_nonfiction",
        "domain_confidence": 0.92,
        "domain_reason": "memoir framing",
        "risks": [
            {
                "source": "customary sense",
                "meaning": "устоявшееся значение",
                "kind": "register",
                "confidence": 0.95,
            }
        ],
        "technical_terms": [],
    })
    memory = BookMemory()
    SourceSemanticProfileVerifier(first, path).analyze(_segments(), memory)
    assert memory.semantic_hints == {}

    second = _Provider({"domain": "other", "domain_confidence": 1.0, "risks": [], "technical_terms": []})
    memory2 = BookMemory()
    stats = SourceSemanticProfileVerifier(second, path).analyze(_segments(), memory2)

    assert stats["cache_hit"] is True
    assert memory2.domain == "narrative_nonfiction"
    assert second.calls == 0


def test_academic_profile_discovers_exact_source_terms_without_target_wording(tmp_path):
    provider = _Provider({
        "domain": "academic_technical",
        "domain_confidence": 0.99,
        "domain_reason": "formal technical exposition",
        "risks": [],
        "technical_terms": [
            {
                "source": "adaptive flux operator",
                "meaning": "an operator mapping observations into a response representation",
                "confidence": 0.94,
            },
            {
                "source": "XRK",
                "meaning": "standalone acronym that should not become a term candidate",
                "confidence": 0.99,
            },
            {
                "source": "response vector",
                "meaning": "вектор отклика",
                "confidence": 0.99,
            },
        ],
    })
    memory = BookMemory()

    stats = SourceSemanticProfileVerifier(provider, tmp_path / "book-bible.json").analyze(_technical_segments(), memory)

    assert memory.domain == "academic_technical"
    assert stats["technical_term_count"] == 1
    assert stats["technical_terms"][0]["source"] == "adaptive flux operator"
    assert "Russian" not in stats["technical_terms"][0]["meaning"]


def test_focus_term_is_promoted_to_consensus_candidate_without_ru_answer():
    memory = BookMemory(domain="academic_technical")
    focus = _technical_segments()
    discovered = [
        {
            "source": "adaptive flux operator",
            "meaning": "an operator mapping observations into a response representation",
            "confidence": 0.94,
        }
    ]

    rows = _verification_candidates(
        focus,
        memory,
        focus_terms=discovered,
        focus_segments=focus,
    )

    row = next(item for item in rows if item.get("source") == "adaptive flux operator")
    assert row["evidence"] == "focus_semantic_term"
    assert row["current_ru"] == ""
    assert row["source_meaning"].startswith("an operator")
