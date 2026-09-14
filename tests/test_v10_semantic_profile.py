from __future__ import annotations

import json

from bookai.models import BookMemory, Segment
from bookai.v10_semantic_profile import SourceSemanticProfileVerifier


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
    })
    memory = BookMemory(domain="literary_fiction")
    verifier = SourceSemanticProfileVerifier(provider, tmp_path / "book-bible.json")

    stats = verifier.analyze(_segments(), memory)

    assert memory.domain == "narrative_nonfiction"
    assert stats["domain_changed"] is True
    assert memory.semantic_hints["surviving branches of our family"].startswith("family members")
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
    })
    memory = BookMemory()
    SourceSemanticProfileVerifier(first, path).analyze(_segments(), memory)
    assert memory.semantic_hints == {}

    second = _Provider({"domain": "other", "domain_confidence": 1.0, "risks": []})
    memory2 = BookMemory()
    stats = SourceSemanticProfileVerifier(second, path).analyze(_segments(), memory2)

    assert stats["cache_hit"] is True
    assert memory2.domain == "narrative_nonfiction"
    assert second.calls == 0
