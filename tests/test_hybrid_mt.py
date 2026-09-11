from __future__ import annotations

from bookai.hybrid_mt import (
    build_hymt_prompt,
    complexity_score,
    decide_route,
    routing_summary,
)
from bookai.models import BookMemory, Segment
from bookai.reference_profile import apply_reference_profile


def seg(sid: str, text: str) -> Segment:
    return Segment(id=sid, text=text, locator=sid, chapter="Chapter One")


def test_router_keeps_simple_prose_on_hymt():
    item = seg("s1", "He opened the door, looked outside and went back to the table.")
    decision = decide_route(item)
    assert decision.route == "hymt"
    assert decision.score < 4.8


def test_router_sends_structurally_complex_paragraph_to_deepseek():
    text = (
        "Although the wheel had already begun to turn, which meant that the smaller gear, whose teeth were badly worn, "
        "would engage the shaft before the spring had fully relaxed, he kept pressure on the lever; because if he let go, "
        "the ratchet would jump, the blade would fall, and the whole mechanism—which had taken three weeks to align—would "
        "have to be stripped down, measured again, and rebuilt from the axle outward; and that, he thought, would be funny, "
        "if only it were happening to somebody else. " * 3
    )
    decision = decide_route(seg("s2", text))
    assert decision.route == "deepseek"
    assert "long" in decision.reasons
    assert "clause_pressure" in decision.reasons


def test_hymt_prompt_uses_context_style_and_locked_term():
    memory = apply_reference_profile(BookMemory(glossary={"Valens": "Валенс"}))
    memory.rolling_summary = "Valens is discussing an engineering problem with an officer."
    item = seg("s3", "Valens picked up the lever and looked at it.")
    prompt = build_hymt_prompt(item, memory)
    assert "Russian" in prompt
    assert "Valens -> Валенс" in prompt
    assert "Book/chapter context" in prompt
    assert item.text in prompt


def test_routing_summary_counts_both_routes():
    easy = seg("s1", "He sat down and waited.")
    hard = seg("s2", ("Although the shaft turned, the wheel slipped; " * 80))
    report = routing_summary([easy, hard])
    assert report["total"] == 2
    assert report["hymt"] == 1
    assert report["deepseek"] == 1


def test_complexity_score_is_monotonic_for_added_clause_pressure():
    base = seg("s1", "He looked at the wheel and waited.")
    complex_item = seg("s2", "He looked at the wheel, waited, checked the shaft; turned the lever; checked the spring; and waited again.")
    base_score, _ = complexity_score(base)
    complex_score, _ = complexity_score(complex_item)
    assert complex_score > base_score
