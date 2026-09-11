from __future__ import annotations

import json

from bookai.literary_context import (
    SourceContextIndex,
    atomic_persist,
    lint_style_rule,
    locked_glossary_violations,
    memory_with_context,
    select_refinement_targets,
)
from bookai.models import BookMemory, Segment


def segment(i: int, text: str, chapter: str = "c1") -> Segment:
    return Segment(id=f"s{i:06d}", text=text, locator=f"/{i}", chapter=chapter)


def test_style_lint_accepts_abstract_rule_and_rejects_literal_examples():
    assert lint_style_rule(
        "Keep irony understated and let comic reversals land late in the sentence."
    ) == []
    flags = lint_style_rule('Use phrases such as "cold iron" and "black rain" throughout the book.')
    assert "quoted_example" in flags
    assert "example_marker" in flags


def test_distant_context_retrieval_prefers_sparse_recurrence():
    segments = [
        segment(0, "The bronze axle fractured under the loaded cart."),
        segment(1, "He walked home in silence."),
        segment(2, "Nothing useful happened here."),
        segment(3, "A short exchange followed."),
        segment(4, "They waited until morning."),
        segment(5, "Someone closed the gate."),
        segment(6, "The weather turned colder."),
        segment(7, "Another unrelated paragraph."),
        segment(8, "The bronze axle fracture explained why the cart wheel had failed."),
        segment(9, "They moved on."),
    ]
    index = SourceContextIndex(segments)
    found = index.retrieve([segments[0]], k=1)
    assert [item.id for item in found] == [segments[8].id]


def test_locked_glossary_accepts_russian_inflection_and_flags_drift():
    original = [segment(1, "Valens looked back at Orsea.")]
    ok = {original[0].id: "Валенса окликнула Орсеа."}
    assert locked_glossary_violations(
        original,
        ok,
        {"Valens": "Валенс", "Orsea": "Орсеа"},
    ) == {}

    bad = {original[0].id: "Валент посмотрел на Орсию."}
    violations = locked_glossary_violations(
        original,
        bad,
        {"Valens": "Валенс", "Orsea": "Орсеа"},
    )
    assert set(violations[original[0].id]) == {"Valens", "Orsea"}


def test_refinement_selector_prefers_complex_or_glossary_risk():
    targets = [
        segment(1, "Short sentence."),
        segment(2, "Although " + ("complex material, " * 70) + "he continued; however, nobody answered."),
        segment(3, "Valens answered."),
    ]
    translations = {
        targets[0].id: "Короткое предложение.",
        targets[1].id: "Сложный перевод " * 45,
        targets[2].id: "Он ответил.",
    }
    selected = select_refinement_targets(
        targets,
        translations,
        limit=2,
        locked_violations={targets[2].id: ["Valens"]},
    )
    ids = {item.id for item in selected}
    assert targets[1].id in ids
    assert targets[2].id in ids


def test_memory_context_is_bounded_and_contains_retrieved_source():
    memory = BookMemory(rolling_summary="old")
    distant = [segment(20, "A recurring mechanical detail returns here.", "c2")]
    updated = memory_with_context(
        memory,
        book_synopsis="book facts",
        chapter_digest="chapter facts",
        distant=distant,
    )
    assert "BOOK CONTINUITY SYNOPSIS" in updated.rolling_summary
    assert "CURRENT CHAPTER DIGEST" in updated.rolling_summary
    assert "s000020" in updated.rolling_summary
    assert len(updated.rolling_summary) <= 14000


def test_atomic_persist_replaces_checkpoint_without_tmp_file(tmp_path):
    path = tmp_path / "state.json"
    memory = BookMemory(title="T")
    atomic_persist(path, {"pipeline_version": "x"}, {"s1": "текст"}, memory)
    data = json.loads(path.read_text("utf-8"))
    assert data["translations"] == {"s1": "текст"}
    assert data["memory"]["title"] == "T"
    assert not path.with_suffix(path.suffix + ".tmp").exists()
