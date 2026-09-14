from __future__ import annotations

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalBookBibleBuilder, FinalV10QualityQA
from bookai.v10_entity_graph import harmonize_composite_entities
from bookai.v10_source_bible import SourceOnlyBookBibleBuilder


def test_composite_entity_uses_independently_accepted_component_canons():
    memory = BookMemory(
        glossary={
            "Aren": "Арен",
            "Vale": "Вейл",
            "Aren Vale": "Арин Валь",
        },
        characters={
            "Aren": "ru=Арен;gender=male;kind=person;role=given_name",
            "Vale": "ru=Вейл;gender=unknown;kind=person;role=surname",
            "Aren Vale": "ru=Арин Валь;gender=male;kind=person;role=full_name",
        },
    )

    stats = harmonize_composite_entities(memory)

    assert stats["changed"] == 1
    assert memory.glossary["Aren Vale"] == "Арен Вейл"
    assert "ru=Арен Вейл" in memory.characters["Aren Vale"]


def test_composite_entity_can_harmonize_confirmed_component_without_inventing_rest():
    memory = BookMemory(
        glossary={
            "Aren": "Арен",
            "Aren Vale": "Арин Валь",
        },
        characters={
            "Aren": "ru=Арен;gender=male;kind=person;role=given_name",
            "Aren Vale": "ru=Арин Валь;gender=male;kind=person;role=full_name",
        },
    )

    stats = harmonize_composite_entities(memory)

    assert stats["changed"] == 1
    assert stats["partial_harmonized"] == 1
    assert memory.glossary["Aren Vale"] == "Арен Валь"
    assert "ru=Арен Валь" in memory.characters["Aren Vale"]


def test_composite_entity_is_not_invented_when_no_component_is_confirmed():
    memory = BookMemory(
        glossary={"Aren Vale": "Арен Вейл"},
        characters={"Aren Vale": "ru=Арен Вейл;gender=male;kind=person;role=full_name"},
    )

    stats = harmonize_composite_entities(memory)

    assert stats["changed"] == 0
    assert memory.glossary["Aren Vale"] == "Арен Вейл"


def test_recurring_composite_exposes_standalone_component_to_name_analysis():
    segments = [
        Segment(id="s1", text="The captain greeted Aren Vale at the door.", locator="x", chapter="Chapter One"),
        Segment(id="s2", text="The clerk saw Aren Vale smile, but Aren stayed silent.", locator="y", chapter="Chapter One"),
    ]

    records = FinalBookBibleBuilder._candidate_records(segments)
    proper = {
        str(row.get("candidate") or ""): row
        for row in records
        if row.get("kind_hint") == "proper"
    }

    assert "Aren Vale" in proper
    assert "Aren" in proper
    assert proper["Aren"].get("component_of") == "Aren Vale"


def test_component_canon_is_not_double_enforced_inside_longer_entity():
    memory = BookMemory(
        glossary={"Aren Vale": "Арен Вейл", "Vale": "Вейл"},
        characters={
            "Aren Vale": "ru=Арен Вейл;gender=male;kind=person;role=full_name",
            "Vale": "ru=Вейл;gender=unknown;kind=person;role=surname",
        },
    )
    segment = Segment(
        id="s1",
        text="Aren Vale entered the room.",
        locator="test",
        chapter="Chapter One",
    )

    issues = FinalV10QualityQA().scan_segment(segment, "Арен Вейл вошёл в комнату.", memory)
    assert not any(issue.code == "name_canon" for issue in issues)


def test_component_canon_remains_hard_when_component_is_used_standalone():
    memory = BookMemory(
        glossary={"Aren Vale": "Арен Вейл", "Vale": "Вейл"},
        characters={
            "Aren Vale": "ru=Арен Вейл;gender=male;kind=person;role=full_name",
            "Vale": "ru=Вейл;gender=unknown;kind=person;role=surname",
        },
    )
    segment = Segment(
        id="s1",
        text="Aren Vale entered. Vale stayed by the door.",
        locator="test",
        chapter="Chapter One",
    )

    issues = FinalV10QualityQA().scan_segment(segment, "Арен Вейл вошёл. Валь остался у двери.", memory)
    assert any(issue.code == "name_canon" and "Vale" in issue.reason for issue in issues)


def test_cyrillic_short_i_is_normalized_symmetrically_in_name_canon():
    memory = BookMemory(
        glossary={"Yor": "Йор"},
        characters={"Yor": "ru=Йор;gender=male;kind=person;role=name"},
    )
    segment = Segment(id="s1", text="Yor arrived.", locator="test", chapter="Chapter One")

    issues = FinalV10QualityQA().scan_segment(segment, "Йор прибыл.", memory)
    assert not any(issue.code == "name_canon" for issue in issues)


def test_real_world_entity_can_use_conventional_non_literal_canon():
    allowed = {"Northland": {"candidate": "Northland"}}
    accepted = SourceOnlyBookBibleBuilder._accept_name_item(
        {
            "source": "Northland",
            "ru": "Северный край",
            "kind": "place",
            "real_world": True,
            "gender": "unknown",
            "confidence": 0.95,
        },
        allowed,
        threshold=0.72,
    )
    assert accepted is not None


def test_untrusted_invented_entity_still_requires_spelling_preservation():
    allowed = {"Xariona": {"candidate": "Xariona"}}
    rejected = SourceOnlyBookBibleBuilder._accept_name_item(
        {
            "source": "Xariona",
            "ru": "Кса",
            "kind": "place",
            "real_world": False,
            "gender": "unknown",
            "confidence": 0.99,
        },
        allowed,
        threshold=0.72,
    )
    assert rejected is None


def test_final_release_does_not_use_one_off_legacy_term_seed():
    segments = [
        Segment(
            id="s1",
            text="He adjusted the gorget and continued walking.",
            locator="test",
            chapter="Chapter One",
        )
    ]

    records = FinalBookBibleBuilder._candidate_records(segments)
    technical = {
        str(row.get("candidate") or "").casefold()
        for row in records
        if row.get("kind_hint") == "technical_term"
    }

    assert "gorget" not in technical
