from __future__ import annotations

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalBookBibleBuilder
from bookai.v10_entity_graph import harmonize_composite_entities


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
        Segment(id="s1", text="Aren Vale entered the room.", locator="x", chapter="Chapter One"),
        Segment(id="s2", text="Later Aren Vale smiled, but Aren stayed silent.", locator="y", chapter="Chapter One"),
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

    # A one-off word may still be discovered later by the generic semantic term
    # verifier if context proves it important, but it must not be injected merely
    # because an older book happened to hard-code it in a legacy source-bible list.
    assert "gorget" not in technical
