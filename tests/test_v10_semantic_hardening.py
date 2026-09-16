from bookai.models import BookMemory, Segment
from bookai.v10_semantic_hardening import (
    actor_polarity_issues,
    canon_inflection_present,
    entity_family_issues,
    extract_entity_families,
    measure_semantics_issues,
    source_has_cross_clause_repeat,
)


def seg(text: str, sid: str = "s000001") -> Segment:
    return Segment(id=sid, text=text, locator="/p[1]", chapter="1")


def test_soft_sign_name_inflection_is_accepted():
    assert canon_inflection_present("Миль", "Мы говорили о Миле вчера.")
    assert canon_inflection_present("Миль", "Я встретил Миля Дукаса.")
    assert not canon_inflection_present("Миль", "Я встретил Валенса.")


def test_cross_clause_source_repeat_is_not_target_hallucination_evidence():
    source = (
        "We would have no chance in a pitched battle. A little later he explained that "
        "heavy infantry was what you needed to win a pitched battle."
    )
    assert source_has_cross_clause_repeat(source)


def test_actor_polarity_catches_reversed_contrast():
    source = '"You didn\'t start my war, Orsea," he said. "I did that."'
    bad = "— Войну начал не я, Орсэа, — сказал он. — Я её и начал."
    issues = actor_polarity_issues(seg(source), bad)
    assert [issue.code for issue in issues] == ["actor_polarity"]


def test_actor_polarity_accepts_faithful_contrast():
    source = '"You didn\'t start my war, Orsea," he said. "I did that."'
    good = "— Не ты начал мою войну, Орсэа, — сказал он. — Это сделал я."
    assert actor_polarity_issues(seg(source), good) == []


def test_hundredweight_literal_weight_calque_is_rejected():
    source = "He arrived carrying a hundredweight of books."
    bad = "Он явился, неся сотню весов книг."
    issues = measure_semantics_issues(seg(source), bad)
    assert [issue.code for issue in issues] == ["measure_semantics"]


def test_hundredweight_mass_rendering_is_accepted():
    source = "He arrived carrying a hundredweight of books."
    good = "Он явился, таща около пятидесяти килограммов книг."
    assert measure_semantics_issues(seg(source), good) == []


def test_source_entity_family_extraction_finds_singular_plural_pair():
    rows = [
        seg("A Mezentine refugee arrived.", "s000001"),
        seg("The Mezentines advanced.", "s000002"),
        seg("The Mezentines withdrew.", "s000003"),
        seg("Another Mezentine engineer spoke.", "s000004"),
    ]
    families = extract_entity_families(rows)
    assert any(row["singular"] == "Mezentine" and row["plural"] == "Mezentines" for row in families)


def test_entity_family_canon_rejects_unrelated_russian_variant():
    memory = BookMemory()
    desc = "gender=unknown;kind=demonym_family;ru_root=мезентин;forms=мезентинец/мезентинцы/мезентинский"
    memory.characters["Mezentine"] = desc
    memory.characters["Mezentines"] = desc
    source = seg("The Mezentines moved north.")
    bad = "Меценаты двинулись на север."
    issues = entity_family_issues(source, bad, memory)
    assert [issue.code for issue in issues] == ["entity_family_canon"]


def test_entity_family_canon_accepts_inflected_shared_root():
    memory = BookMemory()
    desc = "gender=unknown;kind=demonym_family;ru_root=мезентин;forms=мезентинец/мезентинцы/мезентинский"
    memory.characters["Mezentine"] = desc
    memory.characters["Mezentines"] = desc
    source = seg("The Mezentine army moved north.")
    good = "Мезентинская армия двинулась на север."
    assert entity_family_issues(source, good, memory) == []
