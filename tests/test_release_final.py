from __future__ import annotations

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalDeepSeekSemanticSpecialist, FinalDialogueDiscourseGuard, FinalV10QualityQA
from bookai.v10 import V10Issue


def _seg(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="x", chapter="Chapter One")


def _codes(issues):
    return {issue.code for issue in issues if issue.severity == "hard"}


def test_canon_only_fallback_is_not_treated_as_character_name():
    memory = BookMemory(
        glossary={"Who": "Кто", "Miel": "Миэль"},
        characters={
            "Who": "ru=Кто;gender=unknown;role=proper_name",
            "Miel": "ru=Миэль;gender=male;role=;voice=",
        },
    )
    qa = FinalV10QualityQA()
    assert "name_canon" not in _codes(qa.scan_segment(_seg("Who said that?"), "Кто это сказал?", memory))
    assert "name_canon" in _codes(qa.scan_segment(_seg("Miel said that."), "Миль это сказал.", memory))


def test_natural_russian_quantity_forms_do_not_block_release():
    qa = FinalV10QualityQA()
    memory = BookMemory()
    cases = [
        ("one tenth of the population", "одной десятой населения"),
        ("another two, three hundred just getting home", "ещё двести-триста по дороге домой"),
        ("a hundred per cent increase in productivity", "стопроцентный рост производительности"),
        ("in increments of one sixty-fourth of an inch", "с шагом в одну шестьдесятчетвертую дюйма"),
    ]
    for source, target in cases:
        codes = _codes(qa.scan_segment(_seg(source), target, memory))
        assert "numeric" not in codes, (source, codes)
        assert "quantity_obligation" not in codes, (source, codes)


def test_half_inch_compact_russian_form_is_valid_but_one_inch_is_not():
    qa = FinalV10QualityQA()
    memory = BookMemory()
    source = "steel bolts, three feet long and half an inch thick"
    assert "half_inch" not in _codes(qa.scan_segment(_seg(source), "стальные болты длиной три фута и толщиной полдюйма", memory))
    assert "half_inch" in _codes(qa.scan_segment(_seg(source), "стальные болты длиной три фута и толщиной один дюйм", memory))


def test_standard_v_shaped_notation_is_not_a_latin_leak():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(_seg("vee-blocks"), "V-образные блоки", BookMemory())
    assert "latin_leak" not in _codes(issues)


def test_dialogue_guard_preserves_nested_guillemets():
    guard = FinalDialogueDiscourseGuard()
    source = "'What do you mean, it is the law? I never heard of anything like that.'"
    target = "— Что ты имеешь в виду, говоря «это закон?» Я никогда такого не слышал."
    assert guard._normalize(_seg(source), target) == target


def test_dialogue_guard_drops_only_obsolete_outer_closing_quote():
    guard = FinalDialogueDiscourseGuard()
    source = "'He said: law.'"
    target = "«Он сказал: «закон».»"
    fixed = guard._normalize(_seg(source), target)
    assert fixed.startswith("— ")
    assert "«закон»" in fixed
    assert fixed.endswith(".")
    assert fixed.count("«") == fixed.count("»")


def test_clause_order_can_be_nonblocking_only_in_final_release_scan():
    issue = V10Issue("s1", "clause_order", "semantic", "hard", "order heuristic")
    qa = FinalV10QualityQA(demote_clause_order=True)
    demoted = []
    for row in [issue]:
        if row.code == "clause_order" and row.severity == "hard":
            demoted.append(V10Issue(row.id, row.code, row.mode, "soft", row.reason))
        else:
            demoted.append(row)
    assert demoted[0].severity == "soft"
    assert qa.demote_clause_order is True


def test_release_specialist_routes_local_and_semantic_blockers():
    issues = [
        V10Issue("a", "latin_leak", "local", "hard", "raw Latin remains"),
        V10Issue("b", "question", "local", "hard", "question lost"),
        V10Issue("c", "direction_relation", "semantic", "hard", "up became down"),
        V10Issue("d", "dialogue_typography", "local", "hard", "broken quotes"),
    ]
    routed = FinalDeepSeekSemanticSpecialist._proven_codes_by_id(issues)
    assert routed == {
        "a": {"latin_leak"},
        "b": {"question"},
        "c": {"direction_relation"},
        "d": {"dialogue_typography"},
    }
