from __future__ import annotations

import inspect

import bookai.v10_general_release as general_release
from bookai.models import BookMemory, Segment
from bookai.release_final import FinalV10QualityQA


def _seg(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="x", chapter="Chapter One")


def _hard_codes(issues):
    return {issue.code for issue in issues if issue.severity == "hard"}


def test_release_core_has_no_current_book_literals():
    source = inspect.getsource(general_release).casefold()
    forbidden = {
        "mezentine",
        "round wood",
        "butter pass",
        "miel",
        "orsea",
        "ziani",
        "brass bushing",
        "steel cable",
    }
    assert not (forbidden & {literal for literal in forbidden if literal in source})


def test_entity_canon_is_driven_by_book_memory_not_global_constants():
    qa = FinalV10QualityQA()
    memory = BookMemory(
        glossary={"Avarin": "Аварин"},
        characters={"Avarin": "ru=Аварин;gender=male;kind=person;role=engineer"},
    )
    assert "name_canon" in _hard_codes(qa.scan_segment(_seg("Avarin said no."), "Аверин сказал нет.", memory))
    assert "name_canon" not in _hard_codes(qa.scan_segment(_seg("Avarin said no."), "Аварин сказал нет.", memory))


def test_unknown_fallback_token_is_not_a_hard_entity_invariant():
    qa = FinalV10QualityQA()
    memory = BookMemory(
        glossary={"Perhaps": "Возможно"},
        characters={"Perhaps": "ru=Возможно;gender=unknown;kind=other;role=proper_name"},
    )
    assert "name_canon" not in _hard_codes(qa.scan_segment(_seg("Perhaps he left."), "Возможно, он ушёл.", memory))


def test_high_confidence_multiword_book_term_is_dynamic():
    qa = FinalV10QualityQA()
    memory = BookMemory(glossary={"flux bearing": "потоковый подшипник"})
    source = _seg("He replaced the flux bearing before dawn.")
    assert "book_term_canon" in _hard_codes(qa.scan_segment(source, "До рассвета он заменил деталь.", memory))
    assert "book_term_canon" not in _hard_codes(qa.scan_segment(source, "До рассвета он заменил потоковый подшипник.", memory))
    assert "book_term_canon" not in _hard_codes(qa.scan_segment(source, "До рассвета он заменил деталь.", BookMemory()))


def test_material_contrast_rule_is_language_general():
    qa = FinalV10QualityQA()
    source = _seg("The fittings were not copper but bronze.")
    assert "material" in _hard_codes(qa.scan_segment(source, "Фитинги были не латунными, а бронзовыми.", BookMemory()))
    assert "material" not in _hard_codes(qa.scan_segment(source, "Фитинги были не медными, а бронзовыми.", BookMemory()))
