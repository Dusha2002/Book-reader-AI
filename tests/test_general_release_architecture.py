from __future__ import annotations

from pathlib import Path

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalV10QualityQA


ROOT = Path(__file__).resolve().parents[1]


def _segment(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="test", chapter="Chapter One")


def test_release_facade_delegates_to_universal_layer():
    text = (ROOT / "src/bookai/release_final.py").read_text("utf-8")
    assert "v10_universal_release" in text
    assert "class FinalV10QualityQA" not in text


def test_general_release_has_no_fixture_book_vocabulary():
    paths = [
        ROOT / "src/bookai/v10_general_release.py",
        ROOT / "src/bookai/v10_universal_release.py",
        ROOT / "src/bookai/release_final.py",
    ]
    text = "\n".join(path.read_text("utf-8").casefold() for path in paths)
    fixture_terms = {
        "miel",
        "orsea",
        "ziani",
        "vaatzes",
        "mezentine",
        "sphrantzes",
        "butter pass",
        "round wood",
        "brass bushing",
    }
    leaked = sorted(term for term in fixture_terms if term in text)
    assert not leaked, f"book-specific vocabulary leaked into general release core: {leaked}"


def test_name_consistency_is_driven_by_runtime_book_memory():
    qa = FinalV10QualityQA()
    memory = BookMemory(
        glossary={"Xarion": "Ксарион"},
        characters={"Xarion": "ru=Ксарион;gender=male;kind=person;role=protagonist;voice=reserved"},
    )
    issues = qa.scan_segment(_segment("Xarion answered."), "Зарион ответил.", memory)
    assert any(issue.code == "name_canon" and issue.severity == "hard" for issue in issues)


def test_unknown_book_name_is_not_forced_without_runtime_memory():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(_segment("Xarion answered."), "Зарион ответил.", BookMemory())
    assert not any(issue.code == "name_canon" for issue in issues)


def test_source_acronyms_are_allowed_in_technical_translation():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment("An LSTM can run on a GPU."),
        "LSTM может выполняться на GPU.",
        BookMemory(),
    )
    assert not any(issue.code == "latin_leak" for issue in issues)


def test_quoted_single_letter_symbol_is_allowed_when_source_defines_it():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment('The “M” stands for “modified.”'),
        'Буква «M» означает «модифицированный».',
        BookMemory(),
    )
    assert not any(issue.code == "latin_leak" for issue in issues)


def test_technical_brand_can_remain_latin_when_source_and_domain_support_it():
    qa = FinalV10QualityQA()
    memory = BookMemory()
    memory.style.narrative_voice = "Technical academic expository prose."
    issues = qa.scan_segment(
        _segment("The model is used at Google."),
        "Эта модель используется в Google.",
        memory,
    )
    assert not any(issue.code == "latin_leak" for issue in issues)


def test_fictional_latin_name_is_not_exempted_just_for_capitalization():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment("Xarion answered."),
        "Xarion ответил.",
        BookMemory(),
    )
    assert any(issue.code == "latin_leak" and issue.severity == "hard" for issue in issues)


def test_ordinary_untranslated_english_remains_a_hard_leak():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(_segment("He answered anyway."), "Он ответил anyway.", BookMemory())
    assert any(issue.code == "latin_leak" and issue.severity == "hard" for issue in issues)


def test_mixed_turing_hybrid_is_not_exempted_as_notation():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment("Neural Turing machines can access memory."),
        "Нейронные Turing-машины могут обращаться к памяти.",
        BookMemory(),
    )
    assert any(issue.code == "latin_leak" and issue.severity == "hard" for issue in issues)


def test_half_inch_modifier_cannot_become_length_when_source_has_separate_length():
    qa = FinalV10QualityQA()
    source = "The cloud contained half-inch steel rods, three feet long and sharpened at one end."
    bad = "Облако содержало стальные прутки длиной в полдюйма, длиной в три фута, заострённые с одного конца."
    issues = qa.scan_segment(_segment(source), bad, BookMemory())
    assert any(issue.code == "half_inch" and issue.severity == "hard" for issue in issues)


def test_half_inch_modifier_allows_distinct_thickness_and_length():
    qa = FinalV10QualityQA()
    source = "The cloud contained half-inch steel rods, three feet long and sharpened at one end."
    good = "Облако содержало стальные прутки толщиной в полдюйма и длиной в три фута, заострённые с одного конца."
    issues = qa.scan_segment(_segment(source), good, BookMemory())
    assert not any(issue.code == "half_inch" for issue in issues)
