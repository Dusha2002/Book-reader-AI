from __future__ import annotations

from pathlib import Path

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalV10QualityQA


ROOT = Path(__file__).resolve().parents[1]


def _segment(text: str, sid: str = "s1") -> Segment:
    return Segment(id=sid, text=text, locator="test", chapter="Chapter One")


def _technical_memory() -> BookMemory:
    memory = BookMemory()
    memory.style.narrative_voice = "Technical academic expository prose."
    return memory


def test_release_facade_delegates_to_crossdomain_layer():
    text = (ROOT / "src/bookai/release_final.py").read_text("utf-8")
    assert "v10_crossdomain_release" in text
    assert "class FinalV10QualityQA" not in text


def test_general_release_has_no_fixture_book_vocabulary():
    paths = [
        ROOT / "src/bookai/v10_general_release.py",
        ROOT / "src/bookai/v10_universal_release.py",
        ROOT / "src/bookai/v10_crossdomain_release.py",
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
        _technical_memory(),
    )
    assert not any(issue.code == "latin_leak" for issue in issues)
    assert not any(issue.code == "acronym_fidelity" for issue in issues)


def test_missing_acronym_is_hard_in_technical_prose():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment("The model runs on faster CPUs."),
        "Модель работает на более быстрых процессорах.",
        _technical_memory(),
    )
    assert any(issue.code == "acronym_fidelity" and issue.severity == "hard" for issue in issues)


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
    memory = _technical_memory()
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
    bad = "Облако содержало стальные прутки размером полдюйма в длину и три фута, заострённые с одного конца."
    issues = qa.scan_segment(_segment(source), bad, BookMemory())
    assert any(issue.code in {"half_inch", "dimension_relation"} and issue.severity == "hard" for issue in issues)


def test_half_inch_modifier_allows_distinct_thickness_and_length():
    qa = FinalV10QualityQA()
    source = "The cloud contained half-inch steel rods, three feet long and sharpened at one end."
    good = "Облако содержало стальные прутки толщиной в полдюйма и длиной в три фута, заострённые с одного конца."
    issues = qa.scan_segment(_segment(source), good, BookMemory())
    assert not any(issue.code == "dimension_relation" for issue in issues)


def test_dynamic_technical_glossary_is_hard_in_academic_domain():
    qa = FinalV10QualityQA()
    memory = _technical_memory()
    memory.glossary["kernel machines"] = "ядерные методы"
    issues = qa.scan_segment(
        _segment("Kernel machines achieved good results."),
        "Машины опорных векторов показали хорошие результаты.",
        memory,
    )
    assert any(issue.code == "technical_term" and issue.severity == "hard" for issue in issues)


def test_dynamic_technical_glossary_accepts_inflection():
    qa = FinalV10QualityQA()
    memory = _technical_memory()
    memory.glossary["long short-term memory"] = "долгая краткосрочная память"
    issues = qa.scan_segment(
        _segment("They introduced long short-term memory or LSTM."),
        "Они ввели сеть с долгой краткосрочной памятью, или LSTM.",
        memory,
    )
    assert not any(issue.code == "technical_term" for issue in issues)


def test_citation_surname_spelling_is_preserved_in_academic_domain():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(
        _segment("Hochreiter and Schmidhuber (1997) introduced the model."),
        "Хохрайтер и Шмидхубер (1997) представили модель.",
        _technical_memory(),
    )
    assert any(issue.code == "citation_fidelity" and issue.severity == "hard" for issue in issues)


def test_cross_segment_source_continuation_cannot_be_closed_early():
    qa = FinalV10QualityQA()
    segments = [
        _segment("The increase in model size, due to faster CPUs,", "s1"),
        _segment("the advent of GPUs, is an important trend.", "s2"),
    ]
    translated = {
        "s1": "Рост размера моделей, обусловленный более быстрыми CPU.",
        "s2": "появление GPU — важная тенденция.",
    }
    issues = qa.scan(segments, translated, _technical_memory())
    assert any(issue.code == "segment_boundary" and issue.severity == "hard" for issue in issues)
