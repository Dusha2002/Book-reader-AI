from __future__ import annotations

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalV10QualityQA
from bookai.v10_quantity import compare_quantity_fidelity_v2


def _segment(text: str, sid: str = "s1") -> Segment:
    return Segment(id=sid, text=text, locator="test", chapter="Chapter One")


def _technical_memory() -> BookMemory:
    memory = BookMemory()
    memory.domain = "academic_technical"
    memory.style.narrative_voice = "Technical academic expository prose."
    return memory


def test_source_backed_uppercase_math_symbols_may_remain_latin():
    qa = FinalV10QualityQA()
    source = "The loss L depends on the cost J, and gradients of J are computed with respect to W."
    target = "Потери L зависят от стоимости J, а градиенты J вычисляются по W."
    issues = qa.scan_segment(_segment(source), target, _technical_memory())
    assert not any(issue.code == "latin_leak" for issue in issues)


def test_ordinary_uppercase_pronoun_is_not_hidden_as_notation():
    qa = FinalV10QualityQA()
    source = "I describe the method below."
    target = "I описываю метод ниже."
    issues = qa.scan_segment(_segment(source), target, _technical_memory())
    assert any(issue.code == "latin_leak" and issue.severity == "hard" for issue in issues)


def test_source_backed_cited_library_identifier_may_remain_latin():
    qa = FinalV10QualityQA()
    source = "This approach is used by libraries such as TensorKit (Smith et al., 2022)."
    target = "Этот подход используется в библиотеках, таких как TensorKit (Smith et al., 2022)."
    issues = qa.scan_segment(_segment(source), target, _technical_memory())
    assert not any(issue.code == "latin_leak" for issue in issues)


def test_arbitrary_titlecase_english_is_not_whitelisted_in_technical_prose():
    qa = FinalV10QualityQA()
    source = "This method gives a useful result."
    target = "Этот метод даёт Useful результат."
    issues = qa.scan_segment(_segment(source), target, _technical_memory())
    assert any(issue.code == "latin_leak" and issue.severity == "hard" for issue in issues)


def test_symbolic_digit_function_shape_is_not_treated_as_quantity():
    source = "The penalty 7(4) is added, where 4 denotes the parameter vector."
    target = "Добавляется штраф Ω(θ), где θ обозначает вектор параметров."
    result = compare_quantity_fidelity_v2(source, target)
    assert result["ok"]
    assert result["symbolic_digit_counts"] == {7: 1, 4: 2}


def test_real_quantity_still_fails_next_to_symbolic_notation():
    source = "The penalty 7(4) is added, where 4 denotes the parameter vector, using 9 models."
    target = "Добавляется штраф Ω(θ), где θ обозначает вектор параметров."
    result = compare_quantity_fidelity_v2(source, target)
    assert not result["ok"]
    assert 9 in result["base_missing"]


def test_plain_numeric_function_argument_is_not_suppressed_without_symbolic_shape():
    source = "The method uses 9 models."
    target = "Метод использует модели."
    result = compare_quantity_fidelity_v2(source, target)
    assert not result["ok"]
    assert 9 in result["base_missing"]
