from __future__ import annotations

from pathlib import Path

from bookai.models import BookMemory, Segment
from bookai.release_final import FinalV10QualityQA


ROOT = Path(__file__).resolve().parents[1]


def _segment(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="test", chapter="Chapter One")


def test_release_facade_delegates_to_general_layer():
    text = (ROOT / "src/bookai/release_final.py").read_text("utf-8")
    assert "v10_general_release" in text
    assert "class FinalV10QualityQA" not in text


def test_general_release_has_no_fixture_book_vocabulary():
    paths = [
        ROOT / "src/bookai/v10_general_release.py",
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
        characters={"Xarion": "ru=Ксарион;gender=male;role=protagonist;voice=reserved"},
    )
    issues = qa.scan_segment(_segment("Xarion answered."), "Зарион ответил.", memory)
    assert any(issue.code == "name_canon" and issue.severity == "hard" for issue in issues)


def test_unknown_book_name_is_not_forced_without_runtime_memory():
    qa = FinalV10QualityQA()
    issues = qa.scan_segment(_segment("Xarion answered."), "Зарион ответил.", BookMemory())
    assert not any(issue.code == "name_canon" for issue in issues)
