import pytest

from bookai.models import BookMemory, Segment
from bookai.quality import assert_exact_ids, candidate_issues


def codes(segment: Segment, text: str):
    return {x.code for x in candidate_issues(segment, text, BookMemory())}


def test_rejects_unexpected_script_and_english_fallback():
    s = Segment("s000001", "The machine struck the anvil twelve times.", "/body/section/p")
    assert "unexpected_script" in codes(s, "Машина ударила по наковальне 十二 раз.")
    assert "unchanged" in codes(s, s.text)


def test_rejects_changed_numbers():
    s = Segment("s000002", "He waited 12 days and paid 4 marks.", "/body/section/p")
    assert "numbers" in codes(s, "Он ждал двенадцать дней и заплатил 5 марок.")


def test_heading_cannot_expand_into_body():
    s = Segment("s000003", "Chapter Three", "/body/section/title/p")
    assert "heading_multiline" in codes(s, "Глава третья\nА затем началась длинная история")


def test_exact_id_contract_never_silently_falls_back():
    segments = [
        Segment("s000001", "One", "/p"),
        Segment("s000002", "Two", "/p"),
    ]
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(segments, {"s000001": "Один"}, "translator")
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(segments, {"s000001": "Один", "s000002": "Два", "extra": "x"}, "translator")
