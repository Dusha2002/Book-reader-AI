from bookai.models import Segment
from bookai.resume import first_complete_unchecked_chapter, sanitize_resume_state


def _segment(sid: str, chapter: str) -> Segment:
    return Segment(sid, "Source text.", "/body/section/p", chapter=chapter)


def test_resume_preserves_partial_chapter_progress():
    chapters = [
        ("One", [_segment("s000001", "One"), _segment("s000002", "One")]),
        ("Two", [_segment("s000003", "Two"), _segment("s000004", "Two")]),
    ]
    state = {
        "translations": {
            "s000001": "Один.",
            "s000002": "Два.",
            "s000003": "Три.",
            "stale": "лишнее",
            "s000004": "",
        },
        "completed_chapters": ["One", "Two"],
        "polished_chapters": ["One", "Two"],
        "qa_passed_chapters": ["One", "Two"],
        "final_quality": {"hard_issues": 0},
    }

    cleaned, report = sanitize_resume_state(state, chapters)

    assert cleaned["translations"] == {
        "s000001": "Один.",
        "s000002": "Два.",
        "s000003": "Три.",
    }
    assert cleaned["completed_chapters"] == ["One"]
    assert cleaned["polished_chapters"] == ["One"]
    assert cleaned["qa_passed_chapters"] == ["One"]
    assert "final_quality" not in cleaned
    assert report["partial_chapters"] == ["Two"]
    assert report["preserved_translations"] == 3
    assert report["removed_invalid_or_stale"] == 2


def test_resume_keeps_unfinished_translations_even_without_qa_claims():
    chapters = [
        ("One", [_segment("s000001", "One"), _segment("s000002", "One")]),
    ]
    state = {
        "translations": {"s000001": "Готовая половина."},
        "completed_chapters": [],
        "polished_chapters": [],
        "qa_passed_chapters": [],
    }

    cleaned, report = sanitize_resume_state(state, chapters)

    assert cleaned["translations"] == {"s000001": "Готовая половина."}
    assert cleaned["completed_chapters"] == []
    assert cleaned["polished_chapters"] == []
    assert cleaned["qa_passed_chapters"] == []
    assert report["partial_chapters"] == ["One"]
    assert report["preserved_translations"] == 1


def test_best_effort_only_infers_fully_translated_unchecked_chapter():
    chapters = [
        ("One", [_segment("s000001", "One"), _segment("s000002", "One")]),
        ("Two", [_segment("s000003", "Two"), _segment("s000004", "Two")]),
    ]
    state = {
        "translations": {
            "s000001": "Один.",
            "s000002": "Два.",
            "s000003": "Три.",
        },
        "qa_passed_chapters": [],
    }

    assert first_complete_unchecked_chapter(state, chapters) == "One"

    state["qa_passed_chapters"] = ["One"]
    assert first_complete_unchecked_chapter(state, chapters) is None

    state["translations"]["s000004"] = "Четыре."
    assert first_complete_unchecked_chapter(state, chapters) == "Two"
