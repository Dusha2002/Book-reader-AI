from bookai.models import Segment
from bookai.release_guards import provenance_entry
from bookai.resume import first_complete_unchecked_chapter, sanitize_resume_state


def _segment(sid: str, chapter: str, text: str = "Source text.") -> Segment:
    return Segment(sid, text, "/body/section/p", chapter=chapter)


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


def test_review_required_never_counts_as_qa_passed():
    chapters = [("One", [_segment("s000001", "One"), _segment("s000002", "One")])]
    state = {
        "translations": {"s000001": "Один.", "s000002": "Два."},
        "completed_chapters": ["One"],
        "polished_chapters": ["One"],
        "qa_passed_chapters": ["One"],
        "review_required_chapters": {"One": {"phase": "qa", "reason": "unresolved semantic check"}},
        "final_quality": {"hard_issues": 0},
    }

    cleaned, report = sanitize_resume_state(state, chapters)

    assert cleaned["qa_passed_chapters"] == []
    assert list(cleaned["review_required_chapters"]) == ["One"]
    assert "final_quality" not in cleaned
    assert report["review_required_chapters"] == ["One"]


def test_provenance_invalidates_translation_when_source_changes_under_same_id():
    original = _segment("s000001", "One", "Original source.")
    chapters = [("One", [_segment("s000001", "One", "Changed source.")])]
    state = {
        "translations": {"s000001": "Старый перевод."},
        "translation_provenance": {"s000001": provenance_entry(original)},
        "completed_chapters": ["One"],
        "qa_passed_chapters": ["One"],
    }

    cleaned, report = sanitize_resume_state(state, chapters)

    assert cleaned["translations"] == {}
    assert cleaned["qa_passed_chapters"] == []
    assert report["stale_source_hashes"] == 1


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

    state["review_required_chapters"] = {"One": {"phase": "qa"}}
    assert first_complete_unchecked_chapter(state, chapters) is None

    state["translations"]["s000004"] = "Четыре."
    assert first_complete_unchecked_chapter(state, chapters) == "Two"

    state["qa_passed_chapters"] = ["Two"]
    assert first_complete_unchecked_chapter(state, chapters) is None
