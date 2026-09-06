from bookai.models import Segment
from scripts.full_reference_translation import _sanitize_resume_state


def _segment(sid: str, chapter: str) -> Segment:
    return Segment(sid, "Source text.", "/body/section/p", chapter=chapter)


def test_resume_keeps_only_complete_qa_passed_chapters():
    chapters = [
        ("One", [_segment("s000001", "One"), _segment("s000002", "One")]),
        ("Two", [_segment("s000003", "Two")]),
        ("Three", [_segment("s000004", "Three"), _segment("s000005", "Three")]),
    ]
    state = {
        "pipeline_version": "literary-harness-v2.1",
        "translations": {
            "s000001": "Один.",
            "s000002": "Два.",
            "s000003": "Три.",
            "s000004": "Старый черновик.",
            "s000005": "Старый черновик.",
        },
        "completed_chapters": ["One", "Two"],
        "polished_chapters": ["One", "Two", "Three"],
        "qa_passed_chapters": ["One", "Two"],
        "final_quality": {"hard_issues": 0},
    }

    cleaned, report = _sanitize_resume_state(state, chapters)

    assert cleaned["translations"] == {
        "s000001": "Один.",
        "s000002": "Два.",
        "s000003": "Три.",
    }
    assert cleaned["completed_chapters"] == ["One", "Two"]
    assert cleaned["polished_chapters"] == ["One", "Two"]
    assert cleaned["qa_passed_chapters"] == ["One", "Two"]
    assert "final_quality" not in cleaned
    assert report["reset_chapters"] == ["Three"]
    assert report["removed_translations"] == 2


def test_resume_revokes_claimed_qa_chapter_when_translation_is_incomplete():
    chapters = [
        ("One", [_segment("s000001", "One"), _segment("s000002", "One")]),
    ]
    state = {
        "translations": {"s000001": "Только половина."},
        "completed_chapters": ["One"],
        "polished_chapters": ["One"],
        "qa_passed_chapters": ["One"],
    }

    cleaned, report = _sanitize_resume_state(state, chapters)

    assert cleaned["translations"] == {}
    assert cleaned["completed_chapters"] == []
    assert cleaned["polished_chapters"] == []
    assert cleaned["qa_passed_chapters"] == []
    assert report["reset_chapters"] == ["One"]
    assert report["removed_translations"] == 1
