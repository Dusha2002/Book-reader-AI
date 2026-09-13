from bookai.models import BookMemory, Segment
from bookai.v10_dialogue import DialogueDiscourseGuard, source_has_dialogue
from bookai.v10_name_canon import spelling_preserving_v9ad
from bookai.v10_quality import V10QualityQA
from bookai.v10_source_bible import SourceOnlyBookBibleBuilder, _has_clean_russian


def seg(text: str, sid: str = "s000001") -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter="Chapter One")


def test_source_only_bible_rejects_prepositional_file_noise():
    rows = [
        seg("He put it in the file and then moved to the file again.", "s000001"),
        seg("The treadle saw shuddered as the blade bit into the stock.", "s000002"),
        seg("Later he returned to the treadle saw and adjusted it.", "s000003"),
        seg("The armourer checked the cuisses and gorget.", "s000004"),
        seg("Miel spoke to Ziani.", "s000005"),
        seg("Ziani answered Miel.", "s000006"),
    ]
    candidates = SourceOnlyBookBibleBuilder._candidate_records(rows)
    values = {str(row["candidate"]).casefold() for row in candidates}
    assert "in the file" not in values
    assert "to the file" not in values
    assert "the file" not in values
    assert "treadle saw" in values
    assert "cuisses" in values
    assert "miel" in values
    assert "ziani" in values


def test_source_only_bible_rejects_mixed_latin_russian_output():
    assert _has_clean_russian("Миэль")
    assert not _has_clean_russian("Ветраниio")
    assert not _has_clean_russian("Miel")


def test_v9ad_spelling_guard_rejects_collapsed_fictional_name():
    assert spelling_preserving_v9ad("Miel", "Миэль")
    assert not spelling_preserving_v9ad("Miel", "Мель")
    assert spelling_preserving_v9ad("Valens", "Валенс")
    assert not spelling_preserving_v9ad("Valens", "Вальс")
    assert spelling_preserving_v9ad("Orsea", "Орсеа")


def test_v9d_principle_detects_dialogue_after_author_sentence():
    source = 'He looked up. "Come here," she said.'
    assert source_has_dialogue(source)
    guard = DialogueDiscourseGuard()
    translated = {"s000001": 'Он поднял глаза. "Иди сюда", сказала она.'}
    guard.apply([seg(source)], translated)
    assert '— Иди сюда' in translated["s000001"]


def test_v9d_principle_removes_closing_guillemet_after_dash_conversion():
    source = "It was as though he spoke another language. 'I don't understand,' Valens said."
    guard = DialogueDiscourseGuard()
    translated = {"s000001": "Казалось, он говорит на другом языке. «Я не понимаю», — сказал Валенс."}
    guard.apply([seg(source)], translated)
    assert "— Я не понимаю, — сказал Валенс." in translated["s000001"]
    assert "понимаю»" not in translated["s000001"]


def test_v9d_principle_keeps_narrative_quotes_as_quotes_not_dialogue():
    source = "He called it 'the little machine' and smiled."
    assert not source_has_dialogue(source)
    guard = DialogueDiscourseGuard()
    translated = {"s000001": "Он называл это 'маленькой машиной' и улыбался."}
    guard.apply([seg(source)], translated)
    assert "«маленькой машиной»" in translated["s000001"]
    assert not translated["s000001"].startswith("—")


def test_v9ad_risk_qa_catches_up_down_reversal():
    qa = V10QualityQA()
    issues = qa.scan_segment(
        seg("He trudged up the stairs to bed."),
        "Он устало спускался по лестнице к спальне.",
        BookMemory(),
    )
    assert any(issue.code == "direction_relation" and issue.mode == "semantic" for issue in issues)
