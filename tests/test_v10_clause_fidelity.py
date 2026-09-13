from bookai.models import BookMemory, Segment
from bookai.v10_clause_fidelity import (
    compare_clause_order_fidelity,
    compare_duplicate_content_fidelity,
)
from bookai.v10_quality import V10QualityQA


def seg(text: str, sid: str = "s000165") -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter="Chapter One")


def test_duplicate_content_catches_clean6e_constitution_repetition():
    source = (
        "I will, yes. But if you're thinking that's all right, I'll just marry number six, "
        "I've got to tell you that'd be a grave miscalculation. You see, under their constitution-"
    )
    target = (
        "Я выясню, да. Но если вы думаете, что этого достаточно, должен вас предупредить: "
        "вы совершите роковую ошибку. Вы ведь знаете, под их конституцией — я просто женюсь "
        "на шестой, мне нужно вам сказать, что это была бы серьезная ошибка. Вы видите, под их конституцией —"
    )
    result = compare_duplicate_content_fidelity(source, target)
    assert not result["ok"]
    assert "конституц" in result["repeated_phrase"]


def test_duplicate_content_allows_nonrepeated_clean_translation():
    source = "He looked at her, then looked away."
    target = "Он посмотрел на неё, а затем отвёл взгляд."
    assert compare_duplicate_content_fidelity(source, target)["ok"]


def test_clause_order_ignores_natural_name_reordering_inside_one_clause():
    memory = BookMemory(glossary={"Miel": "Миэль", "Ziani": "Зиани"})
    source = "Miel handed the file to Ziani."
    target = "Зиани получил напильник от Миэля."
    assert compare_clause_order_fidelity(source, target, memory)["ok"]


def test_clause_order_catches_reversal_across_distinct_discourse_clauses():
    memory = BookMemory(glossary={"Miel": "Миэль", "Ziani": "Зиани"})
    source = "Miel spoke first. Much later, Ziani answered."
    good = "Сначала заговорил Миэль. Намного позже ответил Зиани."
    bad = "Сначала ответил Зиани. Намного позже заговорил Миэль."
    assert compare_clause_order_fidelity(source, good, memory)["ok"]
    assert not compare_clause_order_fidelity(source, bad, memory)["ok"]


def test_quality_routes_duplicate_content_as_proven_semantic_hard():
    source = (
        "I will, yes. But if you're thinking that's all right, I'll just marry number six, "
        "I've got to tell you that'd be a grave miscalculation. You see, under their constitution-"
    )
    target = (
        "Я выясню, да. Но если вы думаете, что этого достаточно, должен вас предупредить: "
        "вы совершите роковую ошибку. Под их конституцией я женюсь на шестой, а потом снова: "
        "под их конституцией всё будет иначе."
    )
    issues = V10QualityQA().scan_segment(seg(source), target, BookMemory())
    assert any(i.code == "duplicate_content" and i.mode == "semantic" and i.severity == "hard" for i in issues)
