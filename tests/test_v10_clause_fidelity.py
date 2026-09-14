from bookai.models import BookMemory, Segment
from bookai.v10_clause_fidelity import (
    compare_clause_order_fidelity,
    compare_duplicate_content_fidelity,
    scan_clause_fidelity,
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


def test_duplicate_content_allows_expanded_ru_phrase_for_repeated_en_bigram():
    source = (
        "A similar problem occurs if the hidden code is equal to the input, and in the overcomplete case "
        "in which the hidden code is greater than the input."
    )
    target = (
        "Похожая проблема возникает, если размер скрытого кода равен размеру входа, а также в случае, "
        "когда размер скрытого кода превышает размер входа."
    )
    result = compare_duplicate_content_fidelity(source, target)
    assert result["ok"]
    assert result.get("expanded_translation_repetition") is True


def test_runtime_glossary_allows_multiword_target_repeat_for_repeated_source_term():
    source = (
        "Back-propagation made deep training practical and helped popularize the back-propagation algorithm. "
        "The algorithm remains widely used."
    )
    target = (
        "Обратное распространение ошибки сделало глубокое обучение практичным и помогло популяризировать "
        "алгоритм обратного распространения ошибки. Алгоритм по-прежнему широко применяется."
    )
    memory = BookMemory(glossary={"back-propagation": "обратное распространение ошибки"})
    issues = scan_clause_fidelity(seg(source), target, memory)
    assert not any(issue.code == "duplicate_content" for issue in issues)


def test_runtime_glossary_does_not_hide_unrelated_invented_repetition():
    source = "Back-propagation made deep training practical. The algorithm remains widely used."
    target = (
        "Обратное распространение ошибки сделало глубокое обучение практичным. "
        "Алгоритм остаётся широко применяемым, глубокое обучение практичным, алгоритм остаётся широко применяемым."
    )
    memory = BookMemory(glossary={"back-propagation": "обратное распространение ошибки"})
    issues = scan_clause_fidelity(seg(source), target, memory)
    assert any(issue.code == "duplicate_content" for issue in issues)


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
