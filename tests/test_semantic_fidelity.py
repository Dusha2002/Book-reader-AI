from bookai.semantic_fidelity import (
    compare_material_fidelity,
    compare_question_fidelity,
    compare_short_omission_fidelity,
    lexicalized_technical_compound_preserved,
)


def test_question_fidelity_rejects_collapsed_interrogative():
    result = compare_question_fidelity(
        "Why now? Was it then? Had he known? When did it stop?",
        "Почему сейчас? Тогда ли это было? Знал ли он? Когда всё прекратилось.",
    )
    assert result["ok"] is False
    assert result["source_questions"] == 4
    assert result["target_questions"] == 3
    assert result["missing_questions"] == 1


def test_question_fidelity_accepts_all_questions():
    result = compare_question_fidelity(
        "Why now? Was it then?",
        "Почему сейчас? Тогда ли это было?",
    )
    assert result["ok"] is True


def test_short_dialogue_omission_is_hard_failure():
    result = compare_short_omission_fidelity(
        "'Woman,' Bosc replied. 'Odd-looking.'",
        "— Женщина,",
    )
    assert result["ok"] is False
    assert result["applicable"] is True
    assert result["ratio"] < 0.42


def test_complete_short_dialogue_passes():
    result = compare_short_omission_fidelity(
        "'Woman,' Bosc replied. 'Odd-looking.'",
        "— Женщина, — ответил Боск. — Странная на вид.",
    )
    assert result["ok"] is True


def test_short_plain_sentence_is_not_overconstrained_by_length():
    result = compare_short_omission_fidelity(
        "He nodded and left the room.",
        "Он кивнул и вышел.",
    )
    assert result["ok"] is True
    assert result["applicable"] is False


def test_steel_plates_require_material_in_russian():
    result = compare_material_fidelity(
        "steel plates the size of beech leaves",
        "пластины размером с буковые листья",
    )
    assert result["ok"] is False
    assert result["missing"] == ["steel"]


def test_steel_plates_accept_inflected_material():
    result = compare_material_fidelity(
        "steel plates the size of beech leaves",
        "стальные пластины размером с буковые листья",
    )
    assert result["ok"] is True


def test_steel_as_verb_is_not_a_material_obligation():
    result = compare_material_fidelity(
        "He tried to steel himself for the answer.",
        "Он попытался собраться перед ответом.",
    )
    assert result["ok"] is True
    assert result["required"] == []


def test_lead_screw_is_a_lexicalized_technical_term():
    source = "Backlash in the lead-screw was a tragedy."
    target = "Люфт ходового винта был трагедией."
    assert lexicalized_technical_compound_preserved(source, target) is True
    assert compare_material_fidelity(source, target)["ok"] is True
