from bookai.semantic_fidelity import compare_order_fidelity


def test_at_the_front_rejects_reversal_to_last():
    source = "Ziani had put his best stuff at the front."
    bad = "Зиани приберег лучшее напоследок."
    check = compare_order_fidelity(source, bad)
    assert check["ok"] is False
    assert "at_the_front" in check["missing"]
    assert "at_the_front" in check["contradictory"]


def test_at_the_front_accepts_front_meaning():
    source = "Ziani had put his best stuff at the front."
    good = "Зиани поставил всё лучшее впереди."
    assert compare_order_fidelity(source, good)["ok"] is True


def test_at_the_end_accepts_end_meaning():
    source = "He put the note at the end."
    good = "Он поместил записку в конце."
    assert compare_order_fidelity(source, good)["ok"] is True


def test_unrelated_front_word_does_not_create_obligation():
    source = "He stood in front of the door."
    target = "Он стоял перед дверью."
    assert compare_order_fidelity(source, target)["ok"] is True
