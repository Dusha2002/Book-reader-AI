from bookai.numeric_fidelity import compare_numeric_fidelity


def test_number_thirty_rejects_thirty_first():
    result = compare_numeric_fidelity(
        "He lived at number thirty on the next street.",
        "Он жил в доме тридцать первом на соседней улице.",
    )
    assert result["ok"] is False
    assert 30 in result["missing"]
    assert 31 in result["target_values"]


def test_number_thirty_accepts_exact_cardinal():
    result = compare_numeric_fidelity(
        "He lived at number thirty on the next street.",
        "Он жил в доме номер тридцать на соседней улице.",
    )
    assert result["ok"] is True


def test_compound_twenty_eight_accepts_inflected_ordinal():
    result = compare_numeric_fidelity(
        "They stopped at number twenty-eight.",
        "Они остановились у двадцать восьмого дома.",
    )
    assert result["ok"] is True
    assert 28 in result["target_values"]


def test_two_numbered_entities_preserve_both_values():
    result = compare_numeric_fidelity(
        "Number twenty-eight was opposite number thirty.",
        "Дом двадцать восемь стоял напротив дома тридцать один.",
    )
    assert result["ok"] is False
    assert result["missing"] == [30]


def test_digits_can_satisfy_word_number_obligation():
    result = compare_numeric_fidelity(
        "The address was number thirty.",
        "Адрес: дом № 30.",
    )
    assert result["ok"] is True


def test_ambiguous_single_one_is_not_forced_without_numeric_context():
    result = compare_numeric_fidelity(
        "One could hardly blame him.",
        "Его трудно было винить.",
    )
    assert result["ok"] is True
    assert result["source_values"] == []


def test_hundred_compound_is_preserved():
    result = compare_numeric_fidelity(
        "There were one hundred and five pages.",
        "Там было сто пять страниц.",
    )
    assert result["ok"] is True
    assert 105 in result["source_values"]
    assert 105 in result["target_values"]
