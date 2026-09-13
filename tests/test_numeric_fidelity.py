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


def test_generic_first_time_can_lexicalize():
    result = compare_numeric_fidelity(
        "He wrote to him for the first time since his escape.",
        "Он написал ему впервые после побега.",
    )
    assert result["ok"] is True
    assert result["source_values"] == []


def test_generic_third_cousin_is_not_a_hard_numeric_fact():
    result = compare_numeric_fidelity(
        "He met his third cousin.",
        "Он встретил троюродного брата.",
    )
    assert result["ok"] is True
    assert result["source_values"] == []


def test_fourth_day_is_numbered_context_and_must_survive():
    result = compare_numeric_fidelity(
        "At noon on the fourth day they arrived.",
        "В полдень пятого дня они прибыли.",
    )
    assert result["ok"] is False
    assert result["missing"] == [4]


def test_punctuation_separates_island_and_street_numbers():
    result = compare_numeric_fidelity(
        "Island Seventeen, Sixty-Seventh Street.",
        "Остров Семнадцать, Шестьдесят седьмая улица.",
    )
    assert result["ok"] is True
    assert result["source_values"] == [17, 67]
    assert result["target_values"] == [17, 67]


def test_seven_storey_does_not_accept_seventeen_storey():
    result = compare_numeric_fidelity(
        "It was a seven-storey block on the sixth floor.",
        "Это был семнадцатиэтажный дом на шестом этаже.",
    )
    assert result["ok"] is False
    assert 7 in result["missing"]
    assert 6 not in result["missing"]


def test_thirty_thousandths_preserves_numerator_without_becoming_30000():
    result = compare_numeric_fidelity(
        "The gap never varied by more than thirty thousandths of an inch.",
        "Зазор отличался не более чем на тридцать тысячных дюйма.",
    )
    assert result["ok"] is True
    assert 30 in result["source_values"]
    assert 30 in result["target_values"]
    assert 30000 not in result["target_values"]


def test_small_cardinal_quantity_is_strict_and_collective_russian_counts():
    result = compare_numeric_fidelity(
        "There were four guards behind him.",
        "Позади него шли четверо стражников.",
    )
    assert result["ok"] is True
    assert result["source_values"] == [4]
    assert 4 in result["target_values"]
