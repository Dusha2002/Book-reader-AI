from bookai.address_fidelity import normalize_numbered_address_literals
from bookai.numeric_fidelity import compare_numeric_fidelity


def test_full_numbered_street_literal_is_translated_without_llm():
    source = "He did not know where Sixty-Seventh Street was."
    target = "Он не знал, где находится Sixty-Seventh Street."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count >= 1
    assert "Sixty" not in fixed
    assert "Шестьдесят седьмая улица" in fixed
    assert compare_numeric_fidelity(source, fixed)["ok"] is True


def test_between_two_numbered_streets_uses_instrumental_pair():
    source = "Sixty-Seventh Street was between Sixty-Sixth Street and Sixty-Eighth Street."
    target = "Sixty-Seventh Street находится между Sixty-Sixth и Sixty-Eighth Street."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count >= 2
    assert "между Шестьдесят шестой и Шестьдесят восьмой улицами" in fixed
    assert "Sixty" not in fixed
    assert compare_numeric_fidelity(source, fixed)["ok"] is True


def test_mixed_russian_street_prefix_is_cleaned():
    source = "Island Seventeen, Sixty-Seventh Street, was built of yellow brick."
    target = "Остров Семнадцать, улица Sixty-Seventh, был построен из желтого кирпича."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count >= 1
    assert "Sixty" not in fixed
    assert "Шестьдесят седьмая улица" in fixed
    assert compare_numeric_fidelity(source, fixed)["ok"] is True


def test_dropped_island_type_is_restored_from_exact_paired_address_label():
    source = "Island Seventeen, Sixty-Seventh Street, was built of yellow mud brick."
    target = "Семнадцать, Шестьдесят седьмая улица, был построен из желтой сырцовой глины."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count == 1
    assert "Остров Семнадцать, Шестьдесят седьмая улица" in fixed
    assert compare_numeric_fidelity(source, fixed)["ok"] is True


def test_existing_island_type_is_not_duplicated():
    source = "Island Seventeen, Sixty-Seventh Street, was built of yellow brick."
    target = "Остров Семнадцать, Шестьдесят седьмая улица, был построен из желтого кирпича."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count == 0
    assert fixed == target


def test_inflected_russian_street_prefix_does_not_duplicate_street_noun():
    source = "He did not know where Sixty-Seventh Street was."
    target = "Он не знал названия улицы Sixty-Seventh Street."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count >= 1
    assert "улицы «Шестьдесят седьмая улица»" in fixed
    assert "улицы Шестьдесят седьмая улица" not in fixed


def test_non_address_english_number_words_are_untouched():
    source = "He counted sixty-seven coins."
    target = "Он насчитал sixty-seven монет."
    fixed, count = normalize_numbered_address_literals(source, target)
    assert count == 0
    assert fixed == target
