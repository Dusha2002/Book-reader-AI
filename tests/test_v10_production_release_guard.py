from bookai.models import BookMemory, Segment
from bookai.v10_production_release_guard import (
    clean_numeric_result,
    clean_quantity_result,
    localized_gender_issues,
    spurious_between_range_sums,
)
from bookai.numeric_fidelity import compare_numeric_fidelity
from bookai.v10_quantity import compare_quantity_fidelity_v2


def seg(text: str, sid: str = "s000001") -> Segment:
    return Segment(id=sid, text=text, locator="/p[1]", chapter="3")


def test_between_range_does_not_invent_sum_75():
    source = "He could have been anywhere between twenty-five and fifty."
    target = "Ему могло быть сколько угодно — от двадцати пяти до пятидесяти."
    assert spurious_between_range_sums(source) == {75}

    raw_numeric = compare_numeric_fidelity(source, target)
    assert 75 in (raw_numeric.get("missing") or [])
    numeric = clean_numeric_result(source, raw_numeric)
    assert numeric["ok"], numeric
    assert 75 not in numeric["missing"]

    raw_quantity = compare_quantity_fidelity_v2(source, target)
    quantity = clean_quantity_result(source, raw_quantity)
    assert quantity["ok"], quantity
    assert 75 not in quantity["base_missing"]


def test_gender_guard_ignores_other_speaker_feminine_verb():
    memory = BookMemory()
    memory.characters["Vaatzes"] = "ru=Ваатцес;gender=male;kind=person"
    source = seg(
        '"A saddle-cloth," she answered. A little later Vaatzes thought that the difference mattered.'
    )
    target = (
        "— Суконку, — ответила она. Немного погодя Ваатцес решил, "
        "что разница всё-таки важна, подумал Ваатцес."
    )
    assert localized_gender_issues(source, target, memory) == []


def test_gender_guard_keeps_real_local_mismatch_hard():
    memory = BookMemory()
    memory.characters["Vaatzes"] = "ru=Ваатцес;gender=male;kind=person"
    source = seg('"Enough," Vaatzes said.')
    target = "— Хватит, — сказала Ваатцес."
    issues = localized_gender_issues(source, target, memory)
    assert [issue.code for issue in issues] == ["character_gender"]
    assert issues[0].severity == "hard"
