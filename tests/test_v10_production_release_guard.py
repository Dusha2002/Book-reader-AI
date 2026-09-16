from bookai.models import BookMemory, Segment
from bookai.v10 import V10Issue
from bookai.v10_production_release_guard import (
    canonicalize_entity_family_spelling,
    clean_numeric_result,
    clean_quantity_result,
    entity_family_equivalent_present,
    filter_release_false_positives,
    localized_gender_issues,
    spurious_between_range_sums,
    strict_direction_mismatch,
    target_has_substantial_repeat,
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


def test_direction_guard_does_not_treat_gaze_as_locomotion():
    source = (
        "Over the crest of the hill, looking down, he could see the cavalry below him. "
        "There was no call for them to look up the hill."
    )
    target = (
        "За гребнем холма, глядя вниз, он видел кавалерию под собой. "
        "Им незачем было смотреть вверх по склону."
    )
    assert not strict_direction_mismatch(source, target)


def test_direction_guard_keeps_real_motion_reversal_hard():
    assert strict_direction_mismatch(
        "He walked up the hill towards the gate.",
        "Он спустился с холма к воротам.",
    )
    assert strict_direction_mismatch(
        "She went down the steps and crossed the yard.",
        "Она поднялась по ступеням и пересекла двор.",
    )


def test_duplicate_guard_ignores_normal_long_paragraph_recurrence():
    target = (
        "Он затянул подпругу, проверил удила и вывел лошадь во двор. "
        "Лошадь нервничала, но вскоре успокоилась, и он осторожно сел в седло. "
        "Потом он оглянулся на двор и тронул поводья."
    )
    assert not target_has_substantial_repeat(target)


def test_duplicate_guard_confirms_repeated_substantial_clause():
    target = (
        "Он поднял меч и медленно шагнул к двери. Потом остановился и посмотрел назад. "
        "Он поднял меч и медленно шагнул к двери."
    )
    assert target_has_substantial_repeat(target)


def test_entity_family_accepts_only_narrow_initial_e_variant():
    memory = BookMemory()
    desc = "gender=unknown;kind=demonym_family;ru_root=еремиан;forms=еремианец/еремиане/еремианский"
    memory.characters["Eremian"] = desc
    memory.characters["Eremians"] = desc
    source = seg("He bought a good-quality Eremian backsword.")
    assert entity_family_equivalent_present(
        source,
        "Он купил хороший эремианский палаш.",
        memory,
    )
    assert not entity_family_equivalent_present(
        source,
        "Он купил хороший мезентинский палаш.",
        memory,
    )


def test_entity_family_checkpoint_canonicalizes_initial_e_variant():
    families = [{"ru_root": "еремиан"}]
    target = "Он выбрал эремианского мастера и Эремианский клинок."
    assert canonicalize_entity_family_spelling(target, families) == (
        "Он выбрал еремианского мастера и Еремианский клинок."
    )


def test_release_filter_suppresses_only_unconfirmed_false_positives():
    memory = BookMemory()
    memory.characters["Eremian"] = (
        "gender=unknown;kind=demonym_family;ru_root=еремиан;"
        "forms=еремианец/еремиане/еремианский"
    )
    source = seg(
        "Over the crest of the hill, looking down, he saw an Eremian soldier. "
        "There was no reason to look up the hill."
    )
    target = "Глядя вниз с гребня, он увидел эремианского солдата; наверх никто не смотрел."
    issues = [
        V10Issue(source.id, "direction_relation", "semantic", "hard", "coarse direction"),
        V10Issue(source.id, "duplicate_content", "semantic", "hard", "coarse duplicate"),
        V10Issue(source.id, "entity_family_canon", "semantic", "hard", "root variant"),
        V10Issue(source.id, "omission", "semantic", "hard", "real independent issue"),
    ]
    filtered = filter_release_false_positives(source, target, memory, issues)
    assert [issue.code for issue in filtered] == ["omission"]


def test_release_filter_keeps_confirmed_direction_reversal():
    source = seg("He walked up the hill towards the gate.")
    target = "Он спустился с холма к воротам."
    issue = V10Issue(source.id, "direction_relation", "semantic", "hard", "reversal")
    assert filter_release_false_positives(source, target, BookMemory(), [issue]) == [issue]
