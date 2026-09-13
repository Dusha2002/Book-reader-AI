from bookai.models import BookMemory, Segment
from bookai.v10_integrity import SegmentIntegrityGate
from bookai.v10_local_repair import GigaLocalRewriter
from bookai.v10_quality import V10QualityQA
from bookai.v10_quantity import compare_quantity_fidelity_v2


def seg(text: str, sid: str = "s000001") -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter="Chapter One")


def test_integrity_rejects_neighbor_sized_expansion():
    source = "She'd interrupted him and stolen his joke, but he didn't mind. She'd shared his thought. That didn't happen very often."
    target = (
        "Она засмеялась. Она его опередила. И они думают, что если ты съедаешь всё с тарелки, это критика. "
        "Конечно, продолжал он, никто ведь не удосужился мне сказать, я же был просто ребёнком, так что я мрачно жевал свой ужин."
    )
    issues = SegmentIntegrityGate.scan_segment(seg(source), target)
    assert any(i.code == "implausible_expansion" for i in issues)


def test_integrity_rejects_neighbor_sized_compression():
    source = (
        "When he woke up, his father's eyes were open; not looking at him, but out through the tent doorway, "
        "at the sunlight. Valens sat up, stifled a yawn; Father's eyes moved and met his, and then he looked away."
    )
    target = "Полагаю, мне следовало бы что-то сказать, подумал он; но ничего не мог придумать."
    issues = SegmentIntegrityGate.scan_segment(seg(source), target)
    assert any(i.code == "implausible_compression" for i in issues)


def test_integrity_rejects_prompt_labels_even_when_ratio_is_plausible():
    source = "The Chancellor looked at him through a curtain of rain. 'Me?'"
    target = "Канцлер посмотрел на него сквозь стену дождя. — Я? ПЕРСОНАЖИ: нет. Глоссарий: нет."
    issues = SegmentIntegrityGate.scan_segment(seg(source), target)
    assert any(i.code == "protocol_residue" for i in issues)


def test_quantity_v2_counts_twelve_and_a_dozen_as_two_obligations():
    source = "The hammer rang twelve times. You get a dozen hits at the hot metal before it cools."
    bad = "Молот ударил двенадцать раз. Ты успеваешь сделать один-два удара по горячему металлу."
    good = "Молот ударил двенадцать раз. Ты успеваешь сделать ещё двенадцать ударов по горячему металлу."
    assert not compare_quantity_fidelity_v2(source, bad)["ok"]
    assert compare_quantity_fidelity_v2(source, good)["ok"]


def test_quantity_v2_two_dozen_means_twenty_four_not_twenty():
    source = "He made two dozen lunges."
    bad = "Он сделал два десятка выпадов."
    good = "Он сделал двадцать четыре выпада."
    assert not compare_quantity_fidelity_v2(source, bad)["ok"]
    assert compare_quantity_fidelity_v2(source, good)["ok"]


def test_quantity_v2_half_dozen_means_six():
    source = "He tried half a dozen times."
    bad = "Он попробовал дюжину раз."
    good = "Он попробовал полдюжины раз."
    assert not compare_quantity_fidelity_v2(source, bad)["ok"]
    assert compare_quantity_fidelity_v2(source, good)["ok"]


def test_quantity_v2_accepts_instrumental_poluduzhinoi():
    source = "He weakened it with half a dozen hits."
    target = "Он ослабил его полудюжиной ударов."
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_quantity_v2_preserves_numbered_choice_and_governing_action():
    source = "If that's all right, I'll just marry number six."
    missing = "Если этого достаточно, я просто женюсь."
    malformed = "Если этого достаточно, это ошибка. «Шестой» (вариант)."
    good = "Если всё так просто, я женюсь на шестой."
    assert compare_quantity_fidelity_v2(source, missing)["numbered_choice_missing"]
    assert compare_quantity_fidelity_v2(source, malformed)["numbered_choice_missing"]
    assert not compare_quantity_fidelity_v2(source, good)["numbered_choice_missing"]


def test_giga_local_repair_rejects_editorial_variant_residue():
    qa = V10QualityQA()
    source = "If that's all right, I'll just marry number six."
    segment = seg(source)
    current = "Если всё так просто, я просто женюсь."
    malformed = "Если всё так просто, я совершу ошибку. «Шестой» (вариант)."
    rewriter = GigaLocalRewriter(None, qa)
    assert not rewriter._accept(segment, current, malformed, BookMemory())
    assert rewriter.stats["editorial_residue_rejected"] == 1


def test_quantity_v2_accepts_cardinal_numbered_label():
    source = "He faced it down the shaft of a number four spear."
    target = "Он встретил зверя лицом к лицу с копьём номер четыре."
    assert not compare_quantity_fidelity_v2(source, target)["numbered_choice_missing"]


def test_quantity_v2_accepts_twelve_thousand_compound():
    source = "It was a twelve-thousand-line didactic poem."
    target = "Это была двенадцатитысячная дидактическая поэма."
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_quantity_v2_accepts_inflected_six_hundred():
    source = "Licinius had six hundred Guards."
    target = "У Лициния было против шести сотен гвардейцев."
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_quantity_v2_accepts_two_century_compound():
    source = "The war between the two dukedoms, two centuries old, continued."
    target = "Двухвековая война между герцогствами продолжалась."
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_quantity_v2_does_not_lock_approximate_word_or_two():
    source = "He strained to catch a word or two."
    target = "Он старался уловить хоть слово."
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_quantity_v2_does_not_lock_week_or_ten_days():
    source = "Would it disrupt the talks for a week or ten days?"
    target = "Сорвёт ли это переговоры на неделю-другую?"
    assert compare_quantity_fidelity_v2(source, target)["ok"]


def test_v10_qa_emits_numbered_choice_not_only_generic_numeric():
    issues = V10QualityQA().scan_segment(
        seg("If that's all right, I'll just marry number six."),
        "Если этого достаточно, я просто женюсь.",
        BookMemory(),
    )
    codes = {i.code for i in issues}
    assert "numbered_choice" in codes
