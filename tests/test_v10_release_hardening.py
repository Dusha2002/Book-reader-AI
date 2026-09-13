from types import SimpleNamespace

from bookai.models import BookMemory, Segment
from bookai.v10_release import (
    HardenedBookBibleBuilder,
    HardenedDialogueDiscourseGuard,
    HardenedV10QualityQA,
    russian_numbered_chapter_heading,
    select_numbered_chapter,
)


def seg(text: str, sid: str, chapter: str = "Chapter One") -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter=chapter)


def codes(issues):
    return {issue.code for issue in issues}


def test_complete_chapter_selection_keeps_internal_subheading_group():
    document = SimpleNamespace(segments=[
        seg("Chapter Two", "s000001", "Chapter Two"),
        seg("First paragraph of chapter two with enough words to translate.", "s000002", "Chapter Two"),
        seg("SEEMED LIKE A GOOD IDEA AT THE TIME", "s000003", "SEEMED LIKE A GOOD IDEA AT THE TIME"),
        seg("Second section still belongs to chapter two and must not be cut.", "s000004", "SEEMED LIKE A GOOD IDEA AT THE TIME"),
        seg("Chapter Three", "s000005", "Chapter Three"),
        seg("Opening paragraph of chapter three with enough words to translate.", "s000006", "Chapter Three"),
    ])
    _all, name, selected, meta = select_numbered_chapter(document, "Chapter Two")
    assert name == "Chapter Two"
    assert [row.id for row in selected] == ["s000001", "s000002", "s000003", "s000004"]
    assert meta["included_groups"] == ["Chapter Two", "SEEMED LIKE A GOOD IDEA AT THE TIME"]
    assert meta["next_numbered_chapter"] == "Chapter Three"
    assert meta["boundary_complete"] is True


def test_numbered_chapter_heading_is_natural_russian_ordinal():
    assert russian_numbered_chapter_heading("Chapter One") == "Глава первая"
    assert russian_numbered_chapter_heading("Chapter Two") == "Глава вторая"
    assert russian_numbered_chapter_heading("Chapter Three") == "Глава третья"


def test_dialogue_guard_removes_stranded_english_quote_after_comma():
    s = seg("'Very well then,' he said. 'Let us consider the details.'", "s000001")
    translated = {"s000001": "— Хорошо, тогда,' сказал он. — Давайте рассмотрим детали."}
    HardenedDialogueDiscourseGuard().apply([s], translated)
    assert "'" not in translated["s000001"]
    assert translated["s000001"].startswith("— Хорошо, тогда,")


def test_quantity_gate_accepts_russian_fraction_numerator_and_list_tail():
    qa = HardenedV10QualityQA()
    source = (
        "The gear train shall comprise five cogs of ratios forty, thirty, twenty-five, twelve and six to one; "
        "the cogs shall be three eighths of an inch thick."
    )
    target = (
        "Передача должна состоять из пяти шестерен с соотношениями сорок, тридцать, двадцать пять, "
        "двенадцать и шесть к одному; толщина шестерен — три восьмых дюйма."
    )
    issue_codes = codes(qa.scan_segment(seg(source, "s000001"), target, BookMemory()))
    assert "numeric" not in issue_codes
    assert "quantity_obligation" not in issue_codes


def test_quantity_gate_accepts_seven_sixteenths():
    qa = HardenedV10QualityQA()
    source = "The cogs were seven sixteenths of an inch thick."
    target = "Толщина шестерен составляла семь шестнадцатых дюйма."
    issue_codes = codes(qa.scan_segment(seg(source, "s000001"), target, BookMemory()))
    assert "numeric" not in issue_codes
    assert "quantity_obligation" not in issue_codes


def test_half_inch_gate_rejects_full_inch_and_accepts_half_inch():
    qa = HardenedV10QualityQA()
    source = "The cloud was made up of half-inch steel rods."
    bad = qa.scan_segment(seg(source, "s000001"), "Облако состояло из стальных прутьев дюймовой толщины.", BookMemory())
    good = qa.scan_segment(seg(source, "s000001"), "Облако состояло из полудюймовых стальных прутьев.", BookMemory())
    assert "half_inch" in codes(bad)
    assert "half_inch" not in codes(good)


def test_brass_bronze_contract_catches_material_reversal():
    qa = HardenedV10QualityQA()
    source = "Each cog shall ride on a brass bushing."
    bad = qa.scan_segment(seg(source, "s000001"), "Каждая шестерня должна опираться на бронзовую втулку.", BookMemory())
    good = qa.scan_segment(seg(source, "s000001"), "Каждая шестерня должна опираться на латунную втулку.", BookMemory())
    assert "material" in codes(bad)
    assert "material" not in codes(good)


def test_not_brass_but_bronze_requires_both_materials():
    qa = HardenedV10QualityQA()
    source = "Their bushings were not brass but bronze."
    bad = qa.scan_segment(seg(source, "s000001"), "Их втулки были не бронзовыми, а медными.", BookMemory())
    good = qa.scan_segment(seg(source, "s000001"), "Их втулки были не латунными, а бронзовыми.", BookMemory())
    assert "material" in codes(bad)
    assert "material" not in codes(good)


def test_name_canon_allows_inflection_but_rejects_spelling_drift():
    memory = BookMemory()
    memory.glossary["Miel"] = "Миэль"
    memory.characters["Miel"] = "ru=Миэль;gender=male;role=duke"
    qa = HardenedV10QualityQA()
    source = "Miel took charge."
    bad = qa.scan_segment(seg(source, "s000001"), "Миль взял на себя руководство.", memory)
    good = qa.scan_segment(seg(source, "s000001"), "У Миэля было достаточно власти, чтобы взять руководство на себя.", memory)
    assert "name_canon" in codes(bad)
    assert "name_canon" not in codes(good)


def test_mezentine_ethnonym_cannot_turn_into_another_nationality():
    qa = HardenedV10QualityQA()
    source = "You're a Mezentine, but you're nothing to do with the army."
    bad = qa.scan_segment(seg(source, "s000001"), "Вы венезиец, но к армии отношения не имеете.", BookMemory())
    good = qa.scan_segment(seg(source, "s000001"), "Вы мезентиец, но к армии отношения не имеете.", BookMemory())
    assert "ethnonym_canon" in codes(bad)
    assert "ethnonym_canon" not in codes(good)


def test_last_but_one_rejects_contradictory_calque():
    qa = HardenedV10QualityQA()
    source = "Fencing was last lesson but one on a Monday."
    bad = qa.scan_segment(seg(source, "s000001"), "Фехтование было последним, но предпоследним занятием в понедельник.", BookMemory())
    good = qa.scan_segment(seg(source, "s000001"), "В понедельник фехтование было предпоследним занятием.", BookMemory())
    assert "idiom_last_but_one" in codes(bad)
    assert "idiom_last_but_one" not in codes(good)


def test_at_back_of_mind_is_not_physical_order_error():
    qa = HardenedV10QualityQA()
    source = "Also at the back of his mind was another question."
    target = "Также в глубине сознания у него оставался ещё один вопрос."
    assert "order" not in codes(qa.scan_segment(seg(source, "s000001"), target, BookMemory()))


def test_source_with_up_and_down_motion_does_not_false_flag_when_both_survive():
    qa = HardenedV10QualityQA()
    source = "They took the pass up the mountain, then came down into the plain."
    target = "Они поднялись по перевалу в гору, а затем спустились на равнину."
    assert "direction_relation" not in codes(qa.scan_segment(seg(source, "s000001"), target, BookMemory()))


def test_technical_candidate_augmentation_includes_audit_terms():
    rows = HardenedBookBibleBuilder._candidate_records([
        seg("The tuck had no cutting edge, only a point.", "s000001"),
        seg("A steel cable was fastened to the spring.", "s000002"),
        seg("Each cog rode on a brass bushing.", "s000003"),
    ])
    candidates = {str(row.get("candidate") or "").casefold() for row in rows}
    assert "tuck" in candidates
    assert "steel cable" in candidates
    assert "brass bushing" in candidates
