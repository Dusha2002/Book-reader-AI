from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bookai.models import BookMemory, Segment

import chapter_reference_translation_v9g as v9g

v9f = v9g.v9f
v9e = v9f.v9e
v9d = v9e.v9d
v9c = v9e.v9c
v9b = v9c.v9b
v9 = v9g.v9


def seg(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="x", chapter="Chapter Two")


def test_material_contrast_flags_brass_bronze_substitution():
    source = "Each gear shall ride on a brass bushing; the inspector found the bushings were not brass but bronze."
    target = "Каждая шестерня должна вращаться на бронзовой втулке; инспектор установил, что втулки были не латунными, а бронзовыми."
    rows = v9b._deterministic_findings(seg(source), target, BookMemory())
    assert any(row["severity"] == "critical" and row["code"].startswith("material") for row in rows)


def test_material_contrast_accepts_preserved_brass_and_bronze():
    source = "Each gear shall ride on a brass bushing; the inspector found the bushings were not brass but bronze."
    target = "Каждая шестерня должна вращаться на латунной втулке; инспектор установил, что втулки были не латунными, а бронзовыми."
    rows = v9b._deterministic_findings(seg(source), target, BookMemory())
    assert not any(row["code"].startswith("material") for row in rows)


def test_time_and_a_half_rejects_double_pay():
    source = "He was entitled to time and a half."
    target = "Ему полагалась двойная оплата."
    rows = v9b._deterministic_findings(seg(source), target, BookMemory())
    assert any(row["code"] == "number" and row["severity"] == "critical" for row in rows)


def test_time_and_a_half_accepts_one_and_a_half_pay():
    source = "He was entitled to time and a half."
    target = "Ему полагалась полуторная оплата."
    rows = v9b._deterministic_findings(seg(source), target, BookMemory())
    assert not any(row["code"] == "number" and row["severity"] == "critical" for row in rows)


def test_dialogue_formatter_does_not_create_unbalanced_guillemets():
    source = seg("'No,' he said. 'Not today.'")
    value, _ = v9d._format_dialogue_v9d(source, "'Нет,' — сказал он. 'Не сегодня.'")
    assert value.count("«") == value.count("»")
    assert not value.startswith(("'", '"'))
    assert ",," not in value


def test_v9d_formatter_repairs_observed_double_comma_and_stray_quote():
    source = seg("'I should do,' the man replied. 'I used to make them.'")
    value, _ = v9d._format_dialogue_v9d(source, "— Я должен', — ответил мужчина. — Раньше я их делал.")
    assert "'," not in value
    assert ",," not in value
    assert value.startswith("— ")


def test_narrative_paired_quotes_become_safe_guillemets():
    source = seg("He called it 'necessary evil'.")
    value, _ = v9d._format_dialogue_v9d(source, 'Он называл это "необходимым злом".')
    assert "«необходимым злом»" in value
    assert value.count("«") == value.count("»")


def test_embedded_dialogue_after_narration_is_detected_and_normalized():
    source = seg("The man looked at him. 'You mean, what sort of weapon was it?'")
    value, _ = v9d._format_dialogue_v9d(source, 'Человек посмотрел на него. — «Ты имеешь в виду, что это было за оружие?»')
    assert v9d._source_has_dialogue(source.text)
    assert '— «' not in value
    assert value.endswith('?')


def test_possessive_apostrophes_do_not_create_false_dialogue_risk():
    source = "It was typical of Valens' father that he insisted on his son's lessons; the man's patience was endless and didn't help."
    assert not v9e._short_utterance_risk(source)
    assert not v9e._risk_segment_v9e(seg(source))
    assert not v9f._risk_segment_v9f(seg(source))


def test_v9f_keeps_known_discourse_failures_in_scope():
    assert v9f._risk_segment_v9f(seg("'I won't see either of them again, I don't suppose.'"))
    assert v9f._risk_segment_v9f(seg("'I should do,' the man replied. 'I used to make them.'"))
    assert v9f._risk_segment_v9f(seg("'Cocky with it,' Orsea said. 'So, you're an escaped convict.'"))


def test_v9f_does_not_audit_every_tiny_quote():
    assert not v9f._risk_segment_v9f(seg("'Really?' he asked."))
    assert not v9f._risk_segment_v9f(seg("'Hello,' she said."))
    assert not v9f._risk_segment_v9f(seg("'Fine,' he said."))


def test_v9g_forces_only_explicit_referents_without_evidence():
    targets = [
        seg("'I won't see either of them again.'"),
        Segment(id="s2", text="'Cocky with it,' he said.", locator="x", chapter="Chapter Two"),
        Segment(id="s3", text="'Really?' he asked.", locator="x", chapter="Chapter Two"),
    ]
    translations = {"s1": "— Больше их не увижу.", "s2": "— И дерзит, — сказал он.", "s3": "— Правда? — спросил он."}
    rows = v9g._synthetic_confirmed_rows(targets, translations)
    assert {(row["id"], row["code"]) for row in rows} == {("s1", "referent")}


def test_v9g_routes_observed_literal_should_do():
    source = seg("'I should do,' the man replied. 'I used to make them.'")
    assert v9g._literalization_kind(source, "— Я должен, — ответил мужчина. — Раньше я их делал.") == "elliptical_modal"
    assert not v9g._literalization_kind(source, "— Ещё как знаю, — ответил мужчина. — Я их раньше делал.")


def test_v9g_routes_observed_literal_with_it_but_not_good_translation():
    source = seg("'Cocky with it,' Orsea said. 'So, you're an escaped convict.'")
    assert v9g._literalization_kind(source, "— Самоуверенный с ним разговор, — сказал Орсеа.") == "preposition_idiom"
    assert not v9g._literalization_kind(source, "— И ещё дерзит, — сказал Орсеа.")


def test_normal_long_dialogue_is_not_sparse_micro_audit_risk():
    source = "'I walked across the entire valley yesterday because the western bridge had been destroyed by the flood and nobody had repaired it yet,' he said."
    assert not v9e._risk_segment_v9e(seg(source))
    assert not v9f._risk_segment_v9f(seg(source))


def test_long_narration_without_referent_is_not_micro_audit_risk():
    source = "The road crossed the valley and climbed the ridge before disappearing into fog. " * 12
    assert not v9e._risk_segment_v9e(seg(source))
    assert not v9f._risk_segment_v9f(seg(source))


def test_v9f_formatter_removes_stranded_dialogue_apostrophes():
    source = seg("'Slimy sauce,' he repeated. 'Yetch.'")
    value, _ = v9f._format_dialogue_v9f(source, "— Слизистый соус,' повторил он. — Фу.")
    assert "'" not in value
    assert "соус, — повторил" in value


def test_v9f_formatter_repairs_mid_sentence_quote_boundary():
    source = seg("'Fine,' the man went on, 'you two get out.'")
    value, _ = v9f._format_dialogue_v9f(source, "— Ладно,' продолжил мужчина, 'вы двое убирайтесь.")
    assert "'" not in value
    assert ", — продолжил" in value
    assert ", — вы двое" in value
