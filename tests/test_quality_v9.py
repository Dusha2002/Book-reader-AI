from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bookai.models import BookMemory, Segment

import chapter_reference_translation_v9c as v9c

v9b = v9c.v9b
v9 = v9c.v9


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
    value, _ = v9c._format_dialogue_v9c(source, "'Нет,' — сказал он. 'Не сегодня.'")
    assert value.count("«") == value.count("»")
    assert not value.startswith(("'", '"'))
    assert ",," not in value


def test_v9c_formatter_repairs_observed_double_comma_and_stray_quote():
    source = seg("'I should do,' the man replied. 'I used to make them.'")
    value, _ = v9c._format_dialogue_v9c(source, "— Я должен', — ответил мужчина. — Раньше я их делал.")
    assert "'," not in value
    assert ",," not in value
    assert value.startswith("— ")


def test_narrative_paired_quotes_become_safe_guillemets():
    source = seg("He called it 'necessary evil'.")
    value, _ = v9c._format_dialogue_v9c(source, 'Он называл это "необходимым злом".')
    assert "«необходимым злом»" in value
    assert value.count("«") == value.count("»")


def test_referent_and_short_dialogue_are_micro_audit_risks():
    assert v9c._risk_segment(seg("'I won't see either of them again.'"))
    assert v9c._risk_segment(seg("'I should do,' the man replied."))
    assert v9c._risk_segment(seg("'Cocky with it,' Orsea said."))


def test_long_narration_without_referent_is_not_micro_audit_risk():
    source = "The road crossed the valley and climbed the ridge before disappearing into fog. " * 12
    assert not v9c._risk_segment(seg(source))
