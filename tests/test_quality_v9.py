from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bookai.models import BookMemory, Segment

import chapter_reference_translation_v9 as v9


def seg(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="x", chapter="Chapter Two")


def codes(rows):
    return {(row["severity"], row["code"]) for row in rows}


def test_material_contrast_flags_brass_bronze_substitution():
    source = "Each gear shall ride on a brass bushing; the inspector found the bushings were not brass but bronze."
    target = "Каждая шестерня должна вращаться на бронзовой втулке; инспектор установил, что втулки были не латунными, а бронзовыми."
    rows = v9._deterministic_findings(seg(source), target, BookMemory())
    assert ("critical", "material") in codes(rows)


def test_material_contrast_accepts_preserved_brass_and_bronze():
    source = "Each gear shall ride on a brass bushing; the inspector found the bushings were not brass but bronze."
    target = "Каждая шестерня должна вращаться на латунной втулке; инспектор установил, что втулки были не латунными, а бронзовыми."
    rows = v9._deterministic_findings(seg(source), target, BookMemory())
    assert not any(row["code"] == "material" for row in rows)


def test_time_and_a_half_rejects_double_pay():
    source = "He was entitled to time and a half."
    target = "Ему полагалась двойная оплата."
    rows = v9._deterministic_findings(seg(source), target, BookMemory())
    assert any(row["code"] == "number" and row["severity"] == "critical" for row in rows)


def test_time_and_a_half_accepts_one_and_a_half_pay():
    source = "He was entitled to time and a half."
    target = "Ему полагалась полуторная оплата."
    rows = v9._deterministic_findings(seg(source), target, BookMemory())
    assert not any(row["code"] == "number" and row["severity"] == "critical" for row in rows)


def test_dialogue_formatter_does_not_create_unbalanced_guillemets():
    source = seg("'No,' he said. 'Not today.'")
    value, _ = v9._format_dialogue_v9(source, "'Нет,' — сказал он. 'Не сегодня.'")
    assert value.count("«") == value.count("»")
    assert not value.startswith(("'", '"'))


def test_narrative_paired_quotes_become_safe_guillemets():
    source = seg("He called it 'necessary evil'.")
    value, _ = v9._format_dialogue_v9(source, 'Он называл это "необходимым злом".')
    assert "«необходимым злом»" in value
    assert value.count("«") == value.count("»")
