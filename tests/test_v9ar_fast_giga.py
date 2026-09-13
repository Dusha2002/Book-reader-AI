from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import chapter_reference_translation_v9ar_fast_giga as ar  # noqa: E402


def seg(sid: str, text: str):
    return SimpleNamespace(id=sid, text=text)


def test_medium_omission_and_dozen_precision():
    source = "First thought. Second thought. Third thought. " + "x" * 150
    assert ar._medium_omission(source, "Первая мысль. " + "я" * 50)
    assert ar._dozen_mistranslation("A dozen imaginary goblins appeared.", "Появилось полдюжины воображаемых гоблинов.")
    assert not ar._dozen_mistranslation("Half a dozen times.", "Полдюжины раз.")


def test_fluency_gate_rejects_translator_residue():
    assert ar._bad_repair_fluency("железной заготовки (полосы/прутка)")
    assert ar._bad_repair_fluency("После а за (он потерял счет времени)...")
    assert not ar._bad_repair_fluency("Он взял железную заготовку и положил её на верстак.")


def test_context_workshop_and_gender_guards():
    targets = [
        seg("s1", "The old man picked up the file and inspected its teeth."),
        seg("s2", "It's my file."),
        seg("s3", "Chalk it before you use it."),
        seg("s4", "Miel said nothing."),
    ]
    translated = {
        "s1": "Старик взял напильник и осмотрел его зубья.",
        "s2": "Это моё досье.",
        "s3": "Заточи его перед работой.",
        "s4": "Миль говорила молча.",
    }
    memory = SimpleNamespace(characters={"Miel": "ru=Миль;gender=male;role=noble"})
    assert "context:file_tool" in ar._context_codes_for_index(targets, translated, memory, 1)
    assert "context:chalk_file" in ar._context_codes_for_index(targets, translated, memory, 2)
    assert "context:male_character_feminine_verb" in ar._context_codes_for_index(targets, translated, memory, 3)


def test_future_completion_guard():
    targets = [seg("s1", "A day and a half, and the job would be finished.")]
    translated = {"s1": "Прошло полтора дня, и работа была закончена."}
    memory = SimpleNamespace(characters={})
    assert "context:future_completion" in ar._context_codes_for_index(targets, translated, memory, 0)
