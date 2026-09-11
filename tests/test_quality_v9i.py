from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bookai.models import Segment
import chapter_reference_translation_v9i as v9i


def seg(text: str, sid: str = "s1") -> Segment:
    return Segment(id=sid, text=text, locator="x", chapter="Chapter Three")


def test_v9i_short_either_is_priority_rescue():
    source = seg("'One daughter,' he said. 'I won't see either of them again, I don't suppose.'")
    assert v9i._must_referent_rescue(source)
    rows = v9i._synthetic_confirmed_rows_v9i([source], {"s1": "— Я больше ни их, ни себя не увижу."})
    assert len(rows) == 1
    assert rows[0]["severity"] == "critical"
    assert rows[0]["code"] == "referent"


def test_v9i_long_expository_either_not_forced():
    text = ("Torquatus and Vetranio loathed each other, and either of them would do whatever was necessary "
            "to stop the other getting power. " + "The political consequences were obvious. " * 30)
    assert not v9i._must_referent_rescue(seg(text))


def test_v9i_observed_should_literalization_is_critical():
    source = seg("'I should do,' the man replied. 'I used to make them.'")
    rows = v9i._synthetic_confirmed_rows_v9i([source], {"s1": "— Я должен, — ответил мужчина. — Раньше я их делал."})
    assert rows[0]["severity"] == "critical"
    assert rows[0]["code"] == "idiom"


def test_v9i_observed_with_it_literalization_is_critical():
    source = seg("'Cocky with it,' Orsea said. 'So, you're an escaped convict.'")
    rows = v9i._synthetic_confirmed_rows_v9i([source], {"s1": "— Самоуверен с этим, — сказал Орсеа."})
    assert rows[0]["severity"] == "critical"
    assert rows[0]["code"] == "idiom"
