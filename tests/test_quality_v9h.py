from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bookai.models import BookMemory, Segment
import chapter_reference_translation_v9h as v9h


def seg(text: str) -> Segment:
    return Segment(id="s1", text=text, locator="x", chapter="Chapter Three")


def test_v9h_repairs_factual_major_at_high_confidence():
    assert v9h._needs_repair_v9h([{"severity": "major", "confidence": 0.86, "code": "referent", "reason": "wrong antecedent"}])
    assert v9h._needs_repair_v9h([{"severity": "major", "confidence": 0.86, "code": "omission", "reason": "clause omitted"}])


def test_v9h_does_not_repair_low_confidence_style_major():
    assert not v9h._needs_repair_v9h([{"severity": "major", "confidence": 0.84, "code": "grammar", "reason": "possible awkwardness"}])
    assert not v9h._needs_repair_v9h([{"severity": "major", "confidence": 0.89, "code": "idiom", "reason": "possible idiom preference"}])


def test_v9h_known_literalization_bypasses_style_threshold():
    assert v9h._needs_repair_v9h([{
        "severity": "major", "confidence": 0.88, "code": "idiom",
        "reason": "observed Russian draft matches a known literalization pattern: preposition_idiom",
    }])


def test_v9h_deterministic_judge_prefers_safe_proposed_without_llm():
    source = seg("'Cocky with it,' Orsea said.")
    current = "— Самоуверен с этим, — сказал Орсеа."
    proposed = "— И ещё дерзит, — сказал Орсеа."
    chosen = v9h._judge_candidates_v9h(None, [source], source, [current, proposed], BookMemory())
    assert chosen == proposed
