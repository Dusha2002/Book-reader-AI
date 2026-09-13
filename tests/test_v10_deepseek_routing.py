import json

from bookai.models import BookMemory, Segment
from bookai.v10 import V10Issue
from bookai.v10_deepseek import DeepSeekSemanticSpecialist
from bookai.v10_quality import V10QualityQA


def seg(text: str, sid: str) -> Segment:
    return Segment(id=sid, text=text, locator=f"/{sid}", chapter="Chapter One")


class FakeProvider:
    def __init__(self, corrected: dict[str, str]):
        self.corrected = corrected
        self.calls = 0

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1
        payload = json.loads(prompt)
        items = []
        for row in payload["items"]:
            sid = row["id"]
            candidate = self.corrected.get(sid, row["current_ru"])
            items.append({
                "id": sid,
                "change": candidate != row["current_ru"],
                "corrected_ru": candidate,
                "confidence": 0.95,
                "reason": "fix proven fidelity defect",
            })
        return json.dumps({"items": items}, ensure_ascii=False)


def test_proven_quantity_displaces_risk_only_candidate_when_batch_is_full():
    qa = V10QualityQA()
    proven = seg("He made two dozen lunges.", "s000001")
    risky = seg("He would never do it because either choice would hurt them.", "s000002")
    issues = [V10Issue("s000001", "quantity_obligation", "local", "hard", "24 was lost")]
    specialist = DeepSeekSemanticSpecialist(FakeProvider({}), qa, max_segments=1)
    selected = specialist._select([risky, proven], {}, BookMemory(), issues)
    assert [row.id for row in selected] == ["s000001"]


def test_single_deepseek_batch_fixes_proven_quantity_and_accepts_only_after_detector_clears():
    qa = V10QualityQA()
    segment = seg("He made two dozen lunges.", "s000001")
    translated = {"s000001": "Он сделал два десятка выпадов."}
    issues = qa.scan_segment(segment, translated[segment.id], BookMemory())
    provider = FakeProvider({"s000001": "Он сделал двадцать четыре выпада."})
    specialist = DeepSeekSemanticSpecialist(provider, qa, max_segments=8)

    changed = specialist.repair([segment], translated, BookMemory(), issues)

    assert provider.calls == 1
    assert changed == ["s000001"]
    assert translated["s000001"] == "Он сделал двадцать четыре выпада."
    assert specialist.stats["forced_proven_selected"] == 1
    assert specialist.stats["forced_proven_accepted"] == 1
    assert not any(
        issue.code in {"numeric", "quantity_obligation", "numbered_choice", "quarter_inch", "short_omission", "omission"}
        for issue in qa.scan_segment(segment, translated[segment.id], BookMemory())
    )


def test_proven_quantity_rewrite_is_rejected_when_required_defect_survives():
    qa = V10QualityQA()
    segment = seg("He made two dozen lunges.", "s000001")
    translated = {"s000001": "Он сделал два десятка выпадов."}
    issues = qa.scan_segment(segment, translated[segment.id], BookMemory())
    provider = FakeProvider({"s000001": "Он выполнил два десятка выпадов."})
    specialist = DeepSeekSemanticSpecialist(provider, qa, max_segments=8)

    changed = specialist.repair([segment], translated, BookMemory(), issues)

    assert provider.calls == 1
    assert changed == []
    assert translated["s000001"] == "Он сделал два десятка выпадов."
    assert specialist.stats["forced_proven_rejected"] == 1
