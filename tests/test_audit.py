import json

from bookai.audit import brief_is_corrupt, safe_chapter_brief, semantic_gate_batch
from bookai.models import BookMemory, Segment


class SequenceProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "mock"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        self.usage["requests"] += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_corrupt_mixed_script_brief_is_retried():
    assert brief_is_corrupt("Сцена про Валенса, но затем 模型开始胡说 и теряет контекст.")
    provider = SequenceProvider([
        json.dumps({"brief": "Сцена про Валенса, затем 模型 начинает ломаться."}),
        json.dumps({"brief": "Валенс тренируется; сухая ирония строится на бесполезности изящного фехтования."}),
    ])
    segments = [Segment("s000001", "Valens practised fencing.", "/p")]
    brief = safe_chapter_brief(provider, segments, BookMemory(), attempts=2)
    assert "сухая ирония" in brief
    assert len(provider.calls) == 2


def test_brief_transport_failures_fall_back_without_inventing_plot():
    provider = SequenceProvider([TimeoutError("slow"), TimeoutError("slow again")])
    segments = [Segment("s000001", "Valens practised fencing.", "/p")]
    brief = safe_chapter_brief(provider, segments, BookMemory(), attempts=2)
    assert len(provider.calls) == 2
    assert "исходный текст" in brief
    assert "не добавляй фактов" in brief


def test_semantic_gate_promotes_missing_enumeration_to_hard():
    provider = SequenceProvider([
        json.dumps({
            "issues": [{
                "id": "s000010",
                "code": "enumeration",
                "reason": "One of four fencing weapons/terms is omitted from the Russian candidate.",
            }]
        })
    ])
    segment = Segment(
        "s000010",
        "He learned the stock, the tuck, the small-sword and the rapier.",
        "/p",
    )
    findings = semantic_gate_batch(
        provider,
        [segment],
        {"s000010": "Он учился эстоку, шпаге и рапире."},
        BookMemory(),
    )
    assert len(findings) == 1
    assert findings[0].id == "s000010"
    assert findings[0].severity == "hard"
    assert findings[0].reason.startswith("enumeration:")
