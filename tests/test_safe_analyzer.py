import json

from bookai.audit import safe_analyze_memory


class FlakyAnalyzer:
    def __init__(self):
        self.calls = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append(user)
        if len(self.calls) == 1:
            return '{"style": {broken json'
        return json.dumps(
            {
                "style": {
                    "narrative_voice": "Сухой и точный рассказчик.",
                    "rhythm": "Короткие удары чередуются с длинными техническими фразами.",
                    "dialogue": "Сдержанный разговорный регистр.",
                    "humor": "Сухая ирония.",
                    "taboos": ["Не сглаживать иронию."],
                },
                "glossary": {"engineer": "инженер"},
                "characters": {},
                "rolling_summary": "",
            },
            ensure_ascii=False,
        )


def test_safe_analyzer_retries_malformed_json_and_returns_memory():
    provider = FlakyAnalyzer()
    sample = ("Opening prose. " * 4000) + ("Middle prose. " * 4000) + ("Ending prose. " * 4000)

    memory = safe_analyze_memory(provider, sample)

    assert len(provider.calls) == 2
    assert memory.style.narrative_voice == "Сухой и точный рассказчик."
    assert memory.glossary["engineer"] == "инженер"
    # Retry uses the smaller representative budget rather than resending the
    # entire very large sample.
    assert len(provider.calls[1]) < len(provider.calls[0])
