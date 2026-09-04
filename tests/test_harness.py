import json
from pathlib import Path

from bookai.harness import LLMTranslator, TranslationHarness
from bookai.pipeline import translate_book


class RoleProvider:
    def __init__(self, role: str):
        self.role = role
        self.model = role
        self.calls = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        self.usage["requests"] += 1
        if self.role == "analyzer":
            if "CHAPTER_SOURCE:" in user:
                return json.dumps({"brief": "Короткая локальная сцена."})
            return json.dumps({"style": {}, "glossary": {}, "characters": {}, "rolling_summary": ""})
        if self.role == "translator":
            ids = []
            for token in user.split('"'):
                if token.startswith('s') and len(token) == 7 and token[1:].isdigit() and token not in ids:
                    ids.append(token)
            mapping = {"s000000": "Первый.", "s000001": "ПЛОХО", "s000002": "Третий."}
            return json.dumps({sid: mapping.get(sid, "Перевод.") for sid in ids})
        if self.role == "gate":
            if "SEMANTIC COVERAGE auditor" in system:
                return json.dumps({"issues": []})
            if "Choose the better of two Russian translations" in system:
                ids = []
                for token in user.split('"'):
                    if token.startswith('s') and len(token) == 7 and token[1:].isdigit() and token not in ids:
                        ids.append(token)
                return json.dumps({sid: "B" for sid in ids})
            return json.dumps({"issues": [{"id": "s000001", "severity": "medium", "reason": "awkward"}]})
        if self.role == "editor":
            if "LITERARY POLISH PASS" in system:
                raw = user.split("TARGETS:", 1)[1].split("\nSilently", 1)[0]
                targets = json.loads(raw)
                return json.dumps({sid: item["faithful_draft_ru"] for sid, item in targets.items()})
            return json.dumps({"s000001": "Хороший литературный перевод."})
        if self.role == "memory":
            return json.dumps({"glossary": {}, "characters": {}, "rolling_summary": "", "last_chapter": ""})
        if self.role == "hard":
            return json.dumps({})
        return json.dumps({})


def test_optimal_polishes_all_then_only_repairs_flagged_segments(tmp_path: Path):
    src = tmp_path / "book.fb2"
    src.write_text(
        '<?xml version="1.0"?><FictionBook><body><section><p>First.</p><p>Second.</p><p>Third.</p></section></body></FictionBook>',
        "utf-8",
    )
    analyzer = RoleProvider("analyzer")
    translator_p = RoleProvider("translator")
    gate = RoleProvider("gate")
    editor = RoleProvider("editor")
    hard = RoleProvider("hard")
    memory = RoleProvider("memory")
    harness = TranslationHarness(analyzer, LLMTranslator(translator_p), gate, editor, hard, memory)

    out = tmp_path / "out.fb2"
    translate_book(src, out, harness, mode="optimal", cache_dir=tmp_path / "cache")
    text = out.read_text("utf-8")

    assert "Первый." in text
    assert "Хороший литературный перевод." in text
    assert "Третий." in text

    polish_calls = [(system, user) for system, user in editor.calls if "LITERARY POLISH PASS" in system]
    repair_calls = [(system, user) for system, user in editor.calls if "repair pass" in system]
    assert len(polish_calls) == 1
    polish_targets = json.loads(polish_calls[0][1].split("TARGETS:", 1)[1].split("\nSilently", 1)[0])
    assert set(polish_targets) == {"s000000", "s000001", "s000002"}

    # The literary mock keeps flagging the same segment, so two bounded repairs
    # are allowed. The independent semantic auditor reports no semantic defect.
    assert len(repair_calls) == 2
    for _system, user in repair_calls:
        pairs = json.loads(user.split("PAIRS:", 1)[1])
        assert set(pairs) == {"s000001"}
        context = user.split("PAIRS:", 1)[0]
        assert '"s000000"' in context
        assert '"s000002"' in context
    assert not hard.calls
