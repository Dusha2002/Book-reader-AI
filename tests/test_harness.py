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
            return json.dumps({"style": {}, "glossary": {}, "characters": {}, "rolling_summary": ""})
        if self.role == "translator":
            ids = []
            for token in user.split('"'):
                if token.startswith('s') and len(token) == 7 and token[1:].isdigit() and token not in ids:
                    ids.append(token)
            mapping = {"s000000": "Первый.", "s000001": "ПЛОХО", "s000002": "Третий."}
            return json.dumps({sid: mapping.get(sid, "Перевод.") for sid in ids})
        if self.role == "gate":
            return json.dumps({"issues": [{"id": "s000001", "severity": "medium", "reason": "awkward"}]})
        if self.role == "editor":
            return json.dumps({"s000001": "Хороший литературный перевод."})
        if self.role == "memory":
            return json.dumps({"glossary": {}, "characters": {}, "rolling_summary": "", "last_chapter": ""})
        if self.role == "hard":
            return json.dumps({})
        return json.dumps({})


def test_optimal_only_edits_flagged_segments(tmp_path: Path):
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
    assert len(editor.calls) == 1
    assert '"s000001"' in editor.calls[0][1]
    assert '"s000000"' not in editor.calls[0][1]
    assert not hard.calls
