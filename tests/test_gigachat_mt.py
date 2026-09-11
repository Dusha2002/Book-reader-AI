from __future__ import annotations

from bookai.gigachat_mt import GigaChatLightningBackend
from bookai.models import BookMemory, Segment


def segment(sid: str, text: str) -> Segment:
    return Segment(id=sid, text=text, locator=sid, chapter="Chapter One")


def test_gigachat_backend_requires_auth_key(monkeypatch):
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    backend = GigaChatLightningBackend()
    assert backend.available() is False
    assert backend.backend_name == "gigachat-3-lightning"


def test_parse_json_object_accepts_markdown_fence():
    parsed = GigaChatLightningBackend._parse_json_object(
        '```json\n{"s1":"Перевод один","s2":"Перевод два"}\n```'
    )
    assert parsed == {"s1": "Перевод один", "s2": "Перевод два"}


def test_parse_json_object_extracts_object_from_wrapper():
    parsed = GigaChatLightningBackend._parse_json_object(
        'Вот результат:\n{"s1":"Полный перевод"}\nГотово.'
    )
    assert parsed == {"s1": "Полный перевод"}


def test_batching_respects_segment_count(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "stub")
    monkeypatch.setenv("BOOKAI_GIGACHAT_BATCH_SEGMENTS", "2")
    monkeypatch.setenv("BOOKAI_GIGACHAT_BATCH_CHARS", "10000")
    backend = GigaChatLightningBackend()
    batches = backend._batches([
        segment("s1", "one"),
        segment("s2", "two"),
        segment("s3", "three"),
    ])
    assert [[s.id for s in batch] for batch in batches] == [["s1", "s2"], ["s3"]]


def test_batching_respects_character_budget(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "stub")
    monkeypatch.setenv("BOOKAI_GIGACHAT_BATCH_SEGMENTS", "8")
    monkeypatch.setenv("BOOKAI_GIGACHAT_BATCH_CHARS", "1000")
    backend = GigaChatLightningBackend()
    batches = backend._batches([
        segment("s1", "a" * 700),
        segment("s2", "b" * 700),
    ])
    assert len(batches) == 2


def test_prompt_contains_style_context_and_locked_glossary(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "stub")
    backend = GigaChatLightningBackend()
    memory = BookMemory()
    memory.style.narrative_voice = "Сдержанный сухой рассказчик."
    memory.style.rhythm = "Сохранять длинные синтаксические периоды."
    memory.rolling_summary = "Валенс работает над механизмом; отношения персонажей уже установлены."
    memory.glossary = {"Valens": "Валенс", "Orsea": "Орсеа"}
    prompt = backend._prompt([segment("s1", "Valens examined the mechanism.")], memory)
    assert "Сдержанный сухой рассказчик" in prompt
    assert "Сохранять длинные синтаксические периоды" in prompt
    assert "Валенс работает над механизмом" in prompt
    assert "Valens=Валенс" in prompt
    assert "Orsea=Орсеа" not in prompt


def test_translate_many_does_not_fallback_without_auth(monkeypatch):
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    backend = GigaChatLightningBackend()
    rows, errors = backend.translate_many([segment("s1", "Hello")], BookMemory())
    assert rows == {}
    assert "s1" in errors
    assert "GIGACHAT_AUTH_KEY" in errors["s1"]
