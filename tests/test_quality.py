import pytest

from bookai.harness import FLASH_MODEL, LLMTranslator, TranslationHarness
from bookai.models import BookMemory, Segment
from bookai.quality import assert_exact_ids, candidate_issues


def codes(segment: Segment, text: str):
    return {x.code for x in candidate_issues(segment, text, BookMemory())}


def test_rejects_unexpected_script_and_english_fallback():
    s = Segment("s000001", "The machine struck the anvil twelve times.", "/body/section/p")
    assert "unexpected_script" in codes(s, "Машина ударила по наковальне 十二 раз.")
    assert "unchanged" in codes(s, s.text)


def test_rejects_changed_numbers():
    s = Segment("s000002", "He waited 12 days and paid 4 marks.", "/body/section/p")
    assert "numbers" in codes(s, "Он ждал двенадцать дней и заплатил 5 марок.")


def test_heading_cannot_expand_into_body():
    s = Segment("s000003", "Chapter Three", "/body/section/title/p")
    assert "heading_multiline" in codes(s, "Глава третья\nА затем началась длинная история")


def test_exact_id_contract_never_silently_falls_back():
    segments = [
        Segment("s000001", "One", "/p"),
        Segment("s000002", "Two", "/p"),
    ]
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(segments, {"s000001": "Один"}, "translator")
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(segments, {"s000001": "Один", "s000002": "Два", "extra": "x"}, "translator")


def test_single_segment_wrapper_is_recovered_but_multi_segment_wrapper_is_rejected():
    one = [Segment("s000010", "He left.", "/p")]
    assert assert_exact_ids(one, {"type": "translation", "translation": "Он ушёл."}, "editor") == {
        "s000010": "Он ушёл."
    }

    many = [
        Segment("s000010", "He left.", "/p"),
        Segment("s000012", "She stayed.", "/p"),
    ]
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(many, {"type": "translation", "translation": "Он ушёл."}, "editor")


def test_exact_id_transport_envelope_is_safely_unwrapped():
    one = [Segment("s000008", "He left.", "/p")]
    assert assert_exact_ids(
        one,
        {"type": "json_object", "data": {"s000008": "Он ушёл."}},
        "literary polisher",
    ) == {"s000008": "Он ушёл."}

    many = [
        Segment("s000010", "He left.", "/p"),
        Segment("s000012", "She stayed.", "/p"),
    ]
    assert assert_exact_ids(
        many,
        {
            "type": "json_object",
            "data": {"s000010": "Он ушёл.", "s000012": "Она осталась."},
        },
        "literary polisher",
    ) == {"s000010": "Он ушёл.", "s000012": "Она осталась."}


def test_transport_envelope_stays_strict_on_wrong_nested_ids():
    segments = [
        Segment("s000010", "He left.", "/p"),
        Segment("s000012", "She stayed.", "/p"),
    ]
    with pytest.raises(ValueError, match="id contract"):
        assert_exact_ids(
            segments,
            {"type": "json_object", "data": {"s000010": "Он ушёл.", "wrong": "Ошибка"}},
            "literary polisher",
        )


def test_env_cannot_raise_model_above_flash(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setenv("BOOKAI_MAX_MODEL", "deepseek/deepseek-v4-pro-0813")
    monkeypatch.setenv("BOOKAI_EDITOR_MODEL", "qwen/qwen3.8-flash")
    monkeypatch.setenv("BOOKAI_HARD_MODEL", "deepseek/deepseek-v4-pro-0813")
    monkeypatch.setenv("BOOKAI_TRANSLATOR_BACKEND", "api")

    harness = TranslationHarness.from_env()
    assert harness.analyzer.model == FLASH_MODEL
    assert harness.gate.model == FLASH_MODEL
    assert harness.editor.model == FLASH_MODEL
    assert harness.hard_editor.model == FLASH_MODEL
    assert harness.memory_model.model == FLASH_MODEL
    assert isinstance(harness.translator, LLMTranslator)
    assert harness.translator.provider.model == FLASH_MODEL
