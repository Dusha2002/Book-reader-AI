from __future__ import annotations

from bookai.gigachat_mt import GigaChatLightningBackend
from bookai.models import BookMemory, Segment
from bookai.quality import candidate_issues
from bookai.reference_profile import apply_reference_profile


def seg(sid: str, text: str, locator: str | None = None) -> Segment:
    return Segment(id=sid, text=text, locator=locator or f"/body/section/p[{sid}]", chapter="Chapter Nine")


def codes(segment: Segment, candidate: str, memory: BookMemory | None = None) -> set[str]:
    return {issue.code for issue in candidate_issues(segment, candidate, memory)}


def test_chapter_heading_is_deterministic_and_free():
    s = seg("s000001", "Chapter Eleven", "/body/section/title/p")
    assert GigaChatLightningBackend._deterministic_heading(s) == "Глава одиннадцать"


def test_strict_schema_requires_exact_batch_ids():
    batch = [seg("s000010", "Hello."), seg("s000011", "Goodbye.")]
    fmt = GigaChatLightningBackend._strict_response_format(batch)
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    schema = fmt["schema"]
    assert schema["required"] == ["s000010", "s000011"]
    assert set(schema["properties"]) == {"s000010", "s000011"}
    assert schema["additionalProperties"] is False


def test_context_is_explicitly_not_a_translation_target(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "stub")
    backend = GigaChatLightningBackend()
    memory = apply_reference_profile(BookMemory(rolling_summary="Орсеа уже знает о письме."))
    source = [
        seg("s000001", "Before paragraph."),
        seg("s000002", "Who?"),
        seg("s000003", "After paragraph."),
    ]
    prompt = backend._prompt([source[1]], memory, source_segments=source)
    assert "CONTEXT_ONLY (НЕ ПЕРЕВОДИТЬ)" in prompt
    assert "TARGETS (ПЕРЕВЕСТИ)" in prompt
    assert "Before paragraph." in prompt
    assert '"s000002": "Who?"' in prompt
    assert "Miel:" not in prompt


def test_short_reply_cannot_expand_into_context_paragraph():
    s = seg("s000020", "Who?")
    candidate = "Это очень длинный пересказ соседних событий. " * 8
    assert "short_expansion" in codes(s, candidate)


def test_service_instruction_leak_is_hard_failure():
    s = seg("s000021", "He put the coin on the table.")
    assert "service_leak" in codes(s, "Он положил монету на стол. Талер — так и оставь талером.")


def test_repeated_sentence_loop_is_detected():
    s = seg("s000022", "A long enough English paragraph " * 8)
    sentence = "Он снова посмотрел на дверь и ничего не сказал."
    assert "repetition_loop" in codes(s, " ".join([sentence] * 4))


def test_reference_profile_locks_miel_and_orsea_gender():
    memory = apply_reference_profile(BookMemory())
    assert memory.glossary["Miel"] == "Миэль"
    assert "gender=male" in memory.characters["Orsea"]
    assert "ru=Орсеа" in memory.characters["Orsea"]


def test_obvious_orsea_gender_drift_is_hard_failure():
    memory = apply_reference_profile(BookMemory())
    s = seg("s000023", "Orsea repeated the name and turned away.")
    assert "character_gender" in codes(s, "Орсеа повторила имя и отвернулась.", memory)


def test_good_orsea_gender_does_not_trigger():
    memory = apply_reference_profile(BookMemory())
    s = seg("s000024", "Orsea repeated the name and turned away.")
    assert "character_gender" not in codes(s, "Орсеа повторил имя и отвернулся.", memory)


def test_long_translation_expansion_is_hard_failure():
    s = seg("s000025", "This is a sufficiently long English paragraph. " * 5)
    candidate = "Это русский текст, который необоснованно разросся. " * 20
    assert "too_long" in codes(s, candidate)
