from bookai.models import BookMemory, Segment
from bookai.quality_v3 import enhanced_candidate_issues, infer_active_speaker
from bookai.reference_profile import apply_reference_profile
from bookai.gigachat_v3 import GigaChatLightningV3Backend


def memory():
    return apply_reference_profile(BookMemory())


def test_heading_is_deterministic_even_without_title_locator():
    segment = Segment("s000001", "Chapter Twelve", "/body/section/p[1]", "Chapter Twelve")
    assert GigaChatLightningV3Backend._deterministic_heading(segment) == "Глава двенадцать"


def test_context_leak_sentence_inflation_is_hard():
    segment = Segment(
        "s000010",
        "A little later they brought him some food; he ate it, lay down on the bed and stared at the ceiling until he fell asleep. Veatriz Sirupati to Valens Valentinianus; greetings.",
        "/p",
        "Chapter Nine",
    )
    candidate = (
        "К делу это не относится. Был момент, когда ему хотелось убить Дукаса. "
        "Позже ему принесли еду; он поел, лёг и смотрел в потолок, пока не заснул. "
        "Веатрис Сирупати — Валенсу Валентиниану; приветствую."
    )
    issues = enhanced_candidate_issues(segment, candidate, memory())
    assert any(i.severity == "hard" and i.code == "context_leak" for i in issues)


def test_large_omission_is_hard():
    segment = Segment(
        "s000011",
        "(I'm not thinking, of course. Since most of that time we've been at war with you, presumably it's pretty much the same with your people; so you understand it better than I can. I hate the fact that I spent so much time living away from here when I was young. This might as well be a foreign country, for all I understand it.)",
        "/p",
        "Chapter Nine",
    )
    candidate = "(Я, разумеется, не думаю. Поскольку большую часть времени мы воевали с вами, вы понимаете это лучше меня.)"
    issues = enhanced_candidate_issues(segment, candidate, memory())
    assert any(i.severity == "hard" and i.code in {"semantic_omission", "sentence_loss"} for i in issues)


def test_letter_speaker_gender_is_detected():
    rows = [
        Segment("s000020", "Veatriz Sirupati to Valens Valentinianus; greetings.", "/p", "Chapter Nine"),
        Segment("s000021", "If I think about it, I'm almost sure it isn't true.", "/p", "Chapter Nine"),
    ]
    mem = memory()
    speaker = infer_active_speaker(rows[1], rows, mem)
    assert speaker and speaker[0] == "Veatriz" and speaker[1] == "female"
    issues = enhanced_candidate_issues(
        rows[1],
        "Если подумать, я почти уверен, что это неправда.",
        mem,
        source_segments=rows,
    )
    assert any(i.severity == "hard" and i.code == "speaker_gender" for i in issues)


def test_mixed_latin_cyrillic_token_is_hard():
    segment = Segment(
        "s000030",
        "The overhead shafts carry the drive along the gallery.",
        "/p",
        "Chapter Twelve",
    )
    issues = enhanced_candidate_issues(
        segment,
        "Надставные shaftы передают привод вдоль галереи.",
        memory(),
    )
    assert any(i.severity == "hard" and i.code == "mixed_script_token" for i in issues)


def test_single_latin_residue_is_hard_in_russian_prose():
    segment = Segment(
        "s000031",
        "He treated prove as a legitimate rhyme for love.",
        "/p",
        "Chapter Twelve",
    )
    issues = enhanced_candidate_issues(
        segment,
        "Он считал prove вполне допустимой рифмой к love.",
        memory(),
    )
    assert any(i.severity == "hard" and i.code == "latin_residue" for i in issues)


def test_missing_interrogative_is_hard_even_when_length_looks_normal():
    segment = Segment(
        "s000032",
        "When did I stop doing useful work? Was it about the time I acquired power over my fellow citizens?",
        "/p",
        "Chapter Twelve",
    )
    candidate = "Когда я перестал заниматься полезной работой? Наверное, примерно тогда я и получил власть над согражданами."
    issues = enhanced_candidate_issues(segment, candidate, memory())
    assert any(i.severity == "hard" and i.code == "question_loss" for i in issues)


def test_dynamic_character_rendering_is_hard_consistency_constraint():
    mem = BookMemory(
        glossary={"Falier": "Фалиер"},
        characters={"Falier": "ru=Фалиер; gender=male; мужчина"},
    )
    segment = Segment("s000033", "Falier looked at the machine.", "/p", "Chapter Twelve")
    issues = enhanced_candidate_issues(segment, "Фалир посмотрел на машину.", mem)
    assert any(i.severity == "hard" and i.code == "character_name" for i in issues)


def test_dynamic_named_glossary_term_is_hard_consistency_constraint():
    mem = BookMemory(glossary={"Lonazep": "Лоназеп"})
    segment = Segment("s000034", "The ship reached Lonazep before dawn.", "/p", "Chapter Twelve")
    issues = enhanced_candidate_issues(segment, "Корабль достиг Лонацепа до рассвета.", mem)
    assert any(i.severity == "hard" and i.code == "entity_consistency" for i in issues)
