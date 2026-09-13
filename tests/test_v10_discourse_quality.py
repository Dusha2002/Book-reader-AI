from bookai.models import BookMemory, Segment
from bookai.v10_numeric import compare_numeric_fidelity_v10
from bookai.v10_quality import V10QualityQA
from bookai.v10_speaker import DialogueSpeakerContinuityGuard
from bookai.v10_transport import RobustTaggedPrimaryTransport, _looks_like_prompt_leak


def seg(text: str, sid: str) -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter="Chapter One")


def test_transport_rejects_prompt_echo_inside_well_formed_tag():
    leaked = (
        '<s id="s000001">Валенс: ru=Валенс;gender=unknown;role=;voice= '
        'Глоссарий: Валенс = ru=Валенс Источник: Нормальный перевод.</s>'
    )
    assert _looks_like_prompt_leak(leaked)
    assert RobustTaggedPrimaryTransport.parse_tagged(leaked, {"s000001"}) == {}


def test_transport_accepts_normal_literary_output():
    raw = '<s id="s000001">Колотушка ударила двенадцать раз и смолкла.</s>'
    assert RobustTaggedPrimaryTransport.parse_tagged(raw, {"s000001"}) == {
        "s000001": "Колотушка ударила двенадцать раз и смолкла."
    }


def test_v10_numeric_accepts_inflected_hundreds_compound():
    check = compare_numeric_fidelity_v10(
        "It was a three hundred hour job.",
        "Это была трёхсотчасовая работа.",
    )
    assert check["ok"]


def test_v10_numeric_accepts_prepositional_hundreds():
    check = compare_numeric_fidelity_v10(
        "He stopped three hundred yards away.",
        "Он остановился в трёхстах ярдах оттуда.",
    )
    assert check["ok"]


def test_v10_numeric_accepts_natural_one_or_two_to_paru():
    check = compare_numeric_fidelity_v10(
        "Give it one or two minutes.",
        "Дай ему пару минут.",
    )
    assert check["ok"]


def test_quality_detects_mixed_script_word():
    issues = V10QualityQA().scan_segment(
        seg("Licinius and Vetranio were safely locked up.", "s000001"),
        "Лициний и Ветраниio были надежно заперты.",
        BookMemory(),
    )
    assert any(issue.code == "latin_leak" for issue in issues)


def test_quality_routes_bad_hunting_collocation_to_semantic_specialist():
    issues = V10QualityQA().scan_segment(
        seg("They would draw the home coverts in the morning, and drive down the mill-stream in the afternoon.", "s000001"),
        "Утром они должны были выследить домашних птиц, а днем проехать по мельничной запруде.",
        BookMemory(),
    )
    assert any(issue.code == "hunting_collocation" and issue.mode == "semantic" for issue in issues)


def test_speaker_continuity_fixes_gender_only_with_strong_two_speaker_sandwich():
    segments = [
        seg("'Bolt out of the blue,' Licinius said.", "s000001"),
        seg("'Which daughter?' Valens said.", "s000002"),
        seg("'What? Oh, right. I'm not absolutely sure.'", "s000003"),
        seg("'Can you find out?' Valens said.", "s000004"),
        seg("'I've already said yes.'", "s000005"),
        seg("'That's splendid.' Valens said.", "s000006"),
    ]
    translated = {
        "s000001": "— Как гром среди ясного неба, — сказал Лициний.",
        "s000002": "— Которая дочь? — спросил Валенс.",
        "s000003": "— Что? Я не совсем уверен.",
        "s000004": "— Можешь выяснить? — сказал Валенс.",
        "s000005": "— Я уже сказала да.",
        "s000006": "— Вот и прекрасно, — сказал Валенс.",
    }
    guard = DialogueSpeakerContinuityGuard()
    changed = guard.apply(segments, translated)
    assert "s000005" in changed
    assert "Я уже сказал да" in translated["s000005"]
    assert guard.stats["inferred_speakers"] >= 2


def test_speaker_continuity_does_not_guess_without_two_named_speakers():
    segments = [
        seg("'All right,' Valens said.", "s000001"),
        seg("'I've already said yes.'", "s000002"),
    ]
    translated = {
        "s000001": "— Хорошо, — сказал Валенс.",
        "s000002": "— Я уже сказала да.",
    }
    guard = DialogueSpeakerContinuityGuard()
    assert guard.apply(segments, translated) == []
    assert translated["s000002"] == "— Я уже сказала да."
