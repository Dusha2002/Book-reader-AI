from bookai.models import Segment
from bookai.release_guards import (
    provenance_entry,
    source_fingerprint,
    source_grounded_objective_issues,
    specialist_guard_routes,
)


def _segment(sid: str, text: str) -> Segment:
    return Segment(sid, text, "/body/section/p", chapter="Chapter Test")


def test_provenance_fingerprint_tracks_source_and_context():
    before = _segment("s1", "Before.")
    target = _segment("s2", "Target source.")
    after = _segment("s3", "After.")

    row = provenance_entry(target, [before], [after])

    assert row["source_id"] == "s2"
    assert row["source_hash"] == source_fingerprint("Target source.")
    assert row["context_ids"] == ["s1", "s3"]
    assert row["stage"] == "draft"


def test_double_negative_scope_is_mandatory_specialist_route():
    targets = [_segment("s1", "'I don't see why not,' Calaphates said. 'If necessary.'")]
    translated = {"s1": "'Вижу причин не видеть', — сказал Калафат. 'Если потребуется.'"}

    routes = specialist_guard_routes(targets, translated)

    assert len(routes) == 1
    assert routes[0]["code"] == "polarity_scope"
    assert routes[0]["priority"] >= 22


def test_large_unexplained_expansion_is_routed_as_source_contamination():
    source = (
        "A little later they brought him up some food; it was a depressing thought "
        "that the garbage on his plate probably counted as the best this country could offer."
    )
    targets = [_segment("s1", source)]
    translated = {
        "s1": (
            "К делу это не относится. Был момент, когда ему хотелось убить Дукаса или хотя бы сильно "
            "навредить ему, просто за грубое слово. Позже принесли еду; было тоскливо думать, что мусор "
            "на тарелке, вероятно, и есть лучшее, что могла предложить эта страна."
        )
    }

    routes = specialist_guard_routes(targets, translated)

    assert len(routes) == 1
    assert routes[0]["code"] == "source_contamination"
    assert routes[0]["priority"] >= 22


def test_brigandine_and_needle_eye_regressions_are_mandatory_specialist_routes():
    source = (
        "A brigandine or even a thick winter coat would turn one of those points. "
        "The eye of a darning-needle, probably."
    )
    targets = [_segment("s1", source)]
    translated = {"s1": "Кольчуга остановила бы укол. Глазок штопальной иглы, наверное."}

    routes = specialist_guard_routes(targets, translated)

    assert len(routes) == 1
    assert routes[0]["priority"] >= 30
    assert "armor_terminology" in routes[0]["guard_codes"]
    assert "needle_eye_idiom" in routes[0]["guard_codes"]
    assert "бригантина" in routes[0]["reason"]
    assert "ушко штопальной иглы" in routes[0]["reason"]


def test_last_lesson_but_one_is_routed_as_penultimate_idiom():
    targets = [_segment("s1", "Fencing was last lesson but one on a Monday.")]
    translated = {"s1": "Фехтование было последним уроком перед одним понедельником."}

    routes = specialist_guard_routes(targets, translated)

    assert len(routes) == 1
    assert routes[0]["code"] == "penultimate_idiom"
    assert routes[0]["priority"] >= 30
    assert "предпослед" in routes[0]["reason"]


def test_source_grounded_objective_issues_capture_same_release_failures():
    targets = [
        _segment(
            "s1",
            "A brigandine would turn the point. The eye of a darning-needle, probably.",
        ),
        _segment("s2", "Fencing was last lesson but one on a Monday."),
    ]
    translated = {
        "s1": "Кольчуга остановила бы укол. Глазок штопальной иглы, наверное.",
        "s2": "Фехтование было последним уроком перед одним понедельником.",
    }

    issues = source_grounded_objective_issues(targets, translated)
    by_id = {row["id"]: row for row in issues}

    assert set(by_id["s1"]["codes"]) == {"armor_terminology", "needle_eye_idiom"}
    assert by_id["s2"]["codes"] == ["penultimate_idiom"]


def test_source_lexical_guards_do_not_fire_when_meaning_is_preserved():
    targets = [
        _segment("s1", "A brigandine would turn the point; the eye of a darning-needle was smaller."),
        _segment("s2", "Fencing was last lesson but one on a Monday."),
    ]
    translated = {
        "s1": "Бригантина остановила бы укол; ушко штопальной иглы было ещё меньше.",
        "s2": "По понедельникам фехтование было предпоследним уроком.",
    }

    assert specialist_guard_routes(targets, translated) == []
    assert source_grounded_objective_issues(targets, translated) == []


def test_clean_compact_translation_is_not_routed_by_guard():
    targets = [_segment("s1", "Ziani nodded. 'Thank you,' he said.")]
    translated = {"s1": "Зиани кивнул. — Спасибо, — сказал он."}

    assert specialist_guard_routes(targets, translated) == []
    assert source_grounded_objective_issues(targets, translated) == []
