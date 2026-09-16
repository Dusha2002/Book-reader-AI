from bookai.v10_clause_fidelity import compare_duplicate_content_fidelity
from bookai.v10_semantic_hardening import source_has_cross_clause_repeat


def test_cross_clause_repeat_accepts_pronoun_linked_predicate():
    source = (
        "The stupid animal lifted its head and scowled at him. "
        "A moment later it lifted its head and kept still."
    )
    assert source_has_cross_clause_repeat(source) is True


def test_cross_clause_repeat_still_rejects_unrelated_prose():
    source = (
        "The horse backed away from the door. "
        "He picked up the sword and mounted awkwardly."
    )
    assert source_has_cross_clause_repeat(source) is False


def test_core_duplicate_detector_still_flags_real_target_duplication():
    source = "He opened the door, crossed the room, and sat beside the window."
    target = (
        "Он открыл дверь и медленно вошёл в комнату. "
        "Он открыл дверь и медленно вошёл в комнату."
    )
    result = compare_duplicate_content_fidelity(source, target)
    assert result["ok"] is False, result
