import pytest

from bookai.critics import _resolve_issue_id


def test_exact_critic_id_is_unchanged():
    assert _resolve_issue_id("s003324", {"s003324"}, "literary gate") == "s003324"


def test_numbered_critic_sub_id_maps_to_existing_parent_only():
    valid = {"s003324", "s003325"}
    assert _resolve_issue_id("s003324_1", valid, "literary gate") == "s003324"
    assert _resolve_issue_id("s003325-2", valid, "semantic gate") == "s003325"


def test_single_target_positional_id_is_normalized_only_when_unambiguous():
    assert _resolve_issue_id("1", {"s000184"}, "literary gate") == "s000184"
    assert _resolve_issue_id("0", {"s000184"}, "semantic gate") == "s000184"

    with pytest.raises(ValueError, match="unknown id"):
        _resolve_issue_id("1", {"s000184", "s000185"}, "literary gate")


def test_unknown_or_ambiguous_critic_id_is_rejected():
    valid = {"s003324"}
    with pytest.raises(ValueError, match="unknown id"):
        _resolve_issue_id("s003999_1", valid, "literary gate")
    with pytest.raises(ValueError, match="unknown id"):
        _resolve_issue_id("s003324_extra", valid, "literary gate")
    with pytest.raises(ValueError, match="unknown id"):
        _resolve_issue_id("2", valid, "literary gate")
