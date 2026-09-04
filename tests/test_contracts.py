import pytest

from bookai.contracts import issue_rows


def test_zero_false_null_and_empty_issue_variants_mean_no_issues():
    assert issue_rows({"issues": 0}, "critic") == []
    assert issue_rows({"issues": False}, "critic") == []
    assert issue_rows({"issues": None}, "critic") == []
    assert issue_rows({"issues": "[]"}, "critic") == []
    assert issue_rows({"issues": "no issues"}, "critic") == []


def test_single_issue_object_is_unambiguously_normalized():
    row = {"id": "s1", "reason": "missing fact"}
    assert issue_rows({"issues": row}, "critic") == [row]


def test_positive_issue_count_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="issues must be an array"):
        issue_rows({"issues": 2}, "critic")


def test_non_object_issue_row_is_rejected():
    with pytest.raises(ValueError, match="not an object"):
        issue_rows({"issues": ["bad row"]}, "critic")
