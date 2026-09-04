from bookai.models import GateFinding, Segment
from bookai.resilience import resilient_findings, resilient_segment_map


def test_structured_failure_splits_until_single_targets():
    segments = [
        Segment("s1", "One", "/p"),
        Segment("s2", "Two", "/p"),
        Segment("s3", "Three", "/p"),
        Segment("s4", "Four", "/p"),
    ]
    calls = []

    def call(chunk):
        calls.append([s.id for s in chunk])
        if len(chunk) > 1:
            raise ValueError("mock id contract violation")
        return {chunk[0].id: "перевод-" + chunk[0].id}

    out = resilient_segment_map(segments, call, label="editor", attempts=1)
    assert out == {s.id: "перевод-" + s.id for s in segments}
    assert ["s1", "s2", "s3", "s4"] in calls
    assert ["s1"] in calls and ["s4"] in calls


def test_provider_errors_do_not_fan_out_into_many_calls():
    segments = [Segment("s1", "One", "/p"), Segment("s2", "Two", "/p")]
    calls = 0

    def call(chunk):
        nonlocal calls
        calls += 1
        raise TimeoutError("provider unavailable")

    try:
        resilient_segment_map(segments, call, label="editor", attempts=2)
    except RuntimeError as exc:
        assert "provider/transport" in str(exc)
    else:
        raise AssertionError("expected provider failure")
    assert calls == 2


def test_critic_schema_failure_retries_then_splits():
    segments = [
        Segment("s1", "One", "/p"),
        Segment("s2", "Two", "/p"),
        Segment("s3", "Three", "/p"),
    ]
    calls = []

    def critic(chunk):
        calls.append([s.id for s in chunk])
        if len(chunk) > 1:
            raise TypeError("mock issues contract")
        if chunk[0].id == "s2":
            return [GateFinding("s2", "hard", "enumeration: missing item")]
        return []

    findings = resilient_findings(segments, critic, label="semantic_gate", attempts=1)
    assert [(f.id, f.severity) for f in findings] == [("s2", "hard")]
    assert ["s1", "s2", "s3"] in calls
    assert ["s1"] in calls and ["s2"] in calls and ["s3"] in calls


def test_persistent_single_critic_schema_failure_fails_closed():
    segment = Segment("s1", "One", "/p")

    def critic(_chunk):
        raise ValueError("bad schema")

    try:
        resilient_findings([segment], critic, label="literary_gate", attempts=2)
    except RuntimeError as exc:
        assert "failed strict QA contract" in str(exc)
    else:
        raise AssertionError("expected fail-closed critic error")
