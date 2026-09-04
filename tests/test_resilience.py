from bookai.models import Segment
from bookai.resilience import resilient_segment_map


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
