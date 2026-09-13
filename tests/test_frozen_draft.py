from pathlib import Path

from bookai.frozen_draft import make_frozen_draft_backend
from bookai.models import BookMemory, Segment


class DummyBackend:
    name = "dummy"

    def __init__(self):
        self.calls = 0

    def available(self):
        return True

    def translate_many(self, segments, memory, *, source_segments=None):
        self.calls += 1
        return {segment.id: f"RU:{segment.text}" for segment in segments}, {}


def seg(sid: str, text: str) -> Segment:
    return Segment(id=sid, text=text, locator=f"/p/{sid}", chapter="Chapter")


def test_frozen_draft_records_then_replays_without_upstream_call(tmp_path: Path):
    path = tmp_path / "draft.json"
    rows = [seg("s1", "Hello"), seg("s2", "Number thirty")]

    Record = make_frozen_draft_backend(DummyBackend, path=path, mode="record")
    recorder = Record()
    recorded, errors = recorder.translate_many(rows, BookMemory(), source_segments=rows)
    assert errors == {}
    assert recorder.calls == 1
    assert recorded["s2"] == "RU:Number thirty"
    assert path.exists()

    Replay = make_frozen_draft_backend(DummyBackend, path=path, mode="replay")
    replay = Replay()
    replayed, replay_errors = replay.translate_many(rows, BookMemory(), source_segments=rows)
    assert replay_errors == {}
    assert replay.calls == 0
    assert replayed == recorded


def test_frozen_draft_rejects_source_mismatch(tmp_path: Path):
    path = tmp_path / "draft.json"
    original = [seg("s1", "Original")]
    Record = make_frozen_draft_backend(DummyBackend, path=path, mode="record")
    Record().translate_many(original, BookMemory(), source_segments=original)

    changed = [seg("s1", "Changed")]
    Replay = make_frozen_draft_backend(DummyBackend, path=path, mode="replay")
    rows, errors = Replay().translate_many(changed, BookMemory(), source_segments=changed)
    assert rows == {}
    assert errors["s1"] == "frozen draft source mismatch"
