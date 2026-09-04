import json
import re
from pathlib import Path

from bookai.pipeline import translate_book


class FakeProvider:
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        if "translation bible" in system.lower():
            return json.dumps({"style": {}, "glossary": {}, "characters": {}, "rolling_summary": ""})
        ids = sorted(set(re.findall(r'"(s\d{6})"', user)))
        return json.dumps({sid: f"RU-{sid}" for sid in ids})


def test_pipeline_fast(tmp_path: Path):
    src = tmp_path / "book.fb2"
    src.write_text('<?xml version="1.0"?><FictionBook><body><section><p>Hello.</p></section></body></FictionBook>', "utf-8")
    out = tmp_path / "out.fb2"
    translate_book(src, out, FakeProvider(), mode="fast", cache_dir=tmp_path / "cache")
    assert "RU-s000000" in out.read_text("utf-8")
