from pathlib import Path

from bookai.parsers.fb2 import load_fb2, save_fb2


def test_fb2_roundtrip(tmp_path: Path):
    src = tmp_path / "book.fb2"
    src.write_text('<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body><section><p>Hello world.</p><p>Second line.</p></section></body></FictionBook>', "utf-8")
    doc = load_fb2(src)
    assert [s.text for s in doc.segments] == ["Hello world.", "Second line."]
    out = tmp_path / "ru.fb2"
    save_fb2(doc, {doc.segments[0].id: "Привет, мир.", doc.segments[1].id: "Вторая строка."}, out)
    text = out.read_text("utf-8")
    assert "Привет, мир." in text
    assert "data-bookai-id" not in text
