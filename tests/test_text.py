from pathlib import Path

from bookai.parsers.text import load_txt, save_txt


def test_txt_roundtrip(tmp_path: Path):
    src = tmp_path / "book.txt"
    src.write_text("Chapter 1\n\nHello.\nWorld.\n", "utf-8")
    doc = load_txt(src)
    assert doc.segments[1].chapter == "Chapter 1"
    out = tmp_path / "ru.txt"
    save_txt(doc, {s.id: f"RU:{s.text}" for s in doc.segments}, out)
    text = out.read_text("utf-8")
    assert "RU:Hello." in text
    assert "\n\n" in text
