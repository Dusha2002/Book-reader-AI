from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from lxml import etree

from ..models import BookDocument, Segment

_TEXT_TAGS = {"p", "subtitle", "text-author", "date", "v"}
_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"


def _local(tag: str) -> str:
    return etree.QName(tag).localname


def _chapter_for(node: etree._Element, tree: etree._ElementTree) -> str:
    parent = node
    while parent is not None:
        if isinstance(parent.tag, str) and _local(parent.tag) == "section":
            for child in parent:
                if isinstance(child.tag, str) and _local(child.tag) == "title":
                    title = " ".join("".join(child.itertext()).split())
                    if title:
                        return title[:200]
            return tree.getpath(parent)
        parent = parent.getparent()
    return "book"


def load_fb2(path: Path) -> BookDocument:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True)
    tree = etree.parse(str(path), parser)
    segments: list[Segment] = []
    idx = 0
    for node in tree.iter():
        if not isinstance(node.tag, str) or _local(node.tag) not in _TEXT_TAGS:
            continue
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        sid = f"s{idx:06d}"
        segments.append(Segment(id=sid, text=text, locator=tree.getpath(node), chapter=_chapter_for(node, tree)))
        node.set("data-bookai-id", sid)
        idx += 1
    return BookDocument(source=path, format="fb2", segments=segments, payload=tree)


def _replace_text_preserving_inline(node: etree._Element, text: str) -> None:
    # The current segment contract translates a whole FB2 paragraph as one unit.
    # Keep inline element shells/attributes instead of deleting them; their textual
    # content is cleared so English fragments cannot survive inside a translated p.
    node.text = text
    for child in node:
        child.text = None
        child.tail = None


def _set_translation_language(tree: etree._ElementTree) -> None:
    root = tree.getroot()
    ns = root.nsmap.get(None) or _FB2_NS
    title_infos = root.xpath("//*[local-name()='description']/*[local-name()='title-info']")
    for title_info in title_infos:
        lang = next((c for c in title_info if isinstance(c.tag, str) and _local(c.tag) == "lang"), None)
        old_lang = (lang.text or "").strip() if lang is not None else ""
        if lang is None:
            lang = etree.SubElement(title_info, f"{{{ns}}}lang")
        lang.text = "ru"
        src_lang = next((c for c in title_info if isinstance(c.tag, str) and _local(c.tag) == "src-lang"), None)
        if src_lang is None and old_lang and old_lang.lower() != "ru":
            src_lang = etree.SubElement(title_info, f"{{{ns}}}src-lang")
            src_lang.text = old_lang


def save_fb2(document: BookDocument, translations: dict[str, str], output: Path) -> None:
    tree = deepcopy(document.payload)
    assert isinstance(tree, etree._ElementTree)
    for node in tree.iter():
        sid = node.attrib.pop("data-bookai-id", None)
        if sid and sid in translations:
            _replace_text_preserving_inline(node, translations[sid])
    _set_translation_language(tree)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output), encoding="utf-8", xml_declaration=True, pretty_print=False)
