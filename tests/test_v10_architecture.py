from bookai.models import BookMemory, Segment
from bookai.v10 import DeterministicQA, GigaPrimaryTransport
from bookai.v10_bible import AtomicBookBibleBuilder
from bookai.v10_transport import RobustTaggedPrimaryTransport


def seg(text: str, sid: str = "s000001") -> Segment:
    return Segment(id=sid, text=text, locator="/x", chapter="Chapter One")


def test_tagged_transport_parser_recovers_all_ids():
    text = '<s id="s000001">Первый перевод.</s>\n<s id="s000002">Второй\nперевод.</s>'
    rows = RobustTaggedPrimaryTransport.parse_tagged(text, {"s000001", "s000002"})
    assert rows == {"s000001": "Первый перевод.", "s000002": "Второй\nперевод."}


def test_tagged_transport_salvages_unclosed_block_at_next_opening_boundary():
    # The next opening tag is a safe boundary. s000001 may be salvaged without
    # swallowing s000002 even though the first closing tag is missing.
    text = '<s id="s000001">Первый перевод.\n<s id="s000002">Второй перевод.</s>'
    rows = RobustTaggedPrimaryTransport.parse_tagged(text, {"s000001", "s000002"})
    assert rows["s000001"] == "Первый перевод."
    assert rows["s000002"] == "Второй перевод."


def test_tagged_transport_does_not_salvage_final_unclosed_block():
    text = '<s id="s000001">Первый перевод без конца'
    rows = RobustTaggedPrimaryTransport.parse_tagged(text, {"s000001"})
    assert "s000001" not in rows


def test_dozen_cannot_be_half_dozen():
    qa = DeterministicQA()
    issues = qa.scan_segment(seg("A dozen imaginary goblins appeared."), "Появилось полдюжины воображаемых гоблинов.", BookMemory())
    assert any(i.code == "quantity_dozen" and i.severity == "hard" for i in issues)


def test_quarter_inch_requires_quarter_or_635mm():
    qa = DeterministicQA()
    bad = qa.scan_segment(seg("He used a quarter-inch plate."), "Он взял четырехмиллиметровую пластину.", BookMemory())
    good = qa.scan_segment(seg("He used a quarter-inch plate."), "Он взял пластину толщиной в четверть дюйма.", BookMemory())
    assert any(i.code == "quarter_inch" for i in bad)
    assert not any(i.code == "quarter_inch" for i in good)


def test_medium_multibeat_omission_routes_semantic():
    qa = DeterministicQA()
    source = (
        "He stopped and looked back. He had lost track of time completely. "
        "Then he made himself look at the problem from the other side. "
        "It was, he decided, a sort of holiday after all."
    )
    target = "Он остановился и оглянулся. В конце концов это был своего рода отпуск."
    issues = qa.scan_segment(seg(source), target, BookMemory())
    assert any(i.code == "omission" and i.mode == "semantic" for i in issues)


def test_actor_relation_routes_semantic():
    qa = DeterministicQA()
    issues = qa.scan_segment(seg("Did he invite you?"), "А тебя приглашали?", BookMemory())
    assert any(i.code == "actor_relation" and i.mode == "semantic" for i in issues)


def test_book_bible_candidates_scan_across_segments():
    rows = [
        seg("Ziani spoke to Miel about the treadle saw.", "s000001"),
        seg("Much later Miel asked Ziani about the treadle saw again.", "s000002"),
    ]
    candidates = AtomicBookBibleBuilder._candidate_records(rows)
    values = {row["candidate"].casefold() for row in candidates}
    assert "ziani" in values
    assert "treadle saw" in values
