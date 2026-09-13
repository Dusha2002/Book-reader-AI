import json

import pytest

from bookai.json_salvage import parse_json_with_item_salvage


def test_parses_normal_object():
    assert parse_json_with_item_salvage('{"items":[{"id":"s1","corrected_ru":"Да"}]}')["items"][0]["id"] == "s1"


def test_strips_markdown_fence_and_trailing_comma():
    raw = '```json\n{"items":[{"id":"s1","corrected_ru":"Да",}],}\n```'
    obj = parse_json_with_item_salvage(raw)
    assert obj["items"][0]["corrected_ru"] == "Да"


def test_salvages_complete_items_from_truncated_array():
    raw = '{"items":[{"id":"s1","corrected_ru":"Первый"},{"id":"s2","corrected_ru":"Второй"},{"id":"s3","corrected_ru":"оборван'
    obj = parse_json_with_item_salvage(raw)
    assert obj.get("_salvaged_partial") is True
    assert [row["id"] for row in obj["items"]] == ["s1", "s2"]


def test_raises_when_nothing_recoverable():
    with pytest.raises(json.JSONDecodeError):
        parse_json_with_item_salvage("not json at all")
