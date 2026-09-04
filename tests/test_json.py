from bookai.llm import extract_json


def test_extract_json_from_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_from_wrapped_text():
    assert extract_json('result: {"a": 2} ok') == {"a": 2}
