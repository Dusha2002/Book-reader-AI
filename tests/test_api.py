from fastapi.testclient import TestClient

from bookai.api import app


def test_web_ui_and_health():
    client = TestClient(app)
    assert client.get('/health').json()['version'] == '0.3.0'
    page = client.get('/')
    assert page.status_code == 200
    assert 'Book Reader AI' in page.text
    assert 'Qwen3.8 Flash' in page.text
    assert 'DOCX' in page.text
