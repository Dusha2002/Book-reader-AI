from fastapi.testclient import TestClient

from bookai.api import JOBS, LOCK, app


def test_web_ui_and_health():
    client = TestClient(app)
    health = client.get('/health').json()
    assert health['version'] == '0.4.0'
    assert health['progressive_translation'] is True
    page = client.get('/')
    assert page.status_code == 200
    assert 'Book Reader AI' in page.text
    assert 'GigaChat-3-Lightning' in page.text
    assert 'DeepSeek V4.1 Flash' in page.text
    assert 'DOCX' in page.text


def test_progressive_segment_delta_contract():
    client = TestClient(app)
    job_id = 'progressive-test'
    with LOCK:
        JOBS[job_id] = {
            'status': 'running',
            'revision': 2,
            'ready_segments': 3,
            'segment_events': [
                {'revision': 1, 'segments': {'s000001': 'Первый', 's000002': 'Второй'}},
                {'revision': 2, 'segments': {'s000002': 'Второй, исправленный', 's000003': 'Третий'}},
            ],
            'translated_segments': {
                's000001': 'Первый',
                's000002': 'Второй, исправленный',
                's000003': 'Третий',
            },
        }
    try:
        response = client.get(f'/jobs/{job_id}/segments?after=1')
        assert response.status_code == 200
        payload = response.json()
        assert payload['revision'] == 2
        assert payload['ready_segments'] == 3
        assert payload['segments'] == {
            's000002': 'Второй, исправленный',
            's000003': 'Третий',
        }
        assert payload['done'] is False

        status = client.get(f'/jobs/{job_id}').json()
        assert 'segment_events' not in status
        assert 'translated_segments' not in status
    finally:
        with LOCK:
            JOBS.pop(job_id, None)
