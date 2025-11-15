import os
from web import app as flask_app

def test_index_page():
    client = flask_app.test_client()
    r = client.get('/')
    assert r.status_code == 200

def test_api_predict_no_file():
    client = flask_app.test_client()
    r = client.post('/api/predict')
    assert r.status_code == 400
    assert b'no file' in r.data
