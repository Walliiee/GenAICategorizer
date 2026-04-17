"""Unit tests for API extraction and upload robustness."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import ExtractionError, app


def _valid_json_bytes() -> bytes:
    payload = [
        {
            "uuid": "conv-1",
            "chat_messages": [{"text": "Please debug this Python code"}],
        }
    ]
    return json.dumps(payload).encode("utf-8")


def test_api_mixed_valid_and_invalid_files():
    client = TestClient(app)
    files = [
        ("files", ("valid.json", _valid_json_bytes(), "application/json")),
        ("files", ("invalid.json", b"{invalid", "application/json")),
        ("files", ("notes.txt", b"hello", "text/plain")),
    ]
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    body = response.json()
    assert len(body["conversations"]) >= 1
    assert len(body["file_reports"]) == 3
    statuses = {entry["filename"]: entry["status"] for entry in body["file_reports"]}
    assert statuses["valid.json"] == "ok"
    assert statuses["invalid.json"] == "error"
    assert statuses["notes.txt"] == "error"


def test_api_too_many_files_limit():
    from app import MAX_UPLOAD_FILES

    client = TestClient(app)
    files = [
        ("files", (f"{i}.json", _valid_json_bytes(), "application/json"))
        for i in range(MAX_UPLOAD_FILES + 1)
    ]
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 413


def test_extraction_error_class():
    err = ExtractionError("invalid_json", "Malformed JSON")
    assert err.code == "invalid_json"
    assert "Malformed JSON" in err.message
