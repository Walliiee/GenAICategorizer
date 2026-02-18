"""Shared test fixtures.

Provides a live FastAPI server fixture for Playwright UI tests and
sample data helpers used across test modules.

The server fixtures guard their imports so that the regular unit test
suite (``pip install -e ".[dev]"``) still works without web/UI deps.
"""

import json

import pytest

# ---------------------------------------------------------------------------
# Live server (only activated when UI tests request it)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _server_url():
    """Start the FastAPI app on a background thread and yield its base URL."""
    import threading
    import time

    import uvicorn

    from src.app import app

    host = "127.0.0.1"
    port = 8765

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("FastAPI server failed to start within 10 seconds")

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def live_server_url(_server_url):
    """Public fixture that returns the running server's base URL."""
    return _server_url


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_json_file(tmp_path):
    """Create a minimal Claude-style JSON file for upload testing."""
    conversations = [
        {
            "uuid": "conv-001",
            "title": "Python Help",
            "chat_messages": [
                {"text": "Can you help me debug my Python code? I have an error in my function."},
                {"text": "Sure! Please share the code and the error message."},
            ],
        },
        {
            "uuid": "conv-002",
            "title": "Recipe Request",
            "chat_messages": [
                {"text": "What is a good recipe for dinner? I want to cook pasta."},
                {"text": "Here is a simple pasta recipe with tomato sauce."},
            ],
        },
        {
            "uuid": "conv-003",
            "title": "Learning ML",
            "chat_messages": [
                {
                    "text": (
                        "Can you explain how machine learning works?"
                        " I want to understand the basics."
                    )
                },
                {"text": "Machine learning is a subset of AI that learns from data."},
            ],
        },
    ]

    path = tmp_path / "sample_conversations.json"
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path
