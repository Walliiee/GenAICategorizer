"""Tests for the data processing module."""

import pytest

from data_processing import DataProcessor


@pytest.fixture
def processor(tmp_path):
    """Create a DataProcessor with a temporary cache directory."""
    return DataProcessor(cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Text extraction — Claude format
# ---------------------------------------------------------------------------


class TestExtractTextClaude:
    """Test text extraction from Claude conversation exports."""

    def test_direct_text_field(self, processor):
        conv = {
            "chat_messages": [
                {"text": "Hello, how are you?"},
                {"text": "I'm doing well, thanks!"},
            ]
        }
        result = processor.extract_text(conv)
        assert "Hello" in result
        assert "doing well" in result

    def test_content_array_with_type(self, processor):
        conv = {
            "chat_messages": [
                {
                    "content": [
                        {"type": "text", "text": "Can you help me with Python?"}
                    ]
                }
            ]
        }
        result = processor.extract_text(conv)
        assert "Python" in result

    def test_content_array_plain_string(self, processor):
        conv = {
            "chat_messages": [
                {"content": ["Plain string content", {"type": "text", "text": "Dict content"}]}
            ]
        }
        result = processor.extract_text(conv)
        assert "Plain string" in result
        assert "Dict content" in result

    def test_empty_messages(self, processor):
        conv = {"chat_messages": []}
        assert processor.extract_text(conv) == ""

    def test_message_with_empty_text(self, processor):
        conv = {"chat_messages": [{"text": ""}]}
        assert processor.extract_text(conv) == ""


# ---------------------------------------------------------------------------
# Text extraction — ChatGPT format
# ---------------------------------------------------------------------------


class TestExtractTextChatGPT:
    """Test text extraction from ChatGPT conversation exports."""

    def test_basic_parts(self, processor):
        conv = {
            "mapping": {
                "msg1": {
                    "message": {
                        "content": {"parts": ["What is machine learning?"]}
                    }
                },
                "msg2": {
                    "message": {
                        "content": {"parts": ["It is a subset of AI."]}
                    }
                },
            }
        }
        result = processor.extract_text(conv)
        assert "machine learning" in result
        assert "subset of AI" in result

    def test_audio_transcription(self, processor):
        conv = {
            "mapping": {
                "msg1": {
                    "message": {
                        "content": {
                            "parts": [
                                {
                                    "content_type": "audio_transcription",
                                    "text": "Transcribed audio text",
                                }
                            ]
                        }
                    }
                }
            }
        }
        assert "Transcribed audio" in processor.extract_text(conv)

    def test_null_message_skipped(self, processor):
        conv = {"mapping": {"msg1": {"message": None}}}
        assert processor.extract_text(conv) == ""


# ---------------------------------------------------------------------------
# Fallback deep search
# ---------------------------------------------------------------------------


class TestLimitedDeepSearch:
    """Test the recursive fallback text search."""

    def test_finds_text_field(self, processor):
        obj = {"text": "hello world"}
        assert "hello world" in processor._limited_deep_search(obj)

    def test_respects_max_depth(self, processor):
        obj = {"a": {"b": {"c": {"text": "too deep"}}}}
        result = processor._limited_deep_search(obj, max_depth=2)
        assert "too deep" not in result

    def test_handles_list_input(self, processor):
        obj = [{"text": "item one"}, {"text": "item two"}]
        result = processor._limited_deep_search(obj)
        assert "item one" in result
        assert "item two" in result

    def test_empty_structures(self, processor):
        assert processor._limited_deep_search({}) == ""
        assert processor._limited_deep_search([]) == ""

    def test_unknown_format_uses_fallback(self, processor):
        conv = {"unknown_format": {"text": "Found via deep search"}}
        assert "Found via deep search" in processor.extract_text(conv)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestTextHash:
    """Test the text hashing utility."""

    def test_deterministic(self, processor):
        assert processor.get_text_hash("hello") == processor.get_text_hash("hello")

    def test_different_inputs(self, processor):
        assert processor.get_text_hash("hello") != processor.get_text_hash("world")
