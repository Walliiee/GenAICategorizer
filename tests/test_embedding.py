"""Tests for the embedding generation module.

These tests cover utility methods (chunking, caching, device detection)
without loading the actual SentenceTransformer model, so they run fast
and don't require a GPU.
"""

import numpy as np
import pytest

from embedding import EmbeddingGenerator


@pytest.fixture
def generator(tmp_path):
    """Create an EmbeddingGenerator with a temporary cache directory."""
    return EmbeddingGenerator(cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


class TestDeviceDetection:
    """Test compute device selection."""

    def test_returns_valid_device(self, generator):
        assert generator.device in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


class TestChunkTexts:
    """Test the text chunking utility."""

    def test_even_split(self, generator):
        texts = [f"text_{i}" for i in range(9)]
        chunks = generator._chunk_texts(texts, chunk_size=3)
        assert len(chunks) == 3
        assert all(len(c) == 3 for c in chunks)

    def test_remainder_chunk(self, generator):
        texts = [f"text_{i}" for i in range(10)]
        chunks = generator._chunk_texts(texts, chunk_size=3)
        assert len(chunks) == 4
        assert len(chunks[-1]) == 1

    def test_single_chunk_when_small(self, generator):
        chunks = generator._chunk_texts(["a", "b"], chunk_size=100)
        assert len(chunks) == 1

    def test_empty_list(self, generator):
        assert generator._chunk_texts([], chunk_size=5) == []


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Test deterministic cache key generation."""

    def test_deterministic(self, generator):
        texts = ["hello", "world"]
        assert generator._get_cache_key(texts) == generator._get_cache_key(texts)

    def test_varies_with_input(self, generator):
        assert generator._get_cache_key(["hello"]) != generator._get_cache_key(["world"])


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


class TestCacheRoundtrip:
    """Test saving and loading embeddings from disk cache."""

    def test_save_and_load(self, generator):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        generator._save_to_cache(data, "test_key")
        loaded = generator._load_cached("test_key")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, data)

    def test_missing_key_returns_none(self, generator):
        assert generator._load_cached("nonexistent") is None


# ---------------------------------------------------------------------------
# Batch size heuristic
# ---------------------------------------------------------------------------


class TestOptimalBatchSize:
    """Test the batch size heuristic."""

    def test_within_bounds(self, generator):
        bs = generator._get_optimal_batch_size(1000)
        assert 1 <= bs <= 64

    def test_small_input(self, generator):
        bs = generator._get_optimal_batch_size(5)
        assert bs >= 1
