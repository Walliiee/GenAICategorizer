"""Tests for the conversation categorization module."""

import pandas as pd
import pytest

from clustering import CATEGORIES, Categorizer, calculate_metrics


@pytest.fixture
def categorizer():
    """Create a Categorizer instance."""
    return Categorizer()


# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------


class TestCategorizer:
    """Test keyword-based categorization."""

    def test_code_development(self, categorizer):
        texts = [
            "I need help with my Python code. There is a bug in my function "
            "that throws an error."
        ]
        results = categorizer.assign_categories_batch(texts)
        assert results[0]["main_category"] == "Code Development"

    def test_cooking_food(self, categorizer):
        texts = [
            "What is a good recipe to cook for dinner? I want to bake a meal "
            "with fresh food."
        ]
        results = categorizer.assign_categories_batch(texts)
        assert results[0]["main_category"] == "Cooking/Food"

    def test_learning_education(self, categorizer):
        texts = [
            "Can you explain how machine learning works? I want to understand "
            "the tutorial."
        ]
        results = categorizer.assign_categories_batch(texts)
        assert results[0]["main_category"] == "Learning/Education"

    def test_uncategorizable_text(self, categorizer):
        texts = ["xyzzy plugh abcdef"]
        results = categorizer.assign_categories_batch(texts)
        assert results[0]["main_category"] == "Other"
        assert results[0]["subcategory"] == "Uncategorized"
        assert results[0]["confidence_score"] == 0

    def test_batch_returns_correct_count(self, categorizer):
        texts = ["Write Python code", "Bake a cake recipe", "Explain quantum physics"]
        assert len(categorizer.assign_categories_batch(texts)) == 3

    def test_confidence_score_positive_for_matched_text(self, categorizer):
        texts = ["code python function debug error program api"]
        results = categorizer.assign_categories_batch(texts)
        assert results[0]["confidence_score"] > 0

    def test_result_keys(self, categorizer):
        results = categorizer.assign_categories_batch(["test code"])
        expected = {
            "main_category",
            "subcategory",
            "is_voice",
            "confidence_score",
            "all_scores",
        }
        assert expected == set(results[0].keys())


# ---------------------------------------------------------------------------
# Voice & language detection
# ---------------------------------------------------------------------------


class TestVoiceDetection:
    """Test voice conversation detection."""

    def test_detects_audio_keyword(self, categorizer):
        assert categorizer.is_voice_conversation("This is an audio transcript")

    def test_non_voice_text(self, categorizer):
        assert not categorizer.is_voice_conversation("This is a regular text conversation")


class TestDanishDetection:
    """Test Danish language marker detection."""

    def test_detects_danish(self, categorizer):
        assert categorizer.is_danish_text("Hvordan kan jeg hjælp med denne opgave?")

    def test_non_danish(self, categorizer):
        assert not categorizer.is_danish_text("How can I help you with this task?")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Test metrics computation."""

    def test_metrics_structure(self):
        df = pd.DataFrame(
            {
                "text": ["Hello world", "Write some code for me"],
                "main_category": ["Information/Curiosity", "Code Development"],
                "subcategory": ["General Knowledge", "Feature Development"],
                "is_voice": [False, False],
                "confidence_score": [3, 5],
            }
        )
        metrics = calculate_metrics(df)
        assert metrics["total_conversations"] == 2
        assert "category_distribution" in metrics
        assert "complexity_scores" in metrics
        assert "confidence_metrics" in metrics


# ---------------------------------------------------------------------------
# Taxonomy integrity
# ---------------------------------------------------------------------------


class TestCategoryDefinitions:
    """Validate the category taxonomy."""

    def test_all_categories_have_subcategories(self):
        for name, info in CATEGORIES.items():
            assert "subcategories" in info, f"{name} missing subcategories"
            assert len(info["subcategories"]) > 0, f"{name} has empty subcategories"

    def test_all_categories_have_keywords(self):
        for name, info in CATEGORIES.items():
            assert "keywords" in info, f"{name} missing keywords"
            assert len(info["keywords"]) > 0, f"{name} has empty keywords"

    def test_expected_category_count(self):
        assert len(CATEGORIES) == 12
