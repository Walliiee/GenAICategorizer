"""Centralized runtime configuration for GenAI Categorizer."""

from __future__ import annotations

import os


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


COMPLEXITY_SIMPLE_MAX = _get_env_int("GENAI_COMPLEXITY_SIMPLE_MAX", 100)
COMPLEXITY_MEDIUM_MAX = _get_env_int("GENAI_COMPLEXITY_MEDIUM_MAX", 500)

CATEGORY_CONFIDENCE_THRESHOLD = _get_env_float("GENAI_CONFIDENCE_THRESHOLD", 1.0)
KEYWORD_WEIGHT = _get_env_float("GENAI_KEYWORD_WEIGHT", 0.7)
SEMANTIC_WEIGHT = _get_env_float("GENAI_SEMANTIC_WEIGHT", 0.3)
SEMANTIC_MIN_SCORE = _get_env_float("GENAI_SEMANTIC_MIN_SCORE", 0.05)
HIGH_CONFIDENCE_THRESHOLD = _get_env_float("GENAI_HIGH_CONFIDENCE_THRESHOLD", 5.0)

APP_MAX_UPLOAD_FILES = _get_env_int("GENAI_MAX_UPLOAD_FILES", 20)
APP_MAX_FILE_BYTES = _get_env_int("GENAI_MAX_FILE_BYTES", 10 * 1024 * 1024)
APP_MAX_TOTAL_BYTES = _get_env_int("GENAI_MAX_TOTAL_BYTES", 50 * 1024 * 1024)

DEFAULT_BATCH_SIZE = _get_env_int("GENAI_BATCH_SIZE", 1000)


def complexity_label(length: int) -> str:
    """Map text length to the configured complexity bucket."""
    if length <= COMPLEXITY_SIMPLE_MAX:
        return "simple"
    if length <= COMPLEXITY_MEDIUM_MAX:
        return "medium"
    return "complex"
