"""Human-readable summaries for pipeline and evaluation outputs."""

from __future__ import annotations

from typing import Dict


def summarize_metrics(metrics: Dict) -> str:
    """Return a concise plain-text summary for categorization metrics."""
    if not metrics:
        return "No metrics available."

    lines = [
        f"Total conversations: {metrics.get('total_conversations', 0)}",
        f"Average length: {metrics.get('avg_length', 0)}",
        f"Voice conversations: {metrics.get('voice_conversations', 0)}",
    ]

    category_distribution = metrics.get("category_distribution", {})
    if category_distribution:
        top_category, top_count = next(iter(category_distribution.items()))
        lines.append(f"Top category: {top_category} ({top_count})")

    confidence_metrics = metrics.get("confidence_metrics", {})
    if confidence_metrics:
        lines.append(
            "Average confidence: "
            f"{round(confidence_metrics.get('avg_confidence', 0.0), 2)}"
        )

    return "\n".join(lines)


def summarize_evaluation(report: Dict) -> str:
    """Return a concise plain-text summary for evaluation metrics."""
    if not report:
        return "No evaluation report available."

    lines = [
        f"Samples: {report.get('samples', 0)}",
        f"Macro precision: {round(report.get('macro_precision', 0.0), 3)}",
        f"Macro recall: {round(report.get('macro_recall', 0.0), 3)}",
        f"Macro F1: {round(report.get('macro_f1', 0.0), 3)}",
    ]
    return "\n".join(lines)
