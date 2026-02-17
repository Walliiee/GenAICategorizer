"""Conversation categorization module.

Assigns each conversation a primary topic category and subcategory using
compiled regex keyword patterns. Supports batch processing, confidence
scoring, and voice-conversation detection across 12 topic categories.
"""

import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

CATEGORIES: Dict[str, Dict] = {
    "Learning/Education": {
        "keywords": [
            "explain", "teach", "learn", "understand", "what is", "how does",
            "define", "example", "tutorial", "forklar", "forklaring", "forstå",
            "lær", "hvordan",
        ],
        "subcategories": {
            "Concept Understanding": ["what is", "define", "concept", "mean", "forklar", "hvad er"],
            "How-to Learning": ["how to", "steps", "guide", "tutorial", "hvordan"],
            "Problem Solving": ["solve", "solution", "answer", "help me understand"],
            "Academic Topics": ["method", "theory", "model", "metode", "teori", "analyse"],
        },
    },
    "Code Development": {
        "keywords": [
            "code", "program", "function", "debug", "error", "python",
            "javascript", "api", "database", "git",
        ],
        "subcategories": {
            "Bug Fixing": ["error", "bug", "fix", "issue", "debug"],
            "Feature Development": ["create", "implement", "develop", "add feature"],
            "Code Review": ["review", "improve", "optimize", "refactor"],
        },
    },
    "Writing Assistance": {
        "keywords": [
            "write", "draft", "review", "proofread", "text", "article", "edit", "grammar",
        ],
        "subcategories": {
            "Content Creation": ["write", "create", "draft"],
            "Editing": ["edit", "proofread", "review", "improve"],
            "Format/Style": ["format", "style", "structure"],
        },
    },
    "Analysis/Research": {
        "keywords": [
            "analyze", "research", "study", "investigate", "compare", "evaluate", "data",
            "findings",
        ],
        "subcategories": {
            "Data Analysis": ["data", "statistics", "numbers", "trends"],
            "Research Review": ["research", "paper", "study", "literature"],
            "Comparative Analysis": ["compare", "difference", "versus", "pros and cons"],
        },
    },
    "Creative & Ideation": {
        "keywords": [
            "create", "design", "generate", "brainstorm", "imagine", "creative",
            "innovative", "idea", "suggest", "think of", "come up with",
            "possibilities", "concept",
        ],
        "subcategories": {
            "Visual Design": ["design", "visual", "layout", "look"],
            "Idea Generation": ["brainstorm", "ideas", "possibilities", "suggest"],
            "Creative Problem Solving": ["solution", "solve", "address", "improve"],
            "Innovation": ["innovative", "new", "unique", "original"],
            "Concept Development": ["develop", "refine", "enhance", "iterate"],
        },
    },
    "Professional/Business": {
        "keywords": [
            "business", "professional", "company", "client", "strategy", "market", "industry",
        ],
        "subcategories": {
            "Strategy": ["strategy", "plan", "approach"],
            "Client/Customer": ["client", "customer", "service"],
            "Business Analysis": ["analysis", "market", "industry"],
        },
    },
    "Technical Support": {
        "keywords": [
            "help", "fix", "issue", "problem", "support", "error", "troubleshoot",
            "sync", "synch", "calendar", "setup", "connect", "integration",
            "configure", "settings", "apple", "microsoft", "teams", "outlook",
        ],
        "subcategories": {
            "Troubleshooting": ["troubleshoot", "diagnose", "fix", "issue", "error"],
            "Setup/Installation": ["setup", "install", "configure", "connect", "sync"],
            "Usage Help": ["how to use", "help with", "guide", "how do i"],
            "Integration Issues": ["sync", "connect", "integration", "between", "with"],
        },
    },
    "Personal Projects": {
        "keywords": ["project", "personal", "help me with", "my own", "portfolio", "hobby"],
        "subcategories": {
            "Project Planning": ["plan", "structure", "organize"],
            "Implementation": ["build", "create", "develop"],
            "Review/Feedback": ["review", "feedback", "improve"],
        },
    },
    "SoMe/Marketing": {
        "keywords": [
            "social media", "marketing", "post", "content", "campaign", "LinkedIn",
            "Twitter", "engagement", "opslag", "sæson", "hold", "announce",
            "announcement", "season", "team", "group", "start",
        ],
        "subcategories": {
            "Content Creation": ["post", "content", "create", "opslag", "write post"],
            "Event Announcements": [
                "season", "sæson", "event", "start", "announce", "new",
            ],
            "Team/Group Updates": ["team", "hold", "group", "members", "community"],
            "Campaign Planning": ["campaign", "strategy", "plan", "theme", "tema"],
        },
    },
    "DALL-E/Image": {
        "keywords": [
            "image", "picture", "photo", "generate image", "create image",
            "dall-e", "dalle", "draw", "illustration", "visual", "artwork",
        ],
        "subcategories": {
            "Image Generation": ["generate", "create", "make"],
            "Image Editing": ["edit", "modify", "adjust"],
            "Style Transfer": ["style", "artistic", "filter"],
            "Visual Description": ["describe", "detail", "explain image"],
        },
    },
    "Cooking/Food": {
        "keywords": [
            "food", "cook", "recipe", "ingredient", "meal", "dish", "kitchen",
            "bake", "pumpkin", "vegetable", "fruit", "squash", "cuisine", "eat",
            "dinner", "lunch", "breakfast",
        ],
        "subcategories": {
            "Recipe Help": ["recipe", "how to make", "cook", "bake", "prepare"],
            "Ingredient Questions": ["ingredient", "what is", "substitute", "alternative"],
            "Food Identification": [
                "what is this", "identify", "which", "type of", "variety",
            ],
            "Meal Planning": ["meal", "plan", "menu", "dinner", "lunch", "breakfast"],
        },
    },
    "Information/Curiosity": {
        "keywords": [
            "what is", "where is", "why does", "how does", "tell me about",
            "find information", "information about", "can you find", "explain why",
            "hvorfor", "hvor", "hvad", "find ud af", "finde information",
            "information om",
        ],
        "subcategories": {
            "General Knowledge": ["what is", "tell me about", "explain", "fortæl"],
            "Location/Place Questions": ["where is", "location", "place", "hvor"],
            "Cause/Effect": ["why does", "how does", "what causes", "hvorfor"],
            "Research Requests": [
                "find information", "research", "look up", "find ud af", "søg",
            ],
            "Industry/Topic Research": [
                "industry", "sector", "field", "branche", "område",
                "digitalization", "digitalisering",
            ],
        },
    },
}

VOICE_KEYWORDS = [
    "transcript", "audio", "voice", "speech", "spoken", "recording", "sound",
]

DANISH_MARKERS = [
    "hvordan", "hvad", "hvilken", "hvor", "hvem", "hvorfor", "skriv",
    "hjælp", "tak", "opslag", "sæson", "hold", "med", "og", "eller",
    "på", "jeg", "du", "vi", "denne",
]


# ---------------------------------------------------------------------------
# Categorizer
# ---------------------------------------------------------------------------


class Categorizer:
    """Keyword-based conversation categorizer with compiled regex patterns.

    Pre-compiles all keyword patterns at initialization for fast repeated
    matching across large conversation sets.
    """

    def __init__(self) -> None:
        self.categories = CATEGORIES
        self.compiled_patterns = self._compile_patterns()
        self.voice_pattern = self._compile_voice_pattern()
        self.danish_pattern = self._compile_danish_pattern()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    # -- Pattern compilation ------------------------------------------------

    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for every category and subcategory."""
        compiled: Dict = {}
        for category, info in self.categories.items():
            main_alt = "|".join(re.escape(kw) for kw in info["keywords"])
            compiled[category] = {
                "main_pattern": re.compile(rf"\b(?:{main_alt})\b", re.IGNORECASE),
                "subcategories": {},
            }
            for sub_name, sub_keywords in info["subcategories"].items():
                sub_alt = "|".join(re.escape(kw) for kw in sub_keywords)
                compiled[category]["subcategories"][sub_name] = re.compile(
                    rf"\b(?:{sub_alt})\b", re.IGNORECASE
                )
        return compiled

    @staticmethod
    def _compile_voice_pattern() -> re.Pattern:
        """Compile voice-conversation detection pattern."""
        alt = "|".join(re.escape(kw) for kw in VOICE_KEYWORDS)
        return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

    @staticmethod
    def _compile_danish_pattern() -> re.Pattern:
        """Compile Danish language marker pattern."""
        alt = "|".join(re.escape(m) for m in DANISH_MARKERS)
        return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

    # -- Detection helpers --------------------------------------------------

    @lru_cache(maxsize=10_000)
    def is_danish_text(self, text: str) -> bool:
        """Return ``True`` if the text contains two or more Danish markers."""
        return len(self.danish_pattern.findall(text)) >= 2

    @lru_cache(maxsize=10_000)
    def is_voice_conversation(self, text: str) -> bool:
        """Return ``True`` if the text contains voice/audio keywords."""
        return bool(self.voice_pattern.search(text))

    # -- Scoring & assignment -----------------------------------------------

    def _score_texts(self, texts: List[str]) -> List[Dict]:
        """Score every text against all categories."""
        results = []
        for text in texts:
            text_lower = text.lower()
            scores: Dict = {}
            for category, patterns in self.compiled_patterns.items():
                main_hits = len(patterns["main_pattern"].findall(text_lower))
                sub_scores: Dict[str, int] = {}
                sub_total = 0
                for sub_name, sub_pat in patterns["subcategories"].items():
                    n = len(sub_pat.findall(text_lower))
                    sub_scores[sub_name] = n
                    sub_total += n
                scores[category] = {
                    "main_score": main_hits,
                    "sub_scores": sub_scores,
                    "total_score": main_hits + sub_total,
                }
            results.append(scores)
        return results

    def assign_categories_batch(self, texts: List[str]) -> List[Dict]:
        """Categorize a batch of texts and return per-text results.

        Each result dict contains ``main_category``, ``subcategory``,
        ``is_voice``, ``confidence_score``, and the full ``all_scores``.
        """
        all_scores = self._score_texts(texts)
        results: List[Dict] = []

        for text, scores in zip(texts, all_scores):
            best = max(scores.items(), key=lambda x: x[1]["total_score"])
            main_category = best[0]
            sub_scores = best[1]["sub_scores"]
            subcategory = (
                max(sub_scores, key=sub_scores.get) if sub_scores else "Uncategorized"
            )

            if best[1]["total_score"] == 0:
                main_category = "Other"
                subcategory = "Uncategorized"

            results.append(
                {
                    "main_category": main_category,
                    "subcategory": subcategory,
                    "is_voice": self.is_voice_conversation(text),
                    "confidence_score": best[1]["total_score"],
                    "all_scores": scores,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Compute summary metrics from a categorized DataFrame."""
    text_lengths = df["text"].str.len()
    complexity = pd.cut(
        text_lengths,
        bins=[0, 100, 500, float("inf")],
        labels=["simple", "medium", "complex"],
    )

    return {
        "total_conversations": int(len(df)),
        "avg_length": float(text_lengths.mean()),
        "category_distribution": {
            k: int(v) for k, v in df["main_category"].value_counts().items()
        },
        "subcategory_distribution": {
            k: int(v) for k, v in df["subcategory"].value_counts().items()
        },
        "voice_conversations": (
            int(df["is_voice"].sum()) if "is_voice" in df.columns else 0
        ),
        "complexity_scores": {k: int(v) for k, v in complexity.value_counts().items()},
        "interaction_patterns": {
            "questions": int(df["text"].str.count(r"\?").sum()),
            "follow_ups": int(
                df["text"].str.count(r"(?i)follow up|followup|additional").sum()
            ),
        },
        "confidence_metrics": {
            "avg_confidence": (
                float(df["confidence_score"].mean())
                if "confidence_score" in df.columns
                else 0
            ),
            "high_confidence": (
                int((df["confidence_score"] > 5).sum())
                if "confidence_score" in df.columns
                else 0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_clustering(
    cleaned_csv: str, output_csv: str, batch_size: int = 1000
) -> None:
    """Categorize conversations in *cleaned_csv* and write results to *output_csv*."""
    start = time.time()
    categorizer = Categorizer()

    categorizer.logger.info("Loading conversation data...")
    df = pd.read_csv(cleaned_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    total = len(df)
    categorizer.logger.info(
        "Processing %d conversations (batch_size=%d)", total, batch_size
    )

    all_results: List[Dict] = []
    for i in range(0, total, batch_size):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()
        all_results.extend(categorizer.assign_categories_batch(batch_texts))
        categorizer.logger.info(
            "Batch %d/%d complete",
            i // batch_size + 1,
            (total - 1) // batch_size + 1,
        )

    result_df = pd.DataFrame(all_results)
    for col in ("main_category", "subcategory", "is_voice", "confidence_score"):
        if col in result_df.columns:
            df[col] = result_df[col].values

    df["char_length"] = df["text"].str.len()
    df["complexity"] = pd.cut(
        df["char_length"],
        bins=[0, 100, 500, float("inf")],
        labels=["simple", "medium", "complex"],
    )

    metrics = calculate_metrics(df)
    elapsed = time.time() - start
    metrics["performance"] = {
        "processing_time_seconds": round(elapsed, 2),
        "conversations_per_second": round(total / elapsed, 1) if elapsed > 0 else 0,
        "batch_size_used": batch_size,
    }

    metrics_file = os.path.join(
        os.path.dirname(output_csv), "conversation_metrics.json"
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4, default=str)

    df.to_csv(output_csv, index=False)
    categorizer.logger.info("Done in %.2fs — %s", elapsed, output_csv)
    _print_summary(metrics)


def _print_summary(metrics: Dict) -> None:
    """Print a concise categorization summary to stdout."""
    print("\n" + "=" * 50)
    print("CATEGORIZATION SUMMARY")
    print("=" * 50)
    print(f"Total conversations: {metrics['total_conversations']}")

    perf = metrics.get("performance", {})
    if perf:
        print(f"Processing time:     {perf['processing_time_seconds']}s")
        print(f"Throughput:          {perf['conversations_per_second']} conv/s")

    print("\nTop Categories:")
    for cat, count in list(metrics["category_distribution"].items())[:10]:
        pct = count / metrics["total_conversations"] * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"\nVoice conversations: {metrics['voice_conversations']}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    in_csv = str(project_root / "data" / "processed" / "cleaned_conversations.csv")
    out_csv = str(project_root / "data" / "processed" / "categorized_conversations.csv")

    run_clustering(in_csv, out_csv)
