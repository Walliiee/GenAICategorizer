"""Data processing module for AI conversation exports.

Handles JSON parsing, text extraction, and language detection for
conversation data exported from Claude and ChatGPT. Supports parallel
file processing and persistent language detection caching.
"""

import concurrent.futures
import csv
import glob
import hashlib
import json
import logging
import os
import pickle
from collections import Counter
from datetime import datetime
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Use faster JSON library when available
try:
    import orjson

    JSON_LOADS = orjson.loads
except ImportError:
    JSON_LOADS = json.loads

try:
    from langdetect import detect

    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


class DataProcessor:
    """Process raw conversation JSON exports into structured CSV data.

    Supports Claude and ChatGPT export formats with parallel file processing,
    language detection caching, and configurable worker counts.

    Args:
        cache_dir: Directory for persistent caches (language detection, etc.).
        max_workers: Maximum parallel workers for file I/O. Defaults to
            ``min(cpu_count(), 8)``.
    """

    def __init__(self, cache_dir: str = "../data/cache", max_workers: Optional[int] = None):
        self.cache_dir = cache_dir
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.language_cache: Dict[str, str] = {}
        self._setup_cache()
        self._setup_logging()

    def _setup_cache(self) -> None:
        """Create cache directory and load any existing caches."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_language_cache()

    def _setup_logging(self) -> None:
        """Configure module-level logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _load_language_cache(self) -> None:
        """Load persistent language detection cache from disk."""
        cache_file = os.path.join(self.cache_dir, "language_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.language_cache = pickle.load(f)
                logging.getLogger(__name__).info(
                    "Loaded %d cached language detections", len(self.language_cache)
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to load language cache: %s", exc
                )
                self.language_cache = {}

    def _save_language_cache(self) -> None:
        """Persist language detection cache to disk."""
        cache_file = os.path.join(self.cache_dir, "language_cache.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.language_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self.logger.warning("Failed to save language cache: %s", exc)

    @lru_cache(maxsize=10000)
    def detect_language_cached(self, text_hash: str, text: str) -> str:
        """Detect language with LRU + persistent caching.

        Args:
            text_hash: Pre-computed hash of the text for cache lookup.
            text: The text to detect language for.

        Returns:
            ISO 639-1 language code, or ``'unknown'`` on failure.
        """
        if not HAS_LANGDETECT:
            return "unknown"

        if text_hash in self.language_cache:
            return self.language_cache[text_hash]

        try:
            limited_text = text[:1000] if len(text) > 1000 else text
            lang = detect(limited_text)
            self.language_cache[text_hash] = lang
            return lang
        except Exception:
            self.language_cache[text_hash] = "unknown"
            return "unknown"

    @staticmethod
    def get_text_hash(text: str) -> str:
        """Generate a short MD5 hash for cache keying."""
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def extract_text(self, conversation: Dict) -> str:
        """Extract concatenated text from a conversation object.

        Handles Claude (``chat_messages``) and ChatGPT (``mapping``) formats,
        falling back to a limited-depth recursive search for unknown structures.
        """
        texts: List[str] = []

        # Claude format
        if "chat_messages" in conversation:
            for msg in conversation.get("chat_messages", []):
                if msg.get("text"):
                    texts.append(msg["text"].strip())
                    continue

                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                texts.append(item["text"].strip())
                            elif item.get("text"):
                                texts.append(item["text"].strip())
                        elif isinstance(item, str) and item.strip():
                            texts.append(item.strip())
            return " ".join(texts)

        # ChatGPT format
        if "mapping" in conversation:
            for msg_data in conversation.get("mapping", {}).values():
                message = msg_data.get("message")
                if not message:
                    continue

                content = message.get("content", {})
                if isinstance(content, dict) and "parts" in content:
                    for part in content["parts"]:
                        if isinstance(part, dict):
                            if (
                                part.get("content_type") == "audio_transcription"
                                and part.get("text")
                            ):
                                texts.append(part["text"].strip())
                        elif isinstance(part, str) and part.strip():
                            texts.append(part.strip())
            return " ".join(texts)

        # Fallback: limited-depth recursive search
        return self._limited_deep_search(conversation, max_depth=3)

    def _limited_deep_search(
        self, obj: object, max_depth: int = 3, current_depth: int = 0
    ) -> str:
        """Recursively search a JSON-like structure for text fields.

        Stops at *max_depth* and caps collected fragments to prevent
        runaway traversal on deeply nested data.
        """
        if current_depth >= max_depth:
            return ""

        texts: List[str] = []

        if isinstance(obj, dict):
            for field in ("text", "content", "message"):
                if field in obj and isinstance(obj[field], str) and obj[field].strip():
                    texts.append(obj[field].strip())
            for value in list(obj.values())[:10]:
                result = self._limited_deep_search(value, max_depth, current_depth + 1)
                if result:
                    texts.append(result)
                    if len(texts) > 5:
                        break

        elif isinstance(obj, list):
            for item in obj[:20]:
                result = self._limited_deep_search(item, max_depth, current_depth + 1)
                if result:
                    texts.append(result)
                    if len(texts) > 5:
                        break

        return " ".join(texts)


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def process_single_file(file_path: str) -> Tuple[List[Dict], int, int]:
    """Process one JSON file and return extracted conversations.

    Returns:
        Tuple of (rows, total_conversation_count, empty_conversation_count).
    """
    processor = DataProcessor()
    rows: List[Dict] = []
    empty_count = 0
    total_conversations = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = JSON_LOADS(f.read())

        conversations = (
            data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        )

        for conv in conversations:
            total_conversations += 1
            conv_id = conv.get(
                "uuid",
                conv.get("title", f"conv_{total_conversations}_{os.path.basename(file_path)}"),
            )
            conversation_text = processor.extract_text(conv)

            if conversation_text.strip():
                text_hash = processor.get_text_hash(conversation_text)
                lang = processor.detect_language_cached(text_hash, conversation_text)
                rows.append(
                    {
                        "conversation_id": conv_id,
                        "text": conversation_text,
                        "language": lang,
                        "source_file": os.path.basename(file_path),
                    }
                )
            else:
                empty_count += 1

    except Exception as exc:
        logging.error("Error processing %s: %s", file_path, exc)

    return rows, total_conversations, empty_count


def process_raw_files(
    raw_dir: str, output_csv: str, max_workers: Optional[int] = None
) -> None:
    """Process all JSON conversation files in *raw_dir* and write a cleaned CSV.

    Uses parallel I/O, language detection caching, and batch CSV writes.
    """
    processor = DataProcessor(max_workers=max_workers)
    file_paths = glob.glob(os.path.join(raw_dir, "*.json"))

    if not file_paths:
        processor.logger.warning("No JSON files found in %s", raw_dir)
        return

    processor.logger.info("Found %d JSON files in %s", len(file_paths), raw_dir)

    all_rows: List[Dict] = []
    total_conversations = 0
    total_empty = 0
    language_counts: Counter = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, fp): fp for fp in file_paths
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(file_paths),
            desc="Processing files",
        ):
            file_path = future_to_file[future]
            try:
                rows, file_total, file_empty = future.result()
                all_rows.extend(rows)
                total_conversations += file_total
                total_empty += file_empty
                for row in rows:
                    language_counts[row["language"]] += 1
            except Exception as exc:
                processor.logger.error("Error processing %s: %s", file_path, exc)

    processor._save_language_cache()

    # Log statistics
    processor.logger.info("Language Statistics:")
    for lang, count in language_counts.most_common():
        processor.logger.info("  %s: %d conversations", lang, count)

    processor.logger.info("Total conversations processed: %d", total_conversations)
    processor.logger.info("Empty conversations discarded: %d", total_empty)
    processor.logger.info("Non-empty conversations saved: %d", len(all_rows))
    if total_conversations > 0:
        processor.logger.info(
            "Success rate: %.2f%%", len(all_rows) / total_conversations * 100
        )

    # Write output
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = ["conversation_id", "text", "language", "source_file"]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        batch_size = 1000
        for i in range(0, len(all_rows), batch_size):
            writer.writerows(all_rows[i : i + batch_size])

    processor.logger.info("Processed data saved to %s", output_csv)
    _generate_processing_summary(
        all_rows, output_csv, total_conversations, total_empty, language_counts
    )


def _generate_processing_summary(
    rows: List[Dict],
    output_csv: str,
    total_conversations: int,
    total_empty: int,
    language_counts: Counter,
) -> None:
    """Write a JSON summary of processing statistics alongside the output CSV."""
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_files_processed": len({row["source_file"] for row in rows}),
        "total_conversations": total_conversations,
        "empty_conversations": total_empty,
        "valid_conversations": len(rows),
        "success_rate": (
            (len(rows) / total_conversations * 100) if total_conversations > 0 else 0
        ),
        "language_distribution": dict(language_counts),
        "text_length_stats": {
            "min": min((len(row["text"]) for row in rows), default=0),
            "max": max((len(row["text"]) for row in rows), default=0),
            "avg": sum(len(row["text"]) for row in rows) / len(rows) if rows else 0,
        },
    }

    summary_file = output_csv.replace(".csv", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Processing summary saved to {summary_file}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = str(project_root / "data" / "raw")
    output_csv = str(project_root / "data" / "processed" / "cleaned_conversations.csv")

    process_raw_files(raw_dir, output_csv)
