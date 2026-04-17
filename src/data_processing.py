"""Data processing module for AI conversation exports.

Handles JSON parsing, text extraction, and language detection for
conversation data exported from Claude and ChatGPT. Supports parallel
file processing and persistent language detection caching.
"""

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import logging
import os
import threading
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
        self._cache_lock = threading.Lock()
        self._setup_logging()
        self._setup_cache()

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
        cache_file = os.path.join(self.cache_dir, "language_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("language cache must be a JSON object")
                self.language_cache = {
                    str(k): str(v) for k, v in loaded.items() if isinstance(k, str)
                }
                self.logger.info("Loaded %d cached language detections", len(self.language_cache))
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
                self.logger.warning(
                    "Failed to load language cache: %s", exc
                )
                self.language_cache = {}

    def _save_language_cache(self) -> None:
        """Persist language detection cache to disk."""
        cache_file = os.path.join(self.cache_dir, "language_cache.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.language_cache, f)
        except OSError as exc:
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

        with self._cache_lock:
            if text_hash in self.language_cache:
                return self.language_cache[text_hash]

        try:
            limited_text = text[:1000] if len(text) > 1000 else text
            lang = detect(limited_text)
            with self._cache_lock:
                self.language_cache[text_hash] = lang
            return lang
        except Exception as exc:
            self.logger.debug("Language detection failed for hash=%s: %s", text_hash, exc)
            with self._cache_lock:
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


def process_single_file(file_path: str, processor: DataProcessor) -> Tuple[List[Dict], int, int]:
    """Process one JSON file and return extracted conversations.

    Returns:
        Tuple of (rows, total_conversation_count, empty_conversation_count).
    """
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
    raw_dir: str,
    output_csv: str,
    max_workers: Optional[int] = None,
    cache_dir: str = "../data/cache",
) -> None:
    """Process all JSON conversation files in *raw_dir* and write a cleaned CSV.

    Uses parallel I/O, language detection caching, and batch CSV writes.
    """
    processor = DataProcessor(cache_dir=cache_dir, max_workers=max_workers)
    file_paths = glob.glob(os.path.join(raw_dir, "*.json"))

    if not file_paths:
        processor.logger.warning("No JSON files found in %s", raw_dir)
        return

    processor.logger.info("Found %d JSON files in %s", len(file_paths), raw_dir)

    all_rows_written = 0
    total_conversations = 0
    total_empty = 0
    language_counts: Counter = Counter()
    text_min = float("inf")
    text_max = 0
    text_total = 0
    source_files: set[str] = set()

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = ["conversation_id", "text", "language", "source_file"]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile, concurrent.futures.ThreadPoolExecutor(
        max_workers=processor.max_workers
    ) as executor:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        pending_rows: List[Dict] = []
        batch_size = 1000
        future_to_file = {
            executor.submit(process_single_file, fp, processor): fp for fp in file_paths
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(file_paths),
            desc="Processing files",
        ):
            file_path = future_to_file[future]
            try:
                rows, file_total, file_empty = future.result()
                total_conversations += file_total
                total_empty += file_empty
                source_files.add(os.path.basename(file_path))
                for row in rows:
                    language_counts[row["language"]] += 1
                    text_len = len(row["text"])
                    text_min = min(text_min, text_len)
                    text_max = max(text_max, text_len)
                    text_total += text_len
                pending_rows.extend(rows)
                if len(pending_rows) >= batch_size:
                    writer.writerows(pending_rows)
                    all_rows_written += len(pending_rows)
                    pending_rows = []
            except Exception as exc:
                processor.logger.error("Error processing %s: %s", file_path, exc)

        if pending_rows:
            writer.writerows(pending_rows)
            all_rows_written += len(pending_rows)

    processor._save_language_cache()

    # Log statistics
    processor.logger.info("Language Statistics:")
    for lang, count in language_counts.most_common():
        processor.logger.info("  %s: %d conversations", lang, count)

    processor.logger.info("Total conversations processed: %d", total_conversations)
    processor.logger.info("Empty conversations discarded: %d", total_empty)
    processor.logger.info("Non-empty conversations saved: %d", all_rows_written)
    if total_conversations > 0:
        processor.logger.info(
            "Success rate: %.2f%%", all_rows_written / total_conversations * 100
        )

    processor.logger.info("Processed data saved to %s", output_csv)
    _generate_processing_summary(
        output_csv=output_csv,
        total_files_processed=len(source_files),
        total_conversations=total_conversations,
        total_empty=total_empty,
        valid_conversations=all_rows_written,
        language_counts=language_counts,
        text_min=0 if text_min == float("inf") else int(text_min),
        text_max=int(text_max),
        text_avg=(text_total / all_rows_written) if all_rows_written else 0.0,
    )


def _generate_processing_summary(
    output_csv: str,
    total_files_processed: int,
    total_conversations: int,
    total_empty: int,
    valid_conversations: int,
    language_counts: Counter,
    text_min: int,
    text_max: int,
    text_avg: float,
) -> None:
    """Write a JSON summary of processing statistics alongside the output CSV."""
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_files_processed": total_files_processed,
        "total_conversations": total_conversations,
        "empty_conversations": total_empty,
        "valid_conversations": valid_conversations,
        "success_rate": (
            (valid_conversations / total_conversations * 100) if total_conversations > 0 else 0
        ),
        "language_distribution": dict(language_counts),
        "text_length_stats": {
            "min": text_min,
            "max": text_max,
            "avg": text_avg,
        },
    }

    summary_file = output_csv.replace(".csv", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Processing summary saved to {summary_file}")


def main() -> None:
    """Run stage 1 data processing."""
    parser = argparse.ArgumentParser(description="Process raw conversation exports")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("GENAI_RAW_DIR", str(project_root / "data" / "raw")),
    )
    parser.add_argument(
        "--output-csv",
        default=os.getenv(
            "GENAI_CLEANED_OUTPUT",
            str(project_root / "data" / "processed" / "cleaned_conversations.csv"),
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("GENAI_CACHE_DIR", str(project_root / "data" / "cache")),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("GENAI_MAX_WORKERS", "0")) or None,
    )
    args = parser.parse_args()

    process_raw_files(
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
