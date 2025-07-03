import os
import json
import glob
import csv
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import logging
from collections import Counter
import concurrent.futures
from tqdm import tqdm

# Optional faster JSON library
try:
    import orjson as fast_json
    JSON_LOADS = fast_json.loads
except ImportError:
    import json as fast_json
    JSON_LOADS = json.loads

# Optional faster language detection with caching
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

class OptimizedDataProcessor:
    """Optimized data processor with caching, parallel processing, and memory efficiency."""
    
    def __init__(self, cache_dir: str = "../data/cache", max_workers: Optional[int] = None):
        self.cache_dir = cache_dir
        self.max_workers = max_workers or min(cpu_count(), 8)  # Limit to reasonable number
        self.language_cache = {}
        self._setup_cache()
        self._setup_logging()
    
    def _setup_cache(self):
        """Setup cache directory and load existing caches."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_language_cache()
    
    def _setup_logging(self):
        """Setup logging for performance monitoring."""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _load_language_cache(self):
        """Load language detection cache."""
        cache_file = os.path.join(self.cache_dir, "language_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.language_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.language_cache)} cached language detections")
            except Exception as e:
                self.logger.warning(f"Failed to load language cache: {e}")
                self.language_cache = {}
    
    def _save_language_cache(self):
        """Save language detection cache."""
        cache_file = os.path.join(self.cache_dir, "language_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.language_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.warning(f"Failed to save language cache: {e}")
    
    @lru_cache(maxsize=10000)
    def detect_language_cached(self, text_hash: str, text: str) -> str:
        """Cached language detection with fallback."""
        if not HAS_LANGDETECT:
            return 'unknown'
        
        if text_hash in self.language_cache:
            return self.language_cache[text_hash]
        
        try:
            # Limit text length for performance
            limited_text = text[:1000] if len(text) > 1000 else text
            lang = detect(limited_text)
            self.language_cache[text_hash] = lang
            return lang
        except:
            self.language_cache[text_hash] = 'unknown'
            return 'unknown'
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text for caching purposes."""
        return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:16]
    
    def extract_text_optimized(self, conversation: Dict) -> str:
        """Optimized text extraction with early returns and reduced recursion."""
        texts = []
        
        # Claude format - optimized path
        if "chat_messages" in conversation:
            for msg in conversation.get("chat_messages", []):
                # Direct text extraction
                if msg.get("text"):
                    texts.append(msg["text"].strip())
                    continue
                
                # Content array processing
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
        
        # ChatGPT format - optimized path
        elif "mapping" in conversation:
            for msg_data in conversation.get("mapping", {}).values():
                message = msg_data.get("message")
                if not message:
                    continue
                
                content = message.get("content", {})
                if isinstance(content, dict) and "parts" in content:
                    for part in content["parts"]:
                        if isinstance(part, dict):
                            if part.get("content_type") == "audio_transcription" and part.get("text"):
                                texts.append(part["text"].strip())
                        elif isinstance(part, str) and part.strip():
                            texts.append(part.strip())
            
            return " ".join(texts)
        
        # Fallback to limited deep search
        return self._limited_deep_search(conversation, max_depth=3)
    
    def _limited_deep_search(self, obj, max_depth: int = 3, current_depth: int = 0) -> str:
        """Limited depth search for text with early termination."""
        if current_depth >= max_depth:
            return ""
        
        texts = []
        
        if isinstance(obj, dict):
            # Priority fields first
            for field in ["text", "content", "message"]:
                if field in obj and isinstance(obj[field], str) and obj[field].strip():
                    texts.append(obj[field].strip())
            
            # Limited recursive search
            for value in list(obj.values())[:10]:  # Limit to first 10 values
                result = self._limited_deep_search(value, max_depth, current_depth + 1)
                if result:
                    texts.append(result)
                    if len(texts) > 5:  # Limit collected texts
                        break
        
        elif isinstance(obj, list):
            for item in obj[:20]:  # Limit to first 20 items
                result = self._limited_deep_search(item, max_depth, current_depth + 1)
                if result:
                    texts.append(result)
                    if len(texts) > 5:  # Limit collected texts
                        break
        
        return " ".join(texts)

def process_single_file(file_path: str) -> Tuple[List[Dict], int, int]:
    """Process a single JSON file and return conversations, total count, and empty count."""
    processor = OptimizedDataProcessor()
    rows = []
    empty_count = 0
    total_conversations = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Use faster JSON loading
            data = JSON_LOADS(f.read())
        
        conversations = []
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            conversations = [data]
        
        for conv in conversations:
            total_conversations += 1
            conv_id = conv.get("uuid", conv.get("title", f"conv_{total_conversations}_{os.path.basename(file_path)}"))
            conversation_text = processor.extract_text_optimized(conv)
            
            if conversation_text.strip():
                # Generate hash for caching
                text_hash = processor.get_text_hash(conversation_text)
                lang = processor.detect_language_cached(text_hash, conversation_text)
                
                rows.append({
                    "conversation_id": conv_id,
                    "text": conversation_text,
                    "language": lang,
                    "source_file": os.path.basename(file_path)
                })
            else:
                empty_count += 1
                
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
    
    return rows, total_conversations, empty_count

def process_raw_files_optimized(raw_dir: str, output_csv: str, max_workers: Optional[int] = None):
    """Optimized processing with parallel file handling and progress tracking."""
    
    # Setup
    processor = OptimizedDataProcessor(max_workers=max_workers)
    file_paths = glob.glob(os.path.join(raw_dir, "*.json"))
    
    if not file_paths:
        processor.logger.warning(f"No JSON files found in {raw_dir}")
        return
    
    processor.logger.info(f"Found {len(file_paths)} JSON files in {raw_dir}")
    
    # Process files in parallel
    all_rows = []
    total_conversations = 0
    total_empty = 0
    language_counts = Counter()
    
    # Use ThreadPoolExecutor for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
        # Submit all file processing tasks
        future_to_file = {executor.submit(process_single_file, file_path): file_path 
                         for file_path in file_paths}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(file_paths), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                rows, file_total, file_empty = future.result()
                all_rows.extend(rows)
                total_conversations += file_total
                total_empty += file_empty
                
                # Update language counts
                for row in rows:
                    language_counts[row['language']] += 1
                    
            except Exception as e:
                processor.logger.error(f"Error processing {file_path}: {e}")
    
    # Save language cache
    processor._save_language_cache()
    
    # Statistics
    processor.logger.info("\nLanguage Statistics:")
    for lang, count in language_counts.most_common():
        processor.logger.info(f"{lang}: {count} conversations")
    
    processor.logger.info(f"\nTotal conversations processed: {total_conversations}")
    processor.logger.info(f"Empty conversations discarded: {total_empty}")
    processor.logger.info(f"Non-empty conversations saved: {len(all_rows)}")
    
    if total_conversations > 0:
        success_rate = (len(all_rows) / total_conversations) * 100
        processor.logger.info(f"Success rate: {success_rate:.2f}%")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write to CSV with optimization
    fieldnames = ["conversation_id", "text", "language", "source_file"]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write in batches for better memory management
        batch_size = 1000
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            writer.writerows(batch)
    
    processor.logger.info(f"Processed data saved to {output_csv}")
    
    # Generate summary statistics
    generate_processing_summary(all_rows, output_csv, total_conversations, total_empty, language_counts)

def generate_processing_summary(rows: List[Dict], output_csv: str, 
                              total_conversations: int, total_empty: int, 
                              language_counts: Counter):
    """Generate a summary of processing statistics."""
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_files_processed": len(set(row['source_file'] for row in rows)),
        "total_conversations": total_conversations,
        "empty_conversations": total_empty,
        "valid_conversations": len(rows),
        "success_rate": (len(rows) / total_conversations * 100) if total_conversations > 0 else 0,
        "language_distribution": dict(language_counts),
        "text_length_stats": {
            "min": min(len(row['text']) for row in rows) if rows else 0,
            "max": max(len(row['text']) for row in rows) if rows else 0,
            "avg": sum(len(row['text']) for row in rows) / len(rows) if rows else 0
        }
    }
    
    summary_file = output_csv.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing summary saved to {summary_file}")

if __name__ == "__main__":
    # Adjust paths if necessary
    raw_dir = os.path.join("..", "data", "raw")
    output_csv = os.path.join("..", "data", "processed", "cleaned_conversations_optimized.csv")
    
    # Run optimized processing
    process_raw_files_optimized(raw_dir, output_csv)