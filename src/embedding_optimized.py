import os
import pandas as pd
import numpy as np
import hashlib
import pickle
import psutil
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import gc
from tqdm import tqdm

class OptimizedEmbeddingGenerator:
    """Optimized embedding generator with caching, GPU acceleration, and memory management."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2", 
                 cache_dir: str = "../data/cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.device = self._get_optimal_device()
        self._ensure_cache_dir()
        
    def _get_optimal_device(self) -> str:
        """Determine the best device for processing."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key based on texts and model."""
        content = f"{self.model_name}:{len(texts)}:{hash(tuple(texts))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_model(self):
        """Load model with optimizations."""
        if self.model is None:
            print(f"Loading model '{self.model_name}' on device '{self.device}'...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Enable mixed precision for CUDA
            if self.device == "cuda" and hasattr(self.model, '_modules'):
                self.model.half()  # Use FP16 for faster processing
    
    def _get_optimal_batch_size(self, num_texts: int) -> int:
        """Calculate optimal batch size based on available memory and device."""
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Estimate batch size based on GPU memory (conservative estimate)
            if gpu_memory > 8e9:  # 8GB+
                base_batch_size = 64
            elif gpu_memory > 4e9:  # 4GB+
                base_batch_size = 32
            else:
                base_batch_size = 16
        else:
            # CPU-based batch sizing
            available_memory = psutil.virtual_memory().available
            if available_memory > 8e9:  # 8GB+
                base_batch_size = 32
            elif available_memory > 4e9:  # 4GB+
                base_batch_size = 16
            else:
                base_batch_size = 8
        
        # Adjust based on text count
        return min(base_batch_size, max(1, num_texts // 10))
    
    def _chunk_texts(self, texts: List[str], chunk_size: int = 1000) -> List[List[str]]:
        """Split texts into manageable chunks for memory efficiency."""
        return [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    def _load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None
    
    def _save_embeddings_to_cache(self, embeddings: np.ndarray, cache_key: str):
        """Save embeddings to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def generate_embeddings_chunked(self, texts: List[str], 
                                  chunk_size: int = 1000) -> np.ndarray:
        """Generate embeddings with chunking for memory efficiency."""
        self._load_model()
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        cached_embeddings = self._load_cached_embeddings(cache_key)
        if cached_embeddings is not None:
            print(f"Loaded {len(cached_embeddings)} embeddings from cache")
            return cached_embeddings
        
        # Process in chunks
        text_chunks = self._chunk_texts(texts, chunk_size)
        all_embeddings = []
        
        print(f"Processing {len(texts)} texts in {len(text_chunks)} chunks...")
        
        for i, chunk in enumerate(tqdm(text_chunks, desc="Processing chunks")):
            batch_size = self._get_optimal_batch_size(len(chunk))
            
            try:
                chunk_embeddings = self.model.encode(
                    chunk, 
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for better similarity computation
                )
                all_embeddings.append(chunk_embeddings)
                
                # Force garbage collection after each chunk
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                # Fallback to smaller batch size
                chunk_embeddings = self.model.encode(
                    chunk, 
                    show_progress_bar=False,
                    batch_size=1,
                    convert_to_numpy=True
                )
                all_embeddings.append(chunk_embeddings)
        
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        # Cache the results
        self._save_embeddings_to_cache(final_embeddings, cache_key)
        
        return final_embeddings
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        stats = {
            "cpu_memory": {
                "total": psutil.virtual_memory().total / 1e9,
                "available": psutil.virtual_memory().available / 1e9,
                "percent": psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9
            }
        
        return stats

def generate_embeddings_optimized(csv_file: str, output_file: str, 
                                 model_name: str = "paraphrase-multilingual-mpnet-base-v2",
                                 chunk_size: int = 1000):
    """Optimized embedding generation with performance monitoring."""
    
    # Load data
    print("Loading conversation data...")
    df = pd.read_csv(csv_file)
    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")
    
    # Convert to string and filter out empty texts
    texts = df["text"].astype(str).fillna("").tolist()
    texts = [text for text in texts if text.strip()]
    
    print(f"Processing {len(texts)} conversations...")
    
    # Initialize generator
    generator = OptimizedEmbeddingGenerator(model_name)
    
    # Print initial memory usage
    print("Initial memory usage:", generator.get_memory_usage())
    
    # Generate embeddings
    embeddings = generator.generate_embeddings_chunked(texts, chunk_size)
    
    # Print final memory usage
    print("Final memory usage:", generator.get_memory_usage())
    
    # Save embeddings
    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}")
    print(f"Shape: {embeddings.shape}")
    print(f"Memory usage: {embeddings.nbytes / 1e6:.2f} MB")

if __name__ == "__main__":
    # Paths relative to src folder
    csv_file = os.path.join("..", "data", "processed", "cleaned_conversations.csv")
    output_file = os.path.join("..", "data", "processed", "embeddings_optimized.npy")
    
    # Use optimized generation
    generate_embeddings_optimized(csv_file, output_file)