"""Embedding generation module for conversation text.

Uses SentenceTransformer to produce multilingual dense vector embeddings
with automatic GPU/MPS detection, dynamic batch sizing, persistent caching,
and chunked processing for memory efficiency.
"""

import gc
import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate sentence embeddings with caching and hardware acceleration.

    Args:
        model_name: HuggingFace model identifier for SentenceTransformer.
        cache_dir: Directory for persisting computed embeddings.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        cache_dir: str = "../data/cache",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model: Optional[SentenceTransformer] = None
        self.device = self._get_optimal_device()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Device & batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_optimal_device() -> str:
        """Select the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_optimal_batch_size(self, num_texts: int) -> int:
        """Heuristically choose a batch size based on available memory and device."""
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            base = 64 if gpu_mem > 8e9 else (32 if gpu_mem > 4e9 else 16)
        else:
            avail = psutil.virtual_memory().available
            base = 32 if avail > 8e9 else (16 if avail > 4e9 else 8)
        return min(base, max(1, num_texts // 10))

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _get_cache_key(self, texts: List[str]) -> str:
        """Derive a deterministic cache key from model name and text content."""
        content = f"{self.model_name}:{len(texts)}:{hash(tuple(texts))}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cached(self, cache_key: str) -> Optional[np.ndarray]:
        """Return cached embeddings if available, else ``None``."""
        path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"Cache load failed: {exc}")
        return None

    def _save_to_cache(self, embeddings: np.ndarray, cache_key: str) -> None:
        """Persist embeddings to the cache directory."""
        path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"Cache save failed: {exc}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model onto the target device."""
        if self.model is None:
            print(f"Loading model '{self.model_name}' on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda" and hasattr(self.model, "_modules"):
                self.model.half()  # FP16 for faster GPU inference

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_texts(texts: List[str], chunk_size: int = 1000) -> List[List[str]]:
        """Split *texts* into equal-sized chunks for memory-efficient processing."""
        return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    def generate_embeddings(self, texts: List[str], chunk_size: int = 1000) -> np.ndarray:
        """Encode *texts* into embeddings, processing in chunks.

        Results are cached on disk so repeated calls with identical input
        return instantly.

        Args:
            texts: Conversation strings to embed.
            chunk_size: Number of texts per processing chunk.

        Returns:
            NumPy array of shape ``(len(texts), embedding_dim)``.
        """
        self._load_model()

        cache_key = self._get_cache_key(texts)
        cached = self._load_cached(cache_key)
        if cached is not None:
            print(f"Loaded {len(cached)} embeddings from cache")
            return cached

        chunks = self._chunk_texts(texts, chunk_size)
        all_embeddings: List[np.ndarray] = []

        print(f"Processing {len(texts)} texts in {len(chunks)} chunks...")
        for chunk in tqdm(chunks, desc="Generating embeddings"):
            batch_size = self._get_optimal_batch_size(len(chunk))
            try:
                emb = self.model.encode(
                    chunk,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_embeddings.append(emb)
            except Exception:
                emb = self.model.encode(
                    chunk,
                    show_progress_bar=False,
                    batch_size=1,
                    convert_to_numpy=True,
                )
                all_embeddings.append(emb)

            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        result = np.vstack(all_embeddings)
        self._save_to_cache(result, cache_key)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> dict:
        """Return current CPU (and optionally GPU) memory statistics."""
        stats: dict = {
            "cpu_memory": {
                "total_gb": round(psutil.virtual_memory().total / 1e9, 1),
                "available_gb": round(psutil.virtual_memory().available / 1e9, 1),
                "percent": psutil.virtual_memory().percent,
            }
        }
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            }
        return stats


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def generate_embeddings_from_csv(
    csv_file: str,
    output_file: str,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    chunk_size: int = 1000,
) -> None:
    """Read a CSV with a ``text`` column, generate embeddings, and save as ``.npy``."""
    print("Loading conversation data...")
    df = pd.read_csv(csv_file)
    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    texts = [t for t in df["text"].astype(str).fillna("").tolist() if t.strip()]
    print(f"Processing {len(texts)} conversations...")

    generator = EmbeddingGenerator(model_name)
    print("Initial memory:", generator.get_memory_usage())

    embeddings = generator.generate_embeddings(texts, chunk_size)

    print("Final memory:", generator.get_memory_usage())
    np.save(output_file, embeddings)
    print(
        f"Embeddings saved to {output_file}  "
        f"(shape: {embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_path = str(project_root / "data" / "processed" / "cleaned_conversations.csv")
    out_path = str(project_root / "data" / "processed" / "embeddings.npy")

    generate_embeddings_from_csv(csv_path, out_path)
