"""Pipeline performance benchmark.

Measures execution time, memory usage, and throughput for each stage
of the conversation categorization pipeline.
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import psutil


class PipelineBenchmark:
    """Benchmark each stage of the categorization pipeline.

    Args:
        output_dir: Directory where benchmark result JSON files are saved.
    """

    def __init__(self, output_dir: str = "../outputs/benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # System information
    # ------------------------------------------------------------------

    @staticmethod
    def get_system_info() -> Dict:
        """Collect hardware and environment details."""
        info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "memory_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
        }
        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
                info["cuda_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
        except ImportError:
            info["cuda_available"] = False
        return info

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_memory() -> Dict:
        """Snapshot current process memory usage."""
        proc = psutil.Process()
        return {
            "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "system_percent": psutil.virtual_memory().percent,
        }

    def benchmark_stage(
        self, name: str, func: Callable, *args: Any, **kwargs: Any
    ) -> Dict:
        """Run *func* and return timing and memory metrics."""
        gc.collect()
        mem_before = self._measure_memory()
        t0 = time.perf_counter()

        try:
            func(*args, **kwargs)
            success = True
            error = None
        except Exception as exc:
            success = False
            error = str(exc)

        elapsed = time.perf_counter() - t0
        mem_after = self._measure_memory()

        return {
            "stage": name,
            "success": success,
            "error": error,
            "elapsed_seconds": round(elapsed, 3),
            "memory_before_mb": mem_before["rss_mb"],
            "memory_after_mb": mem_after["rss_mb"],
            "memory_delta_mb": round(mem_after["rss_mb"] - mem_before["rss_mb"], 1),
        }

    # ------------------------------------------------------------------
    # Full pipeline benchmark
    # ------------------------------------------------------------------

    def run(self, raw_dir: str = "../data/raw") -> Dict:
        """Benchmark all pipeline stages end-to-end.

        Returns a dict with system info and per-stage metrics, also saved
        as a timestamped JSON file under *output_dir*.
        """
        from clustering import run_clustering
        from data_processing import process_raw_files
        from embedding import generate_embeddings_from_csv

        project_root = Path(__file__).resolve().parent.parent
        processed_dir = project_root / "data" / "processed"
        cleaned_csv = str(processed_dir / "cleaned_conversations.csv")
        embeddings_npy = str(processed_dir / "embeddings.npy")
        categorized_csv = str(processed_dir / "categorized_conversations.csv")

        print("=" * 60)
        print("PIPELINE BENCHMARK")
        print("=" * 60)

        report: Dict[str, Any] = {"system": self.get_system_info(), "stages": []}

        # Stage 1 — Data Processing
        if os.path.isdir(raw_dir):
            result = self.benchmark_stage(
                "data_processing", process_raw_files, raw_dir, cleaned_csv
            )
            report["stages"].append(result)
            print(f"  Data Processing: {result['elapsed_seconds']}s")
        else:
            print(f"  Skipping data processing ('{raw_dir}' not found)")

        # Stage 2 — Embedding Generation
        if os.path.isfile(cleaned_csv):
            result = self.benchmark_stage(
                "embedding", generate_embeddings_from_csv, cleaned_csv, embeddings_npy
            )
            report["stages"].append(result)
            print(f"  Embedding:       {result['elapsed_seconds']}s")

        # Stage 3 — Categorization
        if os.path.isfile(cleaned_csv):
            result = self.benchmark_stage(
                "clustering", run_clustering, cleaned_csv, categorized_csv
            )
            report["stages"].append(result)
            print(f"  Categorization:  {result['elapsed_seconds']}s")

        # Total
        total = sum(s["elapsed_seconds"] for s in report["stages"])
        report["total_seconds"] = round(total, 3)
        print(f"\n  Total: {total:.1f}s")

        # Persist
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f"benchmark_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

        return report


if __name__ == "__main__":
    PipelineBenchmark().run()
