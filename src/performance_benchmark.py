#!/usr/bin/env python3
"""
Performance Benchmark Script

This script compares the performance of the original implementation
vs the optimized versions and provides detailed metrics.
"""

import os
import time
import json
import pandas as pd
import psutil
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import optimized modules
try:
    from embedding_optimized import generate_embeddings_optimized
    from data_processing_optimized import process_raw_files_optimized
    from clustering_optimized import run_clustering_optimized
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    print("Optimized modules not found. Please ensure they are in the same directory.")

# Import original modules for comparison
try:
    from embedding import generate_embeddings
    from data_processing import process_raw_files
    from clustering import run_clustering
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print("Original modules not found. Comparison will be limited.")

class PerformanceBenchmark:
    """Performance benchmarking and comparison tool."""
    
    def __init__(self, output_dir: str = "../outputs/performance"):
        self.output_dir = output_dir
        self.results = {}
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Create output directory for benchmark results."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_system_info(self) -> Dict:
        """Collect system information for benchmarking context."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
            "python_version": f"{psutil.version_info}",
        }
        
        # GPU information
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            info["cuda_available"] = False
        
        return info
    
    def measure_memory_usage(self) -> Dict:
        """Measure current memory usage."""
        process = psutil.Process()
        return {
            "memory_percent": process.memory_percent(),
            "memory_mb": process.memory_info().rss / 1e6,
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_memory_available_gb": psutil.virtual_memory().available / 1e9
        }
    
    def benchmark_function(self, func, *args, **kwargs) -> Dict:
        """Benchmark a function and return performance metrics."""
        # Clear memory before benchmark
        gc.collect()
        
        # Measure initial state
        start_memory = self.measure_memory_usage()
        start_time = time.time()
        start_cpu_times = psutil.cpu_times()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Measure final state
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        end_cpu_times = psutil.cpu_times()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = end_memory["memory_mb"] - start_memory["memory_mb"]
        cpu_delta = end_cpu_times.user - start_cpu_times.user
        
        return {
            "success": success,
            "error": error,
            "execution_time_seconds": execution_time,
            "memory_delta_mb": memory_delta,
            "peak_memory_mb": end_memory["memory_mb"],
            "cpu_time_seconds": cpu_delta,
            "start_memory": start_memory,
            "end_memory": end_memory,
            "result": result
        }
    
    def benchmark_data_processing(self, raw_dir: str, sample_size: Optional[int] = None) -> Dict:
        """Benchmark data processing performance."""
        print("\n=== BENCHMARKING DATA PROCESSING ===")
        
        results = {"system_info": self.get_system_info()}
        
        # Create sample data if needed
        if sample_size:
            print(f"Creating sample dataset with {sample_size} conversations...")
            # Implementation would create sample data
        
        # Benchmark optimized version
        if HAS_OPTIMIZED:
            print("Benchmarking optimized data processing...")
            output_optimized = os.path.join(self.output_dir, "cleaned_conversations_optimized_bench.csv")
            optimized_result = self.benchmark_function(
                process_raw_files_optimized, 
                raw_dir, 
                output_optimized
            )
            results["optimized"] = optimized_result
            print(f"  Optimized: {optimized_result['execution_time_seconds']:.2f}s")
        
        # Benchmark original version
        if HAS_ORIGINAL:
            print("Benchmarking original data processing...")
            output_original = os.path.join(self.output_dir, "cleaned_conversations_original_bench.csv")
            original_result = self.benchmark_function(
                process_raw_files, 
                raw_dir, 
                output_original
            )
            results["original"] = original_result
            print(f"  Original: {original_result['execution_time_seconds']:.2f}s")
        
        # Calculate improvement
        if HAS_OPTIMIZED and HAS_ORIGINAL and results["optimized"]["success"] and results["original"]["success"]:
            improvement = (results["original"]["execution_time_seconds"] - 
                         results["optimized"]["execution_time_seconds"]) / results["original"]["execution_time_seconds"]
            results["improvement"] = {
                "speed_improvement_percent": improvement * 100,
                "memory_savings_mb": results["original"]["peak_memory_mb"] - results["optimized"]["peak_memory_mb"]
            }
            print(f"  Improvement: {improvement*100:.1f}% faster")
        
        return results
    
    def benchmark_embeddings(self, csv_file: str) -> Dict:
        """Benchmark embedding generation performance."""
        print("\n=== BENCHMARKING EMBEDDING GENERATION ===")
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return {"error": "CSV file not found"}
        
        results = {"system_info": self.get_system_info()}
        
        # Benchmark optimized version
        if HAS_OPTIMIZED:
            print("Benchmarking optimized embedding generation...")
            output_optimized = os.path.join(self.output_dir, "embeddings_optimized_bench.npy")
            optimized_result = self.benchmark_function(
                generate_embeddings_optimized, 
                csv_file, 
                output_optimized
            )
            results["optimized"] = optimized_result
            print(f"  Optimized: {optimized_result['execution_time_seconds']:.2f}s")
        
        # Benchmark original version
        if HAS_ORIGINAL:
            print("Benchmarking original embedding generation...")
            output_original = os.path.join(self.output_dir, "embeddings_original_bench.npy")
            original_result = self.benchmark_function(
                generate_embeddings, 
                csv_file, 
                output_original
            )
            results["original"] = original_result
            print(f"  Original: {original_result['execution_time_seconds']:.2f}s")
        
        # Calculate improvement
        if HAS_OPTIMIZED and HAS_ORIGINAL and results["optimized"]["success"] and results["original"]["success"]:
            improvement = (results["original"]["execution_time_seconds"] - 
                         results["optimized"]["execution_time_seconds"]) / results["original"]["execution_time_seconds"]
            results["improvement"] = {
                "speed_improvement_percent": improvement * 100,
                "memory_savings_mb": results["original"]["peak_memory_mb"] - results["optimized"]["peak_memory_mb"]
            }
            print(f"  Improvement: {improvement*100:.1f}% faster")
        
        return results
    
    def benchmark_clustering(self, csv_file: str) -> Dict:
        """Benchmark clustering/categorization performance."""
        print("\n=== BENCHMARKING CATEGORIZATION ===")
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return {"error": "CSV file not found"}
        
        results = {"system_info": self.get_system_info()}
        
        # Benchmark optimized version
        if HAS_OPTIMIZED:
            print("Benchmarking optimized categorization...")
            output_optimized = os.path.join(self.output_dir, "categorized_conversations_optimized_bench.csv")
            optimized_result = self.benchmark_function(
                run_clustering_optimized, 
                csv_file, 
                output_optimized
            )
            results["optimized"] = optimized_result
            print(f"  Optimized: {optimized_result['execution_time_seconds']:.2f}s")
        
        # Benchmark original version
        if HAS_ORIGINAL:
            print("Benchmarking original categorization...")
            output_original = os.path.join(self.output_dir, "categorized_conversations_original_bench.csv")
            original_result = self.benchmark_function(
                run_clustering, 
                csv_file, 
                output_original
            )
            results["original"] = original_result
            print(f"  Original: {original_result['execution_time_seconds']:.2f}s")
        
        # Calculate improvement
        if HAS_OPTIMIZED and HAS_ORIGINAL and results["optimized"]["success"] and results["original"]["success"]:
            improvement = (results["original"]["execution_time_seconds"] - 
                         results["optimized"]["execution_time_seconds"]) / results["original"]["execution_time_seconds"]
            results["improvement"] = {
                "speed_improvement_percent": improvement * 100,
                "memory_savings_mb": results["original"]["peak_memory_mb"] - results["optimized"]["peak_memory_mb"]
            }
            print(f"  Improvement: {improvement*100:.1f}% faster")
        
        return results
    
    def run_full_benchmark(self, raw_dir: str = "../data/raw", 
                          sample_size: Optional[int] = None) -> Dict:
        """Run complete benchmark suite."""
        print("Starting performance benchmark...")
        print(f"Output directory: {self.output_dir}")
        
        benchmark_results = {
            "benchmark_start": datetime.now().isoformat(),
            "system_info": self.get_system_info()
        }
        
        # 1. Data Processing Benchmark
        if os.path.exists(raw_dir):
            benchmark_results["data_processing"] = self.benchmark_data_processing(raw_dir, sample_size)
        else:
            print(f"Raw data directory not found: {raw_dir}")
            benchmark_results["data_processing"] = {"error": "Raw data directory not found"}
        
        # 2. Find cleaned CSV for subsequent benchmarks
        cleaned_csv = None
        if "data_processing" in benchmark_results and "optimized" in benchmark_results["data_processing"]:
            if benchmark_results["data_processing"]["optimized"]["success"]:
                cleaned_csv = os.path.join(self.output_dir, "cleaned_conversations_optimized_bench.csv")
        
        if not cleaned_csv or not os.path.exists(cleaned_csv):
            # Try to find existing cleaned CSV
            possible_paths = [
                "../data/processed/cleaned_conversations.csv",
                "../data/processed/cleaned_conversations_optimized.csv"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    cleaned_csv = path
                    break
        
        # 3. Embedding Benchmark
        if cleaned_csv and os.path.exists(cleaned_csv):
            benchmark_results["embeddings"] = self.benchmark_embeddings(cleaned_csv)
            benchmark_results["clustering"] = self.benchmark_clustering(cleaned_csv)
        else:
            print("No cleaned CSV found for embedding and clustering benchmarks")
            benchmark_results["embeddings"] = {"error": "No cleaned CSV found"}
            benchmark_results["clustering"] = {"error": "No cleaned CSV found"}
        
        # 4. Overall results
        benchmark_results["benchmark_end"] = datetime.now().isoformat()
        
        # Save results
        results_file = os.path.join(self.output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\nBenchmark complete! Results saved to: {results_file}")
        self.print_summary(benchmark_results)
        
        return benchmark_results
    
    def print_summary(self, results: Dict):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        system = results.get("system_info", {})
        print(f"System: {system.get('cpu_count', 'Unknown')} CPU cores, "
              f"{system.get('memory_total_gb', 0):.1f}GB RAM")
        if system.get("cuda_available"):
            print(f"GPU: {system.get('cuda_device_name', 'Unknown')} "
                  f"({system.get('cuda_memory_total', 0):.1f}GB)")
        
        # Print results for each benchmark
        for component in ["data_processing", "embeddings", "clustering"]:
            if component in results and "improvement" in results[component]:
                improvement = results[component]["improvement"]
                print(f"\n{component.replace('_', ' ').title()}:")
                print(f"  Speed improvement: {improvement['speed_improvement_percent']:.1f}%")
                print(f"  Memory savings: {improvement['memory_savings_mb']:.1f}MB")
        
        print("\nFor detailed results, see the generated JSON file.")

def main():
    """Main function to run benchmarks."""
    benchmark = PerformanceBenchmark()
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    return results

if __name__ == "__main__":
    main()