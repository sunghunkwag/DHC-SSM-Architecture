"""
Re-run REAL benchmark after fixes
"""
import json
from tests.benchmark_real import RealBenchmark

if __name__ == "__main__":
    benchmark = RealBenchmark(num_trials=10)
    results = benchmark.run_complete_benchmark()
    with open('benchmark_results_after_fixes.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to benchmark_results_after_fixes.json")
