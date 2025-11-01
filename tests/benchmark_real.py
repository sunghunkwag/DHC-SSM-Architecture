"""
REAL Performance Benchmark for DHC-SSM v2.0

This script provides ACTUAL empirical measurements, not fake marketing metrics.
Tests both v1.0 (legacy) and v2.0 (enhanced) to get real performance comparison.

Author: Sung Hun Kwag (being honest now)
Date: 2025-11-01
"""

import torch
import numpy as np
import time
import json
from typing import Dict, Any, List, Tuple
import traceback
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm import DHCSSMArchitecture, get_small_config


class RealBenchmark:
    """
    ACTUAL performance measurement tool.
    No fake metrics, only real test results.
    """
    
    def __init__(self, num_trials: int = 10, device: str = 'cpu'):
        self.num_trials = num_trials
        self.device = device
        self.results = {
            'v2_enhanced': {
                'forward_pass': {'successful': 0, 'failed': 0, 'errors': []},
                'learning_step': {'successful': 0, 'failed': 0, 'errors': []},
                'execution_times': [],
                'memory_usage': []
            }
        }
        
    def generate_test_data(self, batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test observation pair."""
        obs1 = torch.randn(batch_size, 3, 64, 64, device=self.device)
        obs2 = torch.randn(batch_size, 3, 64, 64, device=self.device)
        return obs1, obs2
    
    def test_forward_pass(self, model: DHCSSMArchitecture) -> Dict[str, Any]:
        """Test forward pass multiple times."""
        print(f"\n{'='*50}")
        print("TESTING FORWARD PASS")
        print(f"{'='*50}")
        
        successful = 0
        failed = 0
        errors = []
        execution_times = []
        
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}...", end=" ")
            
            try:
                # Generate test data
                obs, _ = self.generate_test_data()
                
                # Time the forward pass
                start_time = time.time()
                
                with torch.no_grad():
                    output = model.forward(obs)
                    
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Validate output structure
                required_keys = ['predictions', 'final_prediction', 'processing_info']
                for key in required_keys:
                    if key not in output:
                        raise KeyError(f"Missing required output key: {key}")
                
                # Validate prediction shapes
                final_pred = output['final_prediction']
                if len(final_pred.shape) != 2:
                    raise ValueError(f"Wrong prediction shape: {final_pred.shape}")
                
                successful += 1
                execution_times.append(execution_time)
                print("✓ SUCCESS")
                
            except Exception as e:
                failed += 1
                error_msg = f"Trial {trial + 1}: {str(e)}"
                errors.append(error_msg)
                print(f"✗ FAILED - {str(e)}")
        
        results = {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / self.num_trials * 100,
            'errors': errors,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'execution_times': execution_times
        }
        
        print(f"\nForward Pass Results:")
        print(f"  Successful: {successful}/{self.num_trials} ({results['success_rate']:.1f}%)")
        print(f"  Failed: {failed}/{self.num_trials}")
        print(f"  Avg Execution Time: {results['avg_execution_time']:.4f}s")
        
        return results
    
    def test_learning_step(self, model: DHCSSMArchitecture) -> Dict[str, Any]:
        """Test learning step multiple times."""
        print(f"\n{'='*50}")
        print("TESTING LEARNING STEPS")
        print(f"{'='*50}")
        
        successful = 0
        failed = 0
        errors = []
        execution_times = []
        total_errors = []
        
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}...", end=" ")
            
            try:\n                # Generate test data\n                obs1, obs2 = self.generate_test_data()\n                \n                # Time the learning step\n                start_time = time.time()\n                \n                learning_output = model.deterministic_learning_step(\n                    observation=obs1,\n                    actual_next_observation=obs2\n                )\n                \n                end_time = time.time()\n                execution_time = end_time - start_time\n                \n                # Validate learning output\n                required_keys = ['deterministic_action', 'learning_diagnostics', 'parameter_updates_applied']\n                for key in required_keys:\n                    if key not in learning_output:\n                        raise KeyError(f"Missing required learning key: {key}")\n                \n                # Check deterministic properties\n                diagnostics = learning_output['learning_diagnostics']\n                if not diagnostics.get('deterministic', False):\n                    raise ValueError("Learning is not deterministic!")\n                \n                if learning_output.get('sampling_uncertainty') != 'eliminated':\n                    raise ValueError("Probabilistic sampling not eliminated!")\n                \n                successful += 1\n                execution_times.append(execution_time)\n                total_errors.append(diagnostics['total_error'])\n                print("✓ SUCCESS")\n                \n            except Exception as e:\n                failed += 1\n                error_msg = f"Trial {trial + 1}: {str(e)}"\n                errors.append(error_msg)\n                print(f"✗ FAILED - {str(e)}")\n        \n        results = {\n            'successful': successful,\n            'failed': failed,\n            'success_rate': successful / self.num_trials * 100,\n            'errors': errors,\n            'avg_execution_time': np.mean(execution_times) if execution_times else 0,\n            'avg_total_error': np.mean(total_errors) if total_errors else 0,\n            'error_std': np.std(total_errors) if total_errors else 0,\n            'execution_times': execution_times,\n            'total_errors': total_errors\n        }\n        \n        print(f"\nLearning Step Results:")\n        print(f"  Successful: {successful}/{self.num_trials} ({results['success_rate']:.1f}%)")\n        print(f"  Failed: {failed}/{self.num_trials}")\n        print(f"  Avg Execution Time: {results['avg_execution_time']:.4f}s")\n        print(f"  Avg Total Error: {results['avg_total_error']:.6f} ± {results['error_std']:.6f}")\n        \n        return results\n    \n    def run_complete_benchmark(self) -> Dict[str, Any]:\n        """Run complete benchmark suite."""\n        print("REAL DHC-SSM BENCHMARK")\n        print("=" * 70)\n        print("This provides ACTUAL measurements, not fake marketing metrics!")\n        print(f"Device: {self.device}")\n        print(f"Number of trials: {self.num_trials}")\n        \n        # Initialize model\n        print("\\nInitializing DHC-SSM v2.0 Enhanced...")\n        try:\n            config = get_small_config()  # Use small config for faster testing\n            config.system.device = self.device\n            model = DHCSSMArchitecture(config=config)\n            print("✓ Model initialized successfully")\n        except Exception as e:\n            print(f"✗ Model initialization failed: {e}")\n            return {'initialization_failed': True, 'error': str(e)}\n        \n        # Test forward pass\n        forward_results = self.test_forward_pass(model)\n        \n        # Test learning steps\n        learning_results = self.test_learning_step(model)\n        \n        # Compile final results\n        final_results = {\n            'benchmark_date': '2025-11-01',\n            'device': self.device,\n            'num_trials': self.num_trials,\n            'model_version': '2.0.0',\n            'forward_pass': forward_results,\n            'learning_steps': learning_results,\n            'overall_assessment': {\n                'forward_success_rate': forward_results['success_rate'],\n                'learning_success_rate': learning_results['success_rate'],\n                'completely_functional': (\n                    forward_results['success_rate'] == 100.0 and \n                    learning_results['success_rate'] == 100.0\n                )\n            }\n        }\n        \n        # Print final summary\n        print(f"\\n{'='*70}")\n        print("REAL BENCHMARK RESULTS")\n        print(f"{'='*70}")\n        print(f"Forward Pass Success Rate: {forward_results['success_rate']:.1f}%")\n        print(f"Learning Steps Success Rate: {learning_results['success_rate']:.1f}%")\n        print(f"Overall Functional: {final_results['overall_assessment']['completely_functional']}")\n        \n        if not final_results['overall_assessment']['completely_functional']:\n            print("\\nERRORS DETECTED:")\n            all_errors = forward_results['errors'] + learning_results['errors']\n            for error in all_errors[:5]:  # Show first 5 errors\n                print(f"  - {error}")\n            if len(all_errors) > 5:\n                print(f"  ... and {len(all_errors) - 5} more errors")\n        \n        # Save results to file\n        with open('benchmark_results_real.json', 'w') as f:\n            json.dump(final_results, f, indent=2)\n        print(f"\\nResults saved to benchmark_results_real.json")\n        \n        return final_results


def main():\n    """Run the real benchmark."""\n    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n    \n    benchmark = RealBenchmark(num_trials=10, device=device)\n    results = benchmark.run_complete_benchmark()\n    \n    return results


if __name__ == "__main__":\n    # Set seeds for reproducible results\n    torch.manual_seed(42)\n    np.random.seed(42)\n    \n    results = main()