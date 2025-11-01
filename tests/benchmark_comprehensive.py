"""
Comprehensive Benchmark v2.1

Tests ALL components after complete file integration.
Measures forward pass and learning success rates across different configurations.
"""

import torch
import json
import time
import traceback
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dhc_ssm.integration.dhc_ssm_model import DHCSSMArchitecture
    from dhc_ssm.utils.config import get_default_config, get_cpu_optimized_config, get_debug_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback benchmark...")
    DHCSSMArchitecture = None


class ComprehensiveBenchmark:
    """
    Complete benchmark testing all v2.1 improvements.
    """
    
    def __init__(self, num_trials: int = 15):
        self.num_trials = num_trials
        self.device = 'cpu'  # Safe default
        
    def test_forward_pass(self, model) -> Dict[str, Any]:
        """Test forward pass with various input conditions."""
        successful = 0
        failed = 0
        errors = []
        
        test_cases = [
            # Standard cases
            {'batch_size': 1, 'channels': 3, 'height': 32, 'width': 32},
            {'batch_size': 4, 'channels': 3, 'height': 64, 'width': 64},
            {'batch_size': 2, 'channels': 1, 'height': 28, 'width': 28},
            # Edge cases
            {'batch_size': 1, 'channels': 1, 'height': 16, 'width': 16},
            {'batch_size': 8, 'channels': 3, 'height': 128, 'width': 128},
        ]
        
        for trial, case in enumerate(test_cases * (self.num_trials // len(test_cases) + 1)):
            if trial >= self.num_trials:
                break
                
            try:
                # Create test input
                observation = torch.randn(
                    case['batch_size'], case['channels'], 
                    case['height'], case['width'], 
                    device=self.device
                )
                
                # Forward pass
                with torch.no_grad():
                    output = model(observation)
                
                # Validate output structure
                if isinstance(output, dict) and 'final_prediction' in output:
                    if not torch.isnan(output['final_prediction']).any():
                        successful += 1
                    else:
                        failed += 1
                        errors.append(f"Trial {trial+1}: NaN in final_prediction")
                else:
                    failed += 1
                    errors.append(f"Trial {trial+1}: Invalid output structure")
                    
            except Exception as e:
                failed += 1
                errors.append(f"Trial {trial+1}: {str(e)}")
        
        return {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / self.num_trials * 100,
            'errors': errors[:5]  # Keep first 5 errors
        }
    
    def test_learning_step(self, model) -> Dict[str, Any]:
        """Test learning step functionality."""
        successful = 0
        failed = 0
        errors = []
        
        for trial in range(self.num_trials):
            try:
                # Create test inputs
                current_obs = torch.randn(2, 3, 32, 32, device=self.device)
                next_obs = torch.randn(2, 3, 32, 32, device=self.device)
                
                # Learning step
                learning_output = model.deterministic_learning_step(current_obs, next_obs)
                
                # Validate learning output
                if isinstance(learning_output, dict):
                    if 'error' not in learning_output and 'deterministic_action' in learning_output:
                        action = learning_output['deterministic_action']
                        if not torch.isnan(action).any():
                            successful += 1
                        else:
                            failed += 1
                            errors.append(f"Trial {trial+1}: NaN in deterministic_action")
                    else:
                        failed += 1
                        error_msg = learning_output.get('error', 'Unknown learning error')
                        errors.append(f"Trial {trial+1}: {error_msg}")
                else:
                    failed += 1
                    errors.append(f"Trial {trial+1}: Invalid learning output type")
                    
            except Exception as e:
                failed += 1
                errors.append(f"Trial {trial+1}: {str(e)}")
        
        return {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / self.num_trials * 100,
            'errors': errors[:5]
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark across all configurations."""
        print("DHC-SSM v2.1 - COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        print(f"Testing with {self.num_trials} trials per configuration")
        
        results = {
            'benchmark_date': '2025-11-01 10:05 KST',
            'version': '2.1',
            'num_trials': self.num_trials,
            'configurations_tested': [],
            'overall_results': {},
            'detailed_results': {}
        }
        
        if DHCSSMArchitecture is None:
            # Fallback when imports fail
            print("❌ Cannot import DHC-SSM components")
            results['import_error'] = True
            results['overall_results'] = {
                'forward_pass_success_rate': 0.0,
                'learning_success_rate': 0.0,
                'production_ready': False,
                'error': 'Import failure - components not available'
            }
            return results
        
        # Test configurations
        configs_to_test = [
            ('debug', get_debug_config),
            ('cpu_optimized', get_cpu_optimized_config),
        ]
        
        total_forward_success = 0
        total_learning_success = 0
        total_configs = 0
        
        for config_name, config_fn in configs_to_test:
            print(f"\nTesting {config_name} configuration...")
            results['configurations_tested'].append(config_name)
            
            try:
                # Get configuration
                config = config_fn()
                
                # Initialize model
                model = DHCSSMArchitecture(config=config)
                model.eval()  # Evaluation mode
                
                print(f"  Model initialized: {config.spatial.feature_dim} spatial_dim")
                
                # Test forward pass
                print(f"  Testing forward pass ({self.num_trials} trials)...")
                forward_results = self.test_forward_pass(model)
                
                # Test learning step
                print(f"  Testing learning step ({self.num_trials} trials)...")
                learning_results = self.test_learning_step(model)
                
                # Store results
                config_results = {
                    'forward_pass': forward_results,
                    'learning_step': learning_results,
                    'config_summary': {
                        'spatial_dim': config.spatial.feature_dim,
                        'tactical_dim': config.tactical.state_dim,
                        'strategic_dim': config.strategic.causal_dim,
                        'device': config.system.device
                    }
                }
                
                results['detailed_results'][config_name] = config_results
                
                # Update totals
                total_forward_success += forward_results['success_rate']
                total_learning_success += learning_results['success_rate']
                total_configs += 1
                
                print(f"    Forward Pass: {forward_results['success_rate']:.1f}%")
                print(f"    Learning Step: {learning_results['success_rate']:.1f}%")
                
            except Exception as e:
                print(f"    ❌ Configuration {config_name} failed: {e}")
                results['detailed_results'][config_name] = {
                    'error': str(e),
                    'failed': True
                }
        
        # Compute overall results
        if total_configs > 0:
            avg_forward = total_forward_success / total_configs
            avg_learning = total_learning_success / total_configs
        else:
            avg_forward = 0.0
            avg_learning = 0.0
        
        is_production_ready = avg_forward >= 90.0 and avg_learning >= 90.0
        
        results['overall_results'] = {
            'forward_pass_success_rate': avg_forward,
            'learning_success_rate': avg_learning,
            'production_ready': is_production_ready,
            'configs_tested': total_configs,
            'improvement_from_v2_0': {
                'forward_pass': avg_forward - 70.0,  # v2.0 baseline
                'learning_step': avg_learning - 80.0   # v2.0 baseline
            }
        }
        
        # Production readiness assessment
        print(f"\n{'='*60}")
        print("COMPREHENSIVE BENCHMARK RESULTS v2.1")
        print(f"{'='*60}")
        print(f"Overall Forward Pass: {avg_forward:.1f}%")
        print(f"Overall Learning Steps: {avg_learning:.1f}%")
        print(f"Production Ready: {'✓ YES' if is_production_ready else '✗ NO'}")
        
        if not is_production_ready:
            gaps = []
            if avg_forward < 90:
                gaps.append(f"Forward: {90-avg_forward:.1f}% needed")
            if avg_learning < 90:
                gaps.append(f"Learning: {90-avg_learning:.1f}% needed")
            print(f"Gaps: {', '.join(gaps)}")
        
        return results


if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark(num_trials=20)
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    output_file = 'benchmark_comprehensive_v2_1.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Quick summary
    overall = results['overall_results']
    print(f"\nQUICK SUMMARY:")
    print(f"✓ Forward: {overall['forward_pass_success_rate']:.1f}%")
    print(f"✓ Learning: {overall['learning_success_rate']:.1f}%")
    print(f"✓ Ready: {'YES' if overall['production_ready'] else 'NO'}")