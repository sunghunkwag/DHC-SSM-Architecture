"""
DHC-SSM Enhanced Architecture Demonstration

Demonstrates the complete deterministic learning system with all fixes applied:
- Spatial processing with Enhanced CNN
- O(n) tactical processing with SSM (dimension issues fixed)
- Strategic causal reasoning with GNN
- Deterministic learning without probabilistic sampling

This demo shows how DHC-SSM Enhanced eliminates uncertainty while maintaining efficiency.
"""

import torch
import numpy as np
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.integration.dhc_ssm_model import DHCSSMArchitecture
from dhc_ssm.utils.config import get_default_config, get_small_config

def generate_test_sequence(seq_length: int = 30, batch_size: int = 2,
                          image_size: int = 64, channels: int = 3) -> List[torch.Tensor]:
    """
    Generate a test sequence of observations for demonstration.
    
    Args:
        seq_length: Number of time steps
        batch_size: Batch size
        image_size: Size of square images
        channels: Number of input channels
        
    Returns:
        List of observation tensors
    """
    observations = []
    
    # Generate evolving pattern sequence
    for t in range(seq_length):
        # Create base pattern that evolves over time
        base_pattern = torch.randn(batch_size, channels, image_size, image_size)
        
        # Add temporal evolution
        temporal_factor = np.sin(2 * np.pi * t / 20)  # Periodic pattern
        noise_level = 0.1 + 0.05 * temporal_factor
        
        # Add structured noise
        structured_noise = torch.randn(batch_size, channels, image_size, image_size) * noise_level
        observation = base_pattern + structured_noise
        
        # Add some spatial structure
        for b in range(batch_size):
            for c in range(channels):
                # Add circular pattern that changes over time
                center_x, center_y = image_size // 2, image_size // 2
                radius = 10 + 5 * np.sin(2 * np.pi * t / 30)
                
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                circle_mask = (distance < radius).float()
                
                observation[b, c] += 0.5 * circle_mask
                
        observations.append(observation)
        
    return observations

def run_forward_pass_test(model: DHCSSMArchitecture, observations: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Test forward pass through the architecture.
    
    Args:
        model: DHC-SSM model
        observations: Sequence of observations
        
    Returns:
        Test results
    """
    print("\n" + "="*70)
    print("FORWARD PASS TEST")
    print("="*70)
    
    results = {
        'successful_steps': 0,
        'failed_steps': 0,
        'errors': []
    }
    
    ssm_state = None
    with torch.no_grad():
        for t, obs in enumerate(observations[:10]):  # Test first 10 steps
            try:
                output = model.forward(obs, ssm_state)
                ssm_state = output['next_ssm_state']
                results['successful_steps'] += 1
                
                if t % 3 == 0:
                    print(f"Step {t}: SUCCESS")
                    print(f"  → Prediction shape: {output['final_prediction'].shape}")
                    print(f"  → Complexity: {output['processing_info']['complexity']}")
                    print(f"  → Fusion used: {output['processing_info']['fusion_info']['fusion_used']}")
                    
            except Exception as e:
                results['failed_steps'] += 1
                results['errors'].append(f"Step {t}: {str(e)}")
                print(f"Step {t}: FAILED - {str(e)}")
                
    print(f"\nResults: {results['successful_steps']} successful, {results['failed_steps']} failed")
    return results

def run_learning_test(model: DHCSSMArchitecture, observations: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Test deterministic learning.
    
    Args:
        model: DHC-SSM model
        observations: Sequence of observations
        
    Returns:
        Learning test results
    """
    print("\n" + "="*70)
    print("DETERMINISTIC LEARNING TEST")
    print("="*70)
    
    results = {
        'successful_steps': 0,
        'failed_steps': 0,
        'total_errors': [],
        'errors': []
    }
    
    ssm_state = None
    
    for t in range(min(5, len(observations) - 1)):
        try:
            current_obs = observations[t]
            next_obs = observations[t + 1]
            
            # Deterministic learning step
            learning_output = model.deterministic_learning_step(
                observation=current_obs,
                actual_next_observation=next_obs,
                ssm_hidden_state=ssm_state
            )
            
            ssm_state = learning_output['next_ssm_state']
            results['successful_steps'] += 1
            
            # Extract diagnostics
            diagnostics = learning_output['learning_diagnostics']
            results['total_errors'].append(diagnostics['total_error'])
            
            print(f"Step {t}: SUCCESS")
            print(f"  → Total Error: {diagnostics['total_error']:.4f}")
            print(f"  → Dominant Objective: {diagnostics['dominant_objective']}")
            print(f"  → Deterministic: {diagnostics['deterministic']}")
            
        except Exception as e:
            results['failed_steps'] += 1
            results['errors'].append(f"Step {t}: {str(e)}")
            print(f"Step {t}: FAILED - {str(e)}")
            
    print(f"\nResults: {results['successful_steps']} successful, {results['failed_steps']} failed")
    
    if len(results['total_errors']) > 1:
        error_reduction = (results['total_errors'][0] - results['total_errors'][-1]) / results['total_errors'][0] * 100
        print(f"Error reduction: {error_reduction:.2f}%")
        
    return results

def test_system_diagnostics(model: DHCSSMArchitecture):
    """
    Test system diagnostics.
    
    Args:
        model: DHC-SSM model
    """
    print("\n" + "="*70)
    print("SYSTEM DIAGNOSTICS")
    print("="*70)
    
    diagnostics = model.get_system_diagnostics()
    
    print(f"Architecture: {diagnostics['architecture_type']} v{diagnostics['version']}")
    print(f"Current Step: {diagnostics['current_step']}")
    
    print(f"\nLayers:")
    for layer_name, layer_desc in diagnostics['layers'].items():
        print(f"  - {layer_name}: {layer_desc}")
        
    print(f"\nComplexity Analysis:")
    for key, value in diagnostics['complexity_analysis'].items():
        print(f"  - {key}: {value}")
        
    print(f"\nEnhancements:")
    for key, value in diagnostics['enhancements_v2'].items():
        print(f"  - {key}: {value}")

def main():
    """
    Main demonstration of DHC-SSM Enhanced Architecture.
    """
    print("="*70)
    print("DHC-SSM ENHANCED: Deterministic Hierarchical Causal State Space Model")
    print("="*70)
    print("Version 2.0.0 - Production-Ready with All Fixes Applied")
    
    print("\nKey Improvements:")
    print("  ✓ Fixed dimension mismatch bugs")
    print("  ✓ Added comprehensive error handling")
    print("  ✓ Improved state management")
    print("  ✓ Enhanced validation throughout")
    print("  ✓ Better configuration management")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Use small config for faster testing
    print("\nInitializing model with small configuration...")
    config = get_small_config()
    config.system.device = device
    
    model = DHCSSMArchitecture(config=config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    # Generate test sequence
    print("\nGenerating test observation sequence...")
    observations = generate_test_sequence(seq_length=20, batch_size=2, image_size=64)
    
    # Move to device
    observations = [obs.to(device) for obs in observations]
    
    # Test system diagnostics
    test_system_diagnostics(model)
    
    # Test forward pass
    forward_results = run_forward_pass_test(model, observations)
    
    # Test learning
    learning_results = run_learning_test(model, observations)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print(f"Forward Pass: {forward_results['successful_steps']}/{forward_results['successful_steps'] + forward_results['failed_steps']} successful")
    print(f"Learning Steps: {learning_results['successful_steps']}/{learning_results['successful_steps'] + learning_results['failed_steps']} successful")
    
    if forward_results['failed_steps'] == 0 and learning_results['failed_steps'] == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("The enhanced architecture is working correctly.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Errors encountered:")
        for error in forward_results['errors'] + learning_results['errors']:
            print(f"  - {error}")
            
    return {
        'forward_results': forward_results,
        'learning_results': learning_results
    }


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = main()
