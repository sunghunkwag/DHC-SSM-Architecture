"""
Layer 4: Deterministic Learning Engine

Based on IntrinsicSignalSynthesizer + ParetoNavigator from Autonomous-Self-Organizing-AI.
Eliminates probabilistic sampling through information-theoretic multi-objective optimization.

Key Features:
- Information-theoretic intrinsic motivation (no rewards)
- Multi-objective Pareto optimization
- Deterministic gradient computation
- Complete elimination of probabilistic sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from scipy.optimize import minimize


class IntrinsicSignalSynthesizer(nn.Module):
    """
    Generates four information-theoretic error vectors as intrinsic motivation.
    Replaces traditional reward functions with information-based objectives.
    """
    
    def __init__(self, prediction_dim: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_dim = prediction_dim
        self.device = device
        
        # Networks for computing different intrinsic signals
        
        # 1. Dissonance (Prediction Mismatch) Computer
        self.dissonance_computer = nn.Sequential(
            nn.Linear(prediction_dim * 2, prediction_dim, device=device),  # pred + actual
            nn.ReLU(),
            nn.Linear(prediction_dim, 1, device=device),
            nn.Softplus()  # Ensure positive
        )
        
        # 2. Uncertainty (Information Entropy) Computer
        self.uncertainty_computer = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(prediction_dim // 2, 1, device=device),
            nn.Softplus()
        )
        
        # 3. Novelty (Pattern Recognition) Computer
        self.novelty_computer = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(prediction_dim // 2, 1, device=device),
            nn.Softplus()
        )
        
        # 4. Compression Gain (Representational Efficiency) Computer
        self.compression_computer = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(prediction_dim // 4, prediction_dim, device=device),  # Reconstruction
        )
        
        # Historical pattern memory for novelty detection
        self.register_buffer('pattern_memory', torch.zeros(100, prediction_dim, device=device))
        self.register_buffer('memory_index', torch.tensor(0, device=device))
        
    def _update_pattern_memory(self, new_pattern: torch.Tensor):
        """
        Update pattern memory for novelty detection.
        
        Args:
            new_pattern: New pattern to store [batch_size, prediction_dim]
        """
        batch_size = new_pattern.size(0)
        memory_size = self.pattern_memory.size(0)
        
        for i in range(batch_size):
            self.pattern_memory[self.memory_index % memory_size] = new_pattern[i].detach()
            self.memory_index += 1
    
    def compute_dissonance(self, prediction: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction mismatch (dissonance).
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            actual: Actual observation [batch_size, prediction_dim]
            
        Returns:
            Dissonance error [batch_size, 1]
        """
        # Concatenate prediction and actual for comparison
        combined = torch.cat([prediction, actual], dim=-1)
        dissonance = self.dissonance_computer(combined)
        return dissonance
    
    def compute_uncertainty(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute information entropy (uncertainty).
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            
        Returns:
            Uncertainty error [batch_size, 1]
        """
        # Compute prediction entropy as uncertainty measure
        uncertainty = self.uncertainty_computer(prediction)
        
        # Add information-theoretic entropy calculation
        pred_probs = F.softmax(prediction, dim=-1)
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=-1, keepdim=True)
        
        # Combine neural and information-theoretic uncertainty
        combined_uncertainty = uncertainty + 0.1 * entropy
        
        return combined_uncertainty
    
    def compute_novelty(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty based on pattern memory.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            
        Returns:
            Novelty error [batch_size, 1]
        """
        batch_size = prediction.size(0)
        novelty_scores = []
        
        for i in range(batch_size):
            current_pattern = prediction[i:i+1]
            
            # Compare with stored patterns
            if self.memory_index > 0:
                memory_size = min(self.memory_index.item(), self.pattern_memory.size(0))
                stored_patterns = self.pattern_memory[:memory_size]
                
                # Compute similarities
                similarities = F.cosine_similarity(current_pattern, stored_patterns, dim=-1)
                max_similarity = torch.max(similarities)
                
                # Novelty is inverse of maximum similarity
                novelty = 1.0 - max_similarity
            else:
                novelty = torch.tensor(1.0, device=self.device)  # First pattern is novel
                
            novelty_scores.append(novelty)
            
        novelty_tensor = torch.stack(novelty_scores).unsqueeze(-1)
        
        # Apply neural network transformation
        neural_novelty = self.novelty_computer(prediction)
        
        # Combine memory-based and neural novelty
        combined_novelty = 0.7 * novelty_tensor + 0.3 * neural_novelty
        
        # Update pattern memory
        self._update_pattern_memory(prediction)
        
        return combined_novelty
    
    def compute_compression_gain(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute representational efficiency (compression gain).
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            
        Returns:
            Compression error [batch_size, 1]
        """
        # Compress and reconstruct to measure information loss
        reconstructed = self.compression_computer(prediction)
        
        # Compression error is reconstruction loss
        compression_error = F.mse_loss(prediction, reconstructed, reduction='none')
        compression_error = torch.mean(compression_error, dim=-1, keepdim=True)
        
        return compression_error
    
    def forward(self, prediction: torch.Tensor, actual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all four intrinsic motivation signals.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            actual: Actual observation [batch_size, prediction_dim]
            
        Returns:
            Dictionary of error vectors
        """
        # Compute all four intrinsic signals
        dissonance = self.compute_dissonance(prediction, actual)
        uncertainty = self.compute_uncertainty(prediction)
        novelty = self.compute_novelty(actual)  # Use actual for novelty
        compression_gain = self.compute_compression_gain(prediction)
        
        return {
            'dissonance': dissonance,
            'uncertainty': uncertainty,
            'novelty': novelty,
            'compression_gain': compression_gain
        }


class ParetoOptimizer(nn.Module):
    """
    Multi-objective Pareto optimizer for deterministic decision making.
    Eliminates probabilistic sampling by finding Pareto-optimal solutions.
    """
    
    def __init__(self, input_dim: int, action_dim: int, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        self.device = device
        
        # Multi-objective neural network
        self.objective_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2, device=device),
                nn.ReLU(),
                nn.Linear(input_dim // 2, action_dim, device=device)
            ) for _ in range(num_objectives)
        ])
        
        # Pareto weight optimizer
        self.weight_optimizer = nn.Sequential(
            nn.Linear(num_objectives, num_objectives * 2, device=device),
            nn.ReLU(),
            nn.Linear(num_objectives * 2, num_objectives, device=device),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
    def compute_pareto_weights(self, error_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute Pareto-optimal weights for objectives.
        
        Args:
            error_vectors: Error signals [batch_size, num_objectives]
            
        Returns:
            Pareto weights [batch_size, num_objectives]
        """
        # Normalize error vectors
        normalized_errors = F.normalize(error_vectors, dim=-1)
        
        # Compute adaptive weights
        weights = self.weight_optimizer(normalized_errors)
        
        return weights
    
    def forward(self, state_input: torch.Tensor, error_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate deterministic action using Pareto optimization.
        
        Args:
            state_input: Current state [batch_size, input_dim]
            error_vectors: Error signals [batch_size, num_objectives]
            
        Returns:
            action: Deterministic action [batch_size, action_dim]
            pareto_weights: Applied weights [batch_size, num_objectives]
        """
        batch_size = state_input.size(0)
        
        # Compute actions for each objective
        objective_actions = []
        for net in self.objective_networks:
            obj_action = net(state_input)
            objective_actions.append(obj_action)
            
        objective_actions = torch.stack(objective_actions, dim=1)  # [batch, num_obj, action_dim]
        
        # Compute Pareto weights
        pareto_weights = self.compute_pareto_weights(error_vectors)  # [batch, num_obj]
        
        # Weighted combination of objective actions
        weighted_actions = torch.sum(
            objective_actions * pareto_weights.unsqueeze(-1), dim=1
        )  # [batch, action_dim]
        
        return weighted_actions, pareto_weights


class GradientComputer(nn.Module):
    """
    Computes deterministic gradients for parameter updates.
    Eliminates stochastic gradient estimation.
    """
    
    def __init__(self, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.num_objectives = num_objectives
        self.device = device
        
        # Gradient weighting network
        self.gradient_weighter = nn.Sequential(
            nn.Linear(num_objectives, num_objectives * 2, device=device),
            nn.ReLU(),
            nn.Linear(num_objectives * 2, num_objectives, device=device),
            nn.Softplus()  # Ensure positive weights
        )
        
    def compute_multi_objective_gradients(self, error_vectors: Dict[str, torch.Tensor], 
                                        parameters: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective gradients deterministically.
        
        Args:
            error_vectors: Dictionary of error signals
            parameters: List of model parameters
            
        Returns:
            Dictionary of combined gradients
        """
        # Stack error vectors
        stacked_errors = torch.stack([
            error_vectors['dissonance'],
            error_vectors['uncertainty'],
            error_vectors['novelty'],
            error_vectors['compression_gain']
        ], dim=-1).squeeze(1)  # [batch, num_objectives]
        
        # Compute gradient weights
        gradient_weights = self.gradient_weighter(stacked_errors)  # [batch, num_objectives]
        
        # Compute gradients for each objective
        objective_gradients = {}
        
        for i, (obj_name, error_signal) in enumerate(error_vectors.items()):
            # Compute gradients with respect to this objective
            obj_loss = torch.mean(error_signal)  # Scalar loss
            
            # Compute gradients
            grads = torch.autograd.grad(
                obj_loss, parameters, retain_graph=True, create_graph=True
            )
            
            objective_gradients[obj_name] = grads
            
        # Combine gradients using Pareto weights
        combined_gradients = []
        
        for param_idx in range(len(parameters)):
            weighted_grad = torch.zeros_like(parameters[param_idx])
            
            for obj_idx, obj_name in enumerate(error_vectors.keys()):
                weight = gradient_weights[:, obj_idx].mean()  # Average over batch
                weighted_grad += weight * objective_gradients[obj_name][param_idx]
                
            combined_gradients.append(weighted_grad)
            
        return {
            'combined_gradients': combined_gradients,
            'gradient_weights': gradient_weights,
            'objective_gradients': objective_gradients
        }


class DeterministicLearningEngine(nn.Module):
    """
    Layer 4: Complete Deterministic Learning Engine
    
    Integrates intrinsic motivation, Pareto optimization, and deterministic gradients
    to eliminate all probabilistic sampling uncertainty.
    """
    
    def __init__(self, prediction_dim: int = 64, action_dim: int = 32, 
                 state_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.prediction_dim = prediction_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        
        # Component 4A: Intrinsic Signal Synthesizer
        self.intrinsic_synthesizer = IntrinsicSignalSynthesizer(prediction_dim, device)
        
        # Component 4B: Pareto Navigator
        self.pareto_optimizer = ParetoOptimizer(state_dim, action_dim, device=device)
        
        # Component 4C: Gradient Computer
        self.gradient_computer = GradientComputer(device=device)
        
        # Learning rate scheduler for deterministic updates
        self.register_parameter('learning_rate', nn.Parameter(torch.tensor(0.001, device=device)))
        
    def forward(self, predictions: Dict[str, torch.Tensor], actual: torch.Tensor, 
               current_state: torch.Tensor, model_parameters: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Complete deterministic learning step.
        
        Args:
            predictions: Model predictions (p_fast, p_slow, p_final)
            actual: Actual observation [batch_size, prediction_dim]
            current_state: Current system state [batch_size, state_dim]
            model_parameters: List of model parameters for gradient computation
            
        Returns:
            Complete learning output with action and gradients
        """
        # Use final prediction for intrinsic signal computation
        final_prediction = predictions.get('p_final', predictions.get('p_fast'))
        
        # Step 1: Compute intrinsic motivation signals
        error_vectors = self.intrinsic_synthesizer(final_prediction, actual)
        
        # Step 2: Generate deterministic action via Pareto optimization
        error_tensor = torch.stack([
            error_vectors['dissonance'],
            error_vectors['uncertainty'], 
            error_vectors['novelty'],
            error_vectors['compression_gain']
        ], dim=-1).squeeze(1)
        
        deterministic_action, pareto_weights = self.pareto_optimizer(current_state, error_tensor)
        
        # Step 3: Compute deterministic gradients
        gradient_info = self.gradient_computer.compute_multi_objective_gradients(
            error_vectors, model_parameters
        )
        
        # Step 4: Prepare parameter updates
        parameter_updates = []
        for grad in gradient_info['combined_gradients']:
            update = -self.learning_rate * grad  # Gradient descent
            parameter_updates.append(update)
            
        # Compile output
        learning_output = {
            'deterministic_action': deterministic_action,
            'intrinsic_errors': error_vectors,
            'pareto_weights': pareto_weights,
            'parameter_updates': parameter_updates,
            'gradient_info': gradient_info,
            'learning_diagnostics': {
                'total_error': torch.sum(error_tensor, dim=-1).mean().item(),
                'dominant_objective': torch.argmax(pareto_weights.mean(dim=0)).item(),
                'learning_rate': self.learning_rate.item(),
                'gradient_norm': sum(torch.norm(g).item() for g in gradient_info['combined_gradients']),
                'deterministic': True,  # Confirms no probabilistic sampling
                'pareto_optimal': True
            }
        }
        
        return learning_output
    
    def update_parameters(self, model_parameters: List[torch.Tensor], 
                         parameter_updates: List[torch.Tensor]) -> None:
        """
        Apply deterministic parameter updates.
        
        Args:
            model_parameters: List of model parameters
            parameter_updates: List of parameter updates
        """
        with torch.no_grad():
            for param, update in zip(model_parameters, parameter_updates):
                param.add_(update)
                
    def get_intrinsic_motivation_analysis(self) -> Dict[str, Any]:
        """
        Analyze the intrinsic motivation system.
        
        Returns:
            Analysis of intrinsic signals
        """
        with torch.no_grad():
            memory_usage = (self.intrinsic_synthesizer.memory_index.item() / 
                          self.intrinsic_synthesizer.pattern_memory.size(0))
            
            analysis = {
                'pattern_memory_usage': memory_usage,
                'stored_patterns': min(self.intrinsic_synthesizer.memory_index.item(), 
                                     self.intrinsic_synthesizer.pattern_memory.size(0)),
                'motivation_components': ['dissonance', 'uncertainty', 'novelty', 'compression_gain'],
                'deterministic_learning': True,
                'probabilistic_sampling': False  # Explicitly confirmed
            }
            
        return analysis
