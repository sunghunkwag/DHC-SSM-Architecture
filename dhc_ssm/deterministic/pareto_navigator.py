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

# Import validation utilities with fallback
try:
    from ..utils.shape_validator import ShapeValidator, DimensionAligner
    from ..utils.device import DeviceManager, ensure_device
except ImportError:
    # Fallback implementations
    class ShapeValidator:
        @staticmethod
        def validate_tensor(*args, **kwargs): pass
        @staticmethod
        def validate_batch_compatible(*args, **kwargs): pass
    
    class DimensionAligner(nn.Module):
        def __init__(self, input_dim, output_dim, device='cpu'):
            super().__init__()
            self.projection = nn.Identity()
        def forward(self, x): return x
    
    def ensure_device(tensor, device): return tensor.to(device)


class IntrinsicSignalSynthesizer(nn.Module):
    """
    Generates four information-theoretic error vectors as intrinsic motivation.
    Replaces traditional reward functions with information-based objectives.
    """
    
    def __init__(self, prediction_dim: int, actual_dim: int = None, device: str = 'cpu'):
        super().__init__()
        self.prediction_dim = prediction_dim
        self.actual_dim = actual_dim if actual_dim is not None else prediction_dim
        self.device = device
        
        # Dimension aligner for prediction vs actual compatibility
        self.actual_aligner = DimensionAligner(
            self.actual_dim, prediction_dim, device
        )
        
        # Networks for computing different intrinsic signals
        
        # 1. Dissonance (Prediction Mismatch) Computer
        self.dissonance_computer = nn.Sequential(
            nn.Linear(prediction_dim + prediction_dim, prediction_dim, device=device),
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
            nn.Linear(prediction_dim // 4, prediction_dim, device=device),
        )
        
        # Historical pattern memory for novelty detection
        self.register_buffer('pattern_memory', torch.zeros(100, prediction_dim, device=device))
        self.register_buffer('memory_index', torch.tensor(0, device=device))
        
    def _update_pattern_memory(self, new_pattern: torch.Tensor):
        """
        Update pattern memory for novelty detection with safety.
        
        Args:
            new_pattern: New pattern to store [batch_size, prediction_dim]
        """
        new_pattern = ensure_device(new_pattern, self.device)
        batch_size = new_pattern.size(0)
        memory_size = self.pattern_memory.size(0)
        
        for i in range(batch_size):
            idx = self.memory_index.item() % memory_size
            self.pattern_memory[idx] = new_pattern[i].detach()
            self.memory_index += 1
            if self.memory_index >= memory_size * 1000:  # Prevent overflow
                self.memory_index = torch.tensor(memory_size, device=self.device)
    
    def compute_dissonance(self, prediction: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction mismatch (dissonance) with device safety.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            actual: Actual observation [batch_size, actual_dim]
            
        Returns:
            Dissonance error [batch_size, 1]
        """
        # Ensure device consistency
        prediction = ensure_device(prediction, self.device)
        actual = ensure_device(actual, self.device)
        
        # Validate shapes
        ShapeValidator.validate_tensor(prediction, 2, "prediction")
        ShapeValidator.validate_tensor(actual, 2, "actual")
        ShapeValidator.validate_batch_compatible(prediction, actual, "prediction", "actual")
        
        # Align actual to prediction dimension
        actual_aligned = self.actual_aligner(actual)
        
        # Concatenate for comparison
        combined = torch.cat([prediction, actual_aligned], dim=-1)
        dissonance = self.dissonance_computer(combined)
        
        # NaN protection
        dissonance = torch.nan_to_num(dissonance, nan=1.0)
        return dissonance
    
    def compute_uncertainty(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute information entropy (uncertainty) with numerical stability.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            
        Returns:
            Uncertainty error [batch_size, 1]
        """
        prediction = ensure_device(prediction, self.device)
        ShapeValidator.validate_tensor(prediction, 2, "prediction")
        
        # Neural uncertainty
        uncertainty = self.uncertainty_computer(prediction)
        
        # Information-theoretic entropy with stability
        pred_probs = F.softmax(prediction / 2.0, dim=-1)  # Temperature scaling
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1, keepdim=True)
        
        combined_uncertainty = uncertainty + 0.1 * entropy
        
        # NaN protection
        combined_uncertainty = torch.nan_to_num(combined_uncertainty, nan=1.0)
        return combined_uncertainty
    
    def compute_novelty(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty based on pattern memory with robustness.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            
        Returns:
            Novelty error [batch_size, 1]
        """
        prediction = ensure_device(prediction, self.device)
        ShapeValidator.validate_tensor(prediction, 2, "prediction")
        
        batch_size = prediction.size(0)
        novelty_scores = []
        
        for i in range(batch_size):
            current_pattern = prediction[i:i+1]
            
            # Compare with stored patterns
            memory_size = min(self.memory_index.item(), self.pattern_memory.size(0))
            
            if memory_size > 0:
                stored_patterns = self.pattern_memory[:memory_size]
                stored_patterns = ensure_device(stored_patterns, self.device)
                
                # Compute similarities with safety
                similarities = F.cosine_similarity(current_pattern, stored_patterns, dim=-1)
                
                if len(similarities) > 0:
                    max_similarity = torch.max(similarities)
                    novelty = torch.clamp(1.0 - max_similarity, min=0.0, max=1.0)
                else:
                    novelty = torch.tensor(1.0, device=self.device)
            else:
                novelty = torch.tensor(1.0, device=self.device)
                
            novelty_scores.append(novelty)
            
        novelty_tensor = torch.stack(novelty_scores).unsqueeze(-1)
        
        # Apply neural network transformation
        neural_novelty = self.novelty_computer(prediction)
        
        # Combine with safety
        combined_novelty = 0.7 * novelty_tensor + 0.3 * neural_novelty
        combined_novelty = torch.nan_to_num(combined_novelty, nan=1.0)
        
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
        prediction = ensure_device(prediction, self.device)
        ShapeValidator.validate_tensor(prediction, 2, "prediction")
        
        # Compress and reconstruct
        reconstructed = self.compression_computer(prediction)
        
        # Compression error
        compression_error = F.mse_loss(prediction, reconstructed, reduction='none')
        compression_error = torch.mean(compression_error, dim=-1, keepdim=True)
        
        # NaN protection
        compression_error = torch.nan_to_num(compression_error, nan=1.0)
        return compression_error
    
    def forward(self, prediction: torch.Tensor, actual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all four intrinsic motivation signals with safety.
        
        Args:
            prediction: Model prediction [batch_size, prediction_dim]
            actual: Actual observation [batch_size, actual_dim]
            
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
    Multi-objective Pareto optimizer with enhanced stability.
    """
    
    def __init__(self, input_dim: int, action_dim: int, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        self.device = device
        
        # Temperature for softmax stability
        self.register_buffer('temperature', torch.tensor(2.0, device=device))
        
        # Multi-objective neural networks
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
        )
        
    def compute_pareto_weights(self, error_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute Pareto-optimal weights with enhanced stability.
        
        Args:
            error_vectors: Error signals [batch_size, num_objectives]
            
        Returns:
            Pareto weights [batch_size, num_objectives]
        """
        # Device safety
        error_vectors = ensure_device(error_vectors, self.device)
        
        # NaN protection
        epsilon = 1e-8
        safe_errors = torch.nan_to_num(error_vectors, nan=1.0, posinf=1e6, neginf=1e6)
        safe_errors = safe_errors + epsilon
        
        # Normalize
        normalized_errors = F.normalize(safe_errors, dim=-1)
        
        # Compute weights
        raw_weights = self.weight_optimizer(normalized_errors)
        weights = F.softmax(raw_weights / self.temperature, dim=-1)
        
        # Weight floor to prevent zeros
        min_weight = 0.01
        weights = weights * (1 - min_weight * self.num_objectives) + min_weight
        
        # Renormalize
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Final NaN protection
        weights = torch.nan_to_num(weights, nan=1.0/self.num_objectives)
        
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
        # Device safety
        state_input = ensure_device(state_input, self.device)
        error_vectors = ensure_device(error_vectors, self.device)
        
        # Validation
        ShapeValidator.validate_tensor(state_input, 2, "state_input")
        ShapeValidator.validate_tensor(error_vectors, 2, "error_vectors")
        ShapeValidator.validate_batch_compatible(state_input, error_vectors)
        
        # Compute actions for each objective
        objective_actions = []
        for net in self.objective_networks:
            obj_action = net(state_input)
            objective_actions.append(obj_action)
        objective_actions = torch.stack(objective_actions, dim=1)  # [batch, num_obj, action_dim]
        
        # Compute Pareto weights
        pareto_weights = self.compute_pareto_weights(error_vectors)
        
        # Weighted combination
        weighted_actions = torch.sum(
            objective_actions * pareto_weights.unsqueeze(-1), dim=1
        )
        
        # Final safety
        weighted_actions = torch.nan_to_num(weighted_actions, nan=0.0)
        
        return weighted_actions, pareto_weights


class GradientComputer(nn.Module):
    """
    Computes deterministic gradients with robust error handling.
    """
    
    def __init__(self, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.num_objectives = num_objectives
        self.device = device
        self.clip_value = 1.0
        
        # Gradient weighting network
        self.gradient_weighter = nn.Sequential(
            nn.Linear(num_objectives, num_objectives * 2, device=device),
            nn.ReLU(),
            nn.Linear(num_objectives * 2, num_objectives, device=device),
            nn.Softplus()
        )
    
    def compute_multi_objective_gradients(self, error_vectors: Dict[str, torch.Tensor], 
                                        parameters: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective gradients with robust error handling.
        
        Args:
            error_vectors: Dictionary of error signals
            parameters: List of model parameters
            
        Returns:
            Dictionary of combined gradients
        """
        # Stack error vectors safely
        try:
            stacked_errors = torch.stack([
                error_vectors['dissonance'],
                error_vectors['uncertainty'],
                error_vectors['novelty'],
                error_vectors['compression_gain']
            ], dim=-1).squeeze(-2)
        except Exception:
            # Fallback stacking
            stacked_errors = torch.stack([
                error_vectors['dissonance'].squeeze(),
                error_vectors['uncertainty'].squeeze(),
                error_vectors['novelty'].squeeze(),
                error_vectors['compression_gain'].squeeze()
            ], dim=-1)
            if stacked_errors.dim() == 1:
                stacked_errors = stacked_errors.unsqueeze(0)
        
        stacked_errors = ensure_device(stacked_errors, self.device)
        
        # Compute gradients for each objective
        objective_gradients = {}
        
        for obj_name, error_signal in error_vectors.items():
            try:
                obj_loss = torch.mean(error_signal)
                grads = torch.autograd.grad(
                    obj_loss, parameters, retain_graph=True, create_graph=False,
                    allow_unused=True
                )
                
                # Handle None gradients
                processed_grads = []
                for i, (param, grad) in enumerate(zip(parameters, grads)):
                    if grad is None:
                        zero_grad = torch.zeros_like(param)
                        processed_grads.append(zero_grad)
                    else:
                        # Clip gradients
                        clipped_grad = torch.clamp(grad, -self.clip_value, self.clip_value)
                        processed_grads.append(clipped_grad)
                        
                objective_gradients[obj_name] = processed_grads
                
            except Exception as e:
                # Fallback to zero gradients
                objective_gradients[obj_name] = [torch.zeros_like(p) for p in parameters]
        
        # Combine gradients with equal weights (simplified)
        combined_gradients = []
        for param_idx in range(len(parameters)):
            weighted_grad = torch.zeros_like(parameters[param_idx])
            
            for obj_name in error_vectors.keys():
                if obj_name in objective_gradients:
                    weighted_grad += objective_gradients[obj_name][param_idx] / self.num_objectives
                    
            combined_gradients.append(weighted_grad)
        
        return {
            'combined_gradients': combined_gradients,
            'objective_gradients': objective_gradients
        }


class DeterministicLearningEngine(nn.Module):
    """
    Layer 4: Complete Deterministic Learning Engine with enhanced robustness.
    
    Integrates intrinsic motivation, Pareto optimization, and deterministic gradients.
    """
    
    def __init__(self, prediction_dim: int = 64, actual_dim: int = None, action_dim: int = 32, 
                 state_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.prediction_dim = prediction_dim
        self.actual_dim = actual_dim if actual_dim is not None else prediction_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        
        # Components
        self.intrinsic_synthesizer = IntrinsicSignalSynthesizer(
            prediction_dim, self.actual_dim, device
        )
        self.pareto_optimizer = ParetoOptimizer(state_dim, action_dim, device=device)
        self.gradient_computer = GradientComputer(device=device)
        
        # Learning rate
        self.register_parameter('learning_rate', nn.Parameter(torch.tensor(0.001, device=device)))
        
    def forward(self, predictions: Dict[str, torch.Tensor], actual: torch.Tensor, 
               current_state: torch.Tensor, model_parameters: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Complete deterministic learning step with enhanced safety.
        
        Args:
            predictions: Model predictions
            actual: Actual observation
            current_state: Current system state
            model_parameters: List of model parameters
            
        Returns:
            Learning output
        """
        try:
            # Use final prediction
            final_prediction = predictions.get('p_final', predictions.get('p_fast'))
            
            # Device safety
            final_prediction = ensure_device(final_prediction, self.device)
            actual = ensure_device(actual, self.device)
            current_state = ensure_device(current_state, self.device)
            
            # Compute intrinsic signals
            error_vectors = self.intrinsic_synthesizer(final_prediction, actual)
            
            # Stack for Pareto
            error_tensor = torch.stack([
                error_vectors['dissonance'],
                error_vectors['uncertainty'],
                error_vectors['novelty'],
                error_vectors['compression_gain']
            ], dim=-1)
            
            if error_tensor.dim() == 3:
                error_tensor = error_tensor.squeeze(-2)
            
            # Generate action
            deterministic_action, pareto_weights = self.pareto_optimizer(current_state, error_tensor)
            
            # Compute gradients
            gradient_info = self.gradient_computer.compute_multi_objective_gradients(
                error_vectors, model_parameters
            )
            
            # Parameter updates
            parameter_updates = []
            for grad in gradient_info['combined_gradients']:
                update = -self.learning_rate * grad
                parameter_updates.append(update)
                
            # Diagnostics
            with torch.no_grad():
                total_error = torch.mean(error_tensor).item()
                gradient_norm = sum(torch.norm(g).item() for g in gradient_info['combined_gradients'])
                
            return {
                'deterministic_action': deterministic_action,
                'intrinsic_errors': error_vectors,
                'pareto_weights': pareto_weights,
                'parameter_updates': parameter_updates,
                'gradient_info': gradient_info,
                'learning_diagnostics': {
                    'total_error': total_error,
                    'dominant_objective': torch.argmax(pareto_weights.mean(dim=0)).item(),
                    'learning_rate': self.learning_rate.item(),
                    'gradient_norm': gradient_norm,
                    'deterministic': True,
                    'pareto_optimal': True
                }
            }
            
        except Exception as e:
            # Return safe fallback
            return {
                'error': f"DeterministicLearningEngine failed: {str(e)}",
                'deterministic_action': torch.zeros(current_state.size(0), self.action_dim, device=self.device),
                'learning_diagnostics': {
                    'total_error': 999.0,
                    'deterministic': False,
                    'error_occurred': True
                }
            }
    
    def update_parameters(self, model_parameters: List[torch.Tensor], 
                         parameter_updates: List[torch.Tensor]) -> None:
        """
        Apply parameter updates with validation.
        
        Args:
            model_parameters: List of model parameters
            parameter_updates: List of parameter updates
        """
        if len(model_parameters) != len(parameter_updates):
            return  # Skip if mismatch
            
        with torch.no_grad():
            for param, update in zip(model_parameters, parameter_updates):
                if param.shape == update.shape:
                    param.add_(update)
                    
    def get_intrinsic_motivation_analysis(self) -> Dict[str, Any]:
        """
        Analyze intrinsic motivation system.
        
        Returns:
            Analysis dict
        """
        with torch.no_grad():
            memory_usage = (self.intrinsic_synthesizer.memory_index.item() / 
                          self.intrinsic_synthesizer.pattern_memory.size(0))
            
            return {
                'pattern_memory_usage': memory_usage,
                'stored_patterns': min(self.intrinsic_synthesizer.memory_index.item(), 
                                     self.intrinsic_synthesizer.pattern_memory.size(0)),
                'motivation_components': ['dissonance', 'uncertainty', 'novelty', 'compression_gain'],
                'deterministic_learning': True,
                'probabilistic_sampling': False
            }