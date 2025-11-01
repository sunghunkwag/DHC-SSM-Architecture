"""
Layer 2: Fast Tactical Processor (Enhanced v2.0)

Based on StateSpaceModel from SSM-MetaRL-TestCompute.
Provides O(n) complexity sequence processing, replacing Transformer's O(n²) attention.

Enhancements in v2.0:
- Fixed dimension alignment issues
- Added shape validation
- Improved temporal fusion with automatic alignment
- Device consistency management

Key Features:
- O(n) linear complexity for sequence processing
- Temporal state management
- Real-time tactical predictions
- Memory-efficient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Import validation utilities
try:
    from ..utils.shape_validator import ShapeValidator, DimensionAligner
except ImportError:
    # Fallback if utils not available
    class ShapeValidator:
        @staticmethod
        def validate_tensor(*args, **kwargs):
            pass
        @staticmethod
        def validate_batch_consistency(*args, **kwargs):
            pass
        @staticmethod
        def validate_device_consistency(*args, **kwargs):
            pass
    
    class DimensionAligner(nn.Module):
        def __init__(self, target_dim, device='cpu'):
            super().__init__()
            self.target_dim = target_dim
            self.linear = nn.Linear(target_dim, target_dim, device=device)
        def forward(self, x, source_name="input"):
            return self.linear(x) if x.size(-1) == self.target_dim else x


class StateSpaceProcessor(nn.Module):
    """
    Core O(n) State Space Model for sequential processing.
    
    This replaces Transformer's O(n²) attention mechanism with linear complexity
    state space operations for real-time tactical processing.
    """
    
    def __init__(self, input_dim: int, state_dim: int, output_dim: int, 
                 hidden_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # State transition matrices (learnable)
        self.A = nn.Linear(state_dim, state_dim, device=device, bias=False)
        self.B = nn.Linear(input_dim, state_dim, device=device, bias=False)
        self.C = nn.Linear(state_dim, output_dim, device=device, bias=False)
        self.D = nn.Linear(input_dim, output_dim, device=device, bias=False)
        
        # Non-linear transformations for enhanced expressivity
        self.state_nonlinearity = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim, device=device)
        )
        
        # Output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device)
        )
        
        # Initialize parameters for stability
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters for stable learning."""
        # Initialize A matrix to be stable (eigenvalues < 1)
        with torch.no_grad():
            nn.init.normal_(self.A.weight, std=0.1)
            # Make A matrix more stable by scaling down
            self.A.weight.data *= 0.9
            
        # Initialize B and C with small values
        nn.init.normal_(self.B.weight, std=0.1)
        nn.init.normal_(self.C.weight, std=0.1)
        nn.init.normal_(self.D.weight, std=0.01)
        
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state [batch_size, state_dim]
        """
        return torch.zeros(batch_size, self.state_dim, device=self.device)
    
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through state space model.
        
        Args:
            x: Input features [batch_size, input_dim]
            hidden_state: Previous hidden state [batch_size, state_dim]
            
        Returns:
            output: Model output [batch_size, output_dim]
            next_hidden_state: Updated hidden state [batch_size, state_dim]
        """
        # Validate inputs
        ShapeValidator.validate_tensor(x, 2, "input_x")
        ShapeValidator.validate_tensor(hidden_state, 2, "hidden_state")
        ShapeValidator.validate_batch_consistency(x, hidden_state, names=["x", "hidden_state"])
        ShapeValidator.validate_device_consistency(x, hidden_state, names=["x", "hidden_state"])
        
        # State space equations:
        # s_{t+1} = As_t + Bx_t + nonlinearity(s_t)
        # y_t = Cs_t + Dx_t
        
        # Linear state transition
        linear_state = self.A(hidden_state) + self.B(x)
        
        # Add non-linear component for expressivity
        nonlinear_component = self.state_nonlinearity(hidden_state)
        
        # Combine with residual connection
        next_hidden_state = linear_state + 0.1 * nonlinear_component
        
        # Output computation
        linear_output = self.C(hidden_state) + self.D(x)
        
        # Apply output projection with residual
        output = linear_output + self.output_projection(linear_output)
        
        return output, next_hidden_state
    
    def get_state_info(self, hidden_state: torch.Tensor) -> Dict[str, float]:
        """
        Get diagnostic information about the current state.
        
        Args:
            hidden_state: Current hidden state
            
        Returns:
            Dictionary with state diagnostics
        """
        with torch.no_grad():
            state_norm = torch.norm(hidden_state, dim=-1).mean().item()
            state_var = torch.var(hidden_state, dim=-1).mean().item()
            
            # Compute eigenvalues of A matrix for stability analysis
            try:
                A_eigenvals = torch.linalg.eigvals(self.A.weight)
                # Handle complex eigenvalues safely
                if A_eigenvals.dtype.is_complex:
                    max_eigenval = torch.max(torch.abs(A_eigenvals)).item()
                else:
                    max_eigenval = torch.max(A_eigenvals.real).item()
            except Exception:
                max_eigenval = 1.0  # Default to unstable if computation fails
                
        return {
            'state_norm': state_norm,
            'state_variance': state_var,
            'max_eigenvalue': max_eigenval,
            'is_stable': max_eigenval < 1.0
        }


class TemporalFusion(nn.Module):
    """
    Enhanced Temporal Fusion with automatic dimension alignment.
    Fixes the dimension mismatch issues from v1.0.
    """
    
    def __init__(self, feature_dim: int, temporal_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.device = device
        
        # Dimension aligners for automatic compatibility
        self.feature_aligner = DimensionAligner(output_dim, device)
        self.temporal_aligner = DimensionAligner(output_dim, device)
        
        # Fusion network (now works with aligned dimensions)
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2, device=device),  # Concatenated aligned features
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device)
        )
        
        # Gating mechanism for temporal importance
        self.temporal_gate = nn.Sequential(
            nn.Linear(output_dim, 1, device=device),  # Use aligned temporal dim
            nn.Sigmoid()
        )
        
    def forward(self, current_features: torch.Tensor, 
               temporal_context: torch.Tensor) -> torch.Tensor:
        """
        Fuse current and temporal features with automatic alignment.
        
        Args:
            current_features: Current input features [batch_size, feature_dim]
            temporal_context: Temporal context [batch_size, temporal_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Validate inputs
        ShapeValidator.validate_tensor(current_features, 2, "current_features")
        ShapeValidator.validate_tensor(temporal_context, 2, "temporal_context")
        ShapeValidator.validate_batch_consistency(
            current_features, temporal_context, 
            names=["current_features", "temporal_context"]
        )
        ShapeValidator.validate_device_consistency(
            current_features, temporal_context,
            names=["current_features", "temporal_context"]
        )
        
        # Align dimensions
        aligned_features = self.feature_aligner(current_features, "current_features")
        aligned_temporal = self.temporal_aligner(temporal_context, "temporal_context")
        
        # Compute temporal importance
        temporal_importance = self.temporal_gate(aligned_temporal)
        weighted_temporal = aligned_temporal * temporal_importance
        
        # Concatenate aligned features
        combined = torch.cat([aligned_features, weighted_temporal], dim=-1)
        fused_output = self.fusion_net(combined)
        
        return fused_output


class FastTacticalProcessor(nn.Module):
    """
    Layer 2: Complete Fast Tactical Processor (Enhanced v2.0)
    
    Combines SSM processing with temporal fusion for real-time decision making.
    Achieves O(n) complexity while maintaining high expressivity.
    
    v2.0 Enhancements:
    - Fixed dimension alignment issues
    - Added comprehensive validation
    - Improved error handling
    - Better device consistency
    """
    
    def __init__(self, feature_dim: int = 256, state_dim: int = 128, 
                 prediction_dim: int = 64, device: str = 'cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.prediction_dim = prediction_dim
        self.device = device
        
        # Core SSM processor
        self.ssm = StateSpaceProcessor(
            input_dim=feature_dim,
            state_dim=state_dim,
            output_dim=prediction_dim,
            device=device
        )
        
        # Enhanced temporal fusion with alignment
        self.temporal_fusion = TemporalFusion(
            feature_dim=feature_dim,
            temporal_dim=prediction_dim,  # Assume temporal context is same as prediction dim
            output_dim=prediction_dim,
            device=device
        )
        
        # Prediction head for tactical decisions
        self.tactical_head = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim * 2, device=device),
            nn.ReLU(),
            nn.Linear(prediction_dim * 2, prediction_dim, device=device),
            nn.LayerNorm(prediction_dim, device=device)
        )
        
    def forward(self, features: torch.Tensor, hidden_state: torch.Tensor, 
               temporal_buffer: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process features with O(n) tactical reasoning.
        
        Args:
            features: Spatial features from Layer 1 [batch_size, feature_dim]
            hidden_state: Previous hidden state [batch_size, state_dim]
            temporal_buffer: Optional temporal context [batch_size, ?]
            
        Returns:
            tactical_prediction: Fast tactical prediction [batch_size, prediction_dim]
            next_hidden_state: Updated hidden state [batch_size, state_dim]
            processing_info: Diagnostic information
        """
        # Validate inputs
        ShapeValidator.validate_tensor(features, 2, "features")
        ShapeValidator.validate_tensor(hidden_state, 2, "hidden_state")
        ShapeValidator.validate_batch_consistency(features, hidden_state, names=["features", "hidden_state"])
        ShapeValidator.validate_device_consistency(features, hidden_state, names=["features", "hidden_state"])
        
        # Core SSM processing - O(n) complexity
        ssm_output, next_hidden_state = self.ssm(features, hidden_state)
        
        # Temporal fusion if context is available
        if temporal_buffer is not None:
            # Validate temporal buffer
            ShapeValidator.validate_tensor(temporal_buffer, 2, "temporal_buffer")
            ShapeValidator.validate_batch_consistency(
                features, temporal_buffer, names=["features", "temporal_buffer"]
            )
            ShapeValidator.validate_device_consistency(
                features, temporal_buffer, names=["features", "temporal_buffer"]
            )
            
            fused_output = self.temporal_fusion(features, temporal_buffer)
            # Combine SSM output with temporal fusion
            combined_output = 0.7 * ssm_output + 0.3 * fused_output
        else:
            combined_output = ssm_output
            
        # Generate tactical prediction
        tactical_prediction = self.tactical_head(combined_output)
        
        # Compute processing diagnostics
        with torch.no_grad():
            state_info = self.ssm.get_state_info(next_hidden_state)
            prediction_norm = torch.norm(tactical_prediction, dim=-1).mean().item()
            
            processing_info = {
                'prediction_norm': prediction_norm,
                'state_stability': state_info['is_stable'],
                'processing_complexity': 'O(n)',
                'temporal_fusion_used': temporal_buffer is not None,
                'input_dim_validated': True,
                'hidden_dim_validated': True,
                'state_info': state_info
            }
        
        return tactical_prediction, next_hidden_state, processing_info
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state for the processor.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state [batch_size, state_dim]
        """
        return self.ssm.init_hidden(batch_size)
    
    def get_temporal_importance(self, features: torch.Tensor, 
                               temporal_context: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights for temporal context.
        
        Args:
            features: Current features
            temporal_context: Historical context
            
        Returns:
            Importance weights
        """
        # Align temporal context first
        aligned_temporal = self.temporal_fusion.temporal_aligner(temporal_context, "temporal_importance")
        return self.temporal_fusion.temporal_gate(aligned_temporal)
