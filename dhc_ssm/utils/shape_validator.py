"""
Shape Validation Utilities

Provides robust tensor shape validation and automatic dimension alignment
to prevent runtime errors from dimension mismatches.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union

class ShapeValidator:
    """
    Validates and reports tensor shapes with informative error messages.
    """
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_dims: int,
        name: str = "tensor",
        allow_batch: bool = True
    ) -> None:
        """
        Validate tensor has expected number of dimensions.
        
        Args:
            tensor: Tensor to validate
            expected_dims: Expected number of dimensions
            name: Name of tensor for error messages
            allow_batch: If True, allow one extra dimension for batch
            
        Raises:
            ValueError: If tensor dimensions don't match expected
        """
        actual_dims = len(tensor.shape)
        if allow_batch:
            if actual_dims not in [expected_dims, expected_dims + 1]:
                raise ValueError(
                    f"{name} has {actual_dims} dimensions, "
                    f"expected {expected_dims} or {expected_dims + 1} (with batch). "
                    f"Shape: {tensor.shape}"
                )
        else:
            if actual_dims != expected_dims:
                raise ValueError(
                    f"{name} has {actual_dims} dimensions, "
                    f"expected {expected_dims}. "
                    f"Shape: {tensor.shape}"
                )
    
    @staticmethod
    def validate_batch_consistency(
        *tensors: torch.Tensor,
        names: Optional[List[str]] = None
    ) -> None:
        """
        Validate that all tensors have the same batch size.
        
        Args:
            *tensors: Tensors to validate
            names: Optional names for tensors
        """
        if not tensors:
            return
            
        batch_sizes = [t.size(0) for t in tensors]
        if len(set(batch_sizes)) > 1:
            tensor_info = []
            for i, (tensor, batch_size) in enumerate(zip(tensors, batch_sizes)):
                name = names[i] if names and i < len(names) else f"tensor_{i}"
                tensor_info.append(f"{name}: batch_size={batch_size}, shape={tensor.shape}")
            
            raise ValueError(
                f"Batch size mismatch detected:\n" + "\n".join(tensor_info)
            )
    
    @staticmethod
    def validate_device_consistency(
        *tensors: torch.Tensor,
        names: Optional[List[str]] = None
    ) -> None:
        """
        Validate that all tensors are on the same device.
        
        Args:
            *tensors: Tensors to validate
            names: Optional names for tensors
        """
        if not tensors:
            return
            
        devices = [str(t.device) for t in tensors]
        if len(set(devices)) > 1:
            tensor_info = []
            for i, (tensor, device) in enumerate(zip(tensors, devices)):
                name = names[i] if names and i < len(names) else f"tensor_{i}"
                tensor_info.append(f"{name}: device={device}, shape={tensor.shape}")
            
            raise ValueError(
                f"Device mismatch detected:\n" + "\n".join(tensor_info)
            )


class DimensionAligner(nn.Module):
    """
    Automatically aligns tensor dimensions for compatibility.
    """
    
    def __init__(self, target_dim: int, device: str = 'cpu'):
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.projection_layers = nn.ModuleDict()
        
    def forward(self, x: torch.Tensor, source_name: str = "input") -> torch.Tensor:
        """
        Align tensor to target dimension.
        
        Args:
            x: Input tensor [..., input_dim]
            source_name: Name for caching projection layer
            
        Returns:
            Aligned tensor [..., target_dim]
        """
        input_dim = x.size(-1)
        
        if input_dim == self.target_dim:
            return x
            
        # Create or get projection layer
        if source_name not in self.projection_layers:
            self.projection_layers[source_name] = nn.Linear(
                input_dim, self.target_dim, device=self.device
            )
            
        return self.projection_layers[source_name](x)


class FlexibleConcatenation(nn.Module):
    """
    Concatenates tensors with automatic dimension alignment.
    """
    
    def __init__(self, target_dim: int, device: str = 'cpu'):
        super().__init__()
        self.target_dim = target_dim
        self.aligner = DimensionAligner(target_dim, device)
        
    def forward(self, *tensors: torch.Tensor, names: Optional[List[str]] = None) -> torch.Tensor:
        """
        Concatenate tensors with dimension alignment.
        
        Args:
            *tensors: Tensors to concatenate
            names: Optional names for each tensor
            
        Returns:
            Concatenated tensor
        """
        if not tensors:
            raise ValueError("At least one tensor must be provided")
            
        # Validate batch consistency
        ShapeValidator.validate_batch_consistency(*tensors, names=names)
        ShapeValidator.validate_device_consistency(*tensors, names=names)
        
        # Align dimensions
        aligned_tensors = []
        for i, tensor in enumerate(tensors):
            name = names[i] if names and i < len(names) else f"tensor_{i}"
            aligned = self.aligner(tensor, name)
            aligned_tensors.append(aligned)
            
        # Concatenate along last dimension
        return torch.cat(aligned_tensors, dim=-1)


class ShapeReporter:
    """
    Reports tensor shapes for debugging.
    """
    
    @staticmethod
    def report_shapes(stage_name: str, **tensors: torch.Tensor) -> None:
        """
        Report tensor shapes for debugging.
        
        Args:
            stage_name: Name of the processing stage
            **tensors: Named tensors to report
        """
        print(f"\n=== {stage_name} Shape Report ===")
        for name, tensor in tensors.items():
            if tensor is not None:
                print(f"{name}: {tensor.shape} (device: {tensor.device})")
            else:
                print(f"{name}: None")
        print("=" * (len(stage_name) + 20))


def validate_dhc_ssm_flow(
    spatial_features: torch.Tensor,
    tactical_prediction: torch.Tensor,
    strategic_prediction: Optional[torch.Tensor],
    final_prediction: torch.Tensor,
    error_vectors: dict
) -> Dict[str, bool]:
    """
    Validate complete DHC-SSM architecture flow.
    
    Args:
        spatial_features: Output from Layer 1
        tactical_prediction: Output from Layer 2
        strategic_prediction: Output from Layer 3 (can be None)
        final_prediction: Fused prediction
        error_vectors: Intrinsic error signals
        
    Returns:
        Validation results
    """
    results = {}
    
    try:
        # Validate spatial features
        ShapeValidator.validate_tensor(spatial_features, 2, "spatial_features")
        results['spatial_valid'] = True
    except ValueError as e:
        print(f"Spatial validation failed: {e}")
        results['spatial_valid'] = False
        
    try:
        # Validate tactical prediction
        ShapeValidator.validate_tensor(tactical_prediction, 2, "tactical_prediction")
        results['tactical_valid'] = True
    except ValueError as e:
        print(f"Tactical validation failed: {e}")
        results['tactical_valid'] = False
        
    try:
        # Validate strategic prediction (if exists)
        if strategic_prediction is not None:
            ShapeValidator.validate_tensor(strategic_prediction, 2, "strategic_prediction")
            results['strategic_valid'] = True
        else:
            results['strategic_valid'] = True  # None is valid
    except ValueError as e:
        print(f"Strategic validation failed: {e}")
        results['strategic_valid'] = False
        
    try:
        # Validate final prediction
        ShapeValidator.validate_tensor(final_prediction, 2, "final_prediction")
        results['final_valid'] = True
    except ValueError as e:
        print(f"Final prediction validation failed: {e}")
        results['final_valid'] = False
        
    try:
        # Validate error vectors
        for name, error in error_vectors.items():
            ShapeValidator.validate_tensor(error, 2, f"error_{name}")
        results['errors_valid'] = True
    except ValueError as e:
        print(f"Error vector validation failed: {e}")
        results['errors_valid'] = False
        
    # Overall validation
    results['overall_valid'] = all([
        results['spatial_valid'],
        results['tactical_valid'], 
        results['strategic_valid'],
        results['final_valid'],
        results['errors_valid']
    ])
    
    return results