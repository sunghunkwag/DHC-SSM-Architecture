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
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: Tuple[Optional[int], ...],
        name: str = "tensor"
    ) -> None:
        """
        Validate tensor matches expected shape (None for any size).
        
        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape (None for flexible dimensions)
            name: Name of tensor for error messages
            
        Raises:
            ValueError: If tensor shape doesn't match expected
        """
        actual_shape = tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{name} has shape {actual_shape}, "
                f"expected {len(expected_shape)} dimensions"
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} is {actual}, expected {expected}. "
                    f"Full shape: {actual_shape}"
                )
    
    @staticmethod
    def validate_batch_compatible(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        name1: str = "tensor1",
        name2: str = "tensor2"
    ) -> None:
        """
        Validate two tensors have compatible batch dimensions.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            name1: Name of first tensor
            name2: Name of second tensor
            
        Raises:
            ValueError: If batch dimensions are incompatible
        """
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError(
                f"{name1} batch size {tensor1.shape[0]} != "
                f"{name2} batch size {tensor2.shape[0]}"
            )


class DimensionAligner(nn.Module):
    """
    Automatically aligns tensor dimensions using learned projections.
    """
    
    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        """
        Initialize dimension aligner.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            device: Device to place module on
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, device=device)
            self.needs_projection = True
        else:
            self.projection = nn.Identity()
            self.needs_projection = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Align tensor to output dimension.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            Aligned tensor [..., output_dim]
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} != expected {self.input_dim}"
            )
        
        return self.projection(x)


class FlexibleConcatenation(nn.Module):
    """
    Concatenates tensors with automatic dimension alignment.
    """
    
    def __init__(self, target_dim: int, device: str = 'cpu'):
        """
        Initialize flexible concatenation.
        
        Args:
            target_dim: Target dimension for all inputs
            device: Device to place module on
        """
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.aligners = nn.ModuleDict()
    
    def forward(self, tensors: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
        """
        Concatenate tensors with automatic alignment.
        
        Args:
            tensors: List of tensors to concatenate
            dim: Dimension along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        aligned_tensors = []
        
        for i, tensor in enumerate(tensors):
            input_dim = tensor.shape[dim]
            
            # Create or retrieve aligner for this dimension
            key = f"aligner_{input_dim}"
            if key not in self.aligners:
                self.aligners[key] = DimensionAligner(
                    input_dim, self.target_dim, self.device
                )
            
            # Align tensor
            aligned = self.aligners[key](tensor)
            aligned_tensors.append(aligned)
        
        # Concatenate aligned tensors
        return torch.cat(aligned_tensors, dim=dim)


def ensure_batch_dim(tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """
    Ensure tensor has batch dimension.
    
    Args:
        tensor: Input tensor
        batch_size: Batch size to add if missing
        
    Returns:
        Tensor with batch dimension
    """
    if len(tensor.shape) == 0:
        # Scalar tensor
        return tensor.unsqueeze(0).expand(batch_size)
    elif tensor.shape[0] == batch_size:
        # Already has correct batch dimension
        return tensor
    else:
        # Add batch dimension
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)


def broadcast_to_batch(
    tensor: torch.Tensor,
    batch_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Broadcast tensor to match batch size.
    
    Args:
        tensor: Input tensor
        batch_size: Target batch size
        device: Device to place result on
        
    Returns:
        Broadcasted tensor with batch dimension
    """
    if device is None:
        device = tensor.device
        
    if len(tensor.shape) == 0:
        # Scalar
        return tensor.unsqueeze(0).expand(batch_size).to(device)
    elif tensor.shape[0] == 1:
        # Single batch, expand
        return tensor.expand(batch_size, *tensor.shape[1:]).to(device)
    elif tensor.shape[0] == batch_size:
        # Already correct
        return tensor.to(device)
    else:
        raise ValueError(
            f"Cannot broadcast tensor with batch size {tensor.shape[0]} "
            f"to batch size {batch_size}"
        )