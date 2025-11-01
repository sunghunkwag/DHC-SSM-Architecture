"""
Enhanced DimensionAligner with edge-case handling and device safety.
"""
import torch
import torch.nn as nn
from typing import Optional

class DimensionAligner(nn.Module):
    def __init__(self, target_dim: int, device: str = 'cpu'):
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.projection_layers = nn.ModuleDict()
        
    def forward(self, x: torch.Tensor, source_name: str = "input", allow_expand: bool = True) -> torch.Tensor:
        # Guard: ensure tensor on correct device
        if x.device.type != self.device:
            x = x.to(self.device)
        
        in_dim = x.size(-1)
        
        # Fast path
        if in_dim == self.target_dim:
            return x
        
        # Edge-case: scalar or 1D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Create projection layer per source
        key = f"{source_name}_{in_dim}->{self.target_dim}"
        if key not in self.projection_layers:
            self.projection_layers[key] = nn.Linear(in_dim, self.target_dim, device=self.device)
        
        # Handle very small dims by zero-pad before projection (optional)
        if allow_expand and in_dim < self.target_dim:
            pad = self.target_dim - in_dim
            pad_tensor = torch.zeros(*x.shape[:-1], pad, device=self.device)
            x = torch.cat([x, pad_tensor], dim=-1)
            in_dim = self.target_dim
            # refresh key to avoid mismatch (no need new layer now)
        
        # Final projection
        proj = self.projection_layers[key](x)
        
        # NaN protection
        proj = torch.nan_to_num(proj, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return proj
