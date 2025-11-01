"""
GradientComputer: add robust fallback for None grads and clipping utilities
"""
import torch
from torch import nn

class GradientComputer(nn.Module):
    def __init__(self, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.num_objectives = num_objectives
        self.device = device
        self.clip_value = 1.0
    
    def set_clip(self, value: float):
        self.clip_value = float(value)
    
    def compute_multi_objective_gradients(self, error_vectors, parameters):
        # Compute per-objective gradients with allow_unused
        objective_grads = {}
        for name, signal in error_vectors.items():
            loss = torch.mean(signal)
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
            )
            # Replace None with zeros
            fixed = []
            for p, g in zip(parameters, grads):
                if g is None:
                    fixed.append(torch.zeros_like(p))
                else:
                    fixed.append(torch.clamp(g, -self.clip_value, self.clip_value))
            objective_grads[name] = fixed
        
        # Equal weights for now (could be replaced by learned weights)
        combined = []
        for idx in range(len(parameters)):
            acc = torch.zeros_like(parameters[idx])
            for name in error_vectors.keys():
                acc += objective_grads[name][idx] / self.num_objectives
            combined.append(acc)
        
        return {
            'combined_gradients': combined,
            'objective_gradients': objective_grads
        }
