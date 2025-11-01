"""
Pareto Optimizer: Add NaN protection and temperature scheduling
"""
import torch
import torch.nn.functional as F
from torch import nn

class ParetoOptimizer(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, num_objectives: int = 4, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        self.device = device
        self.register_buffer('temperature', torch.tensor(2.0, device=device))
        
        self.objective_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2, device=device),
                nn.ReLU(),
                nn.Linear(input_dim // 2, action_dim, device=device)
            ) for _ in range(num_objectives)
        ])
        self.weight_optimizer = nn.Sequential(
            nn.Linear(num_objectives, num_objectives * 2, device=device),
            nn.ReLU(),
            nn.Linear(num_objectives * 2, num_objectives, device=device),
        )
    
    def step_temperature(self, decay: float = 0.99, min_temp: float = 0.5):
        with torch.no_grad():
            self.temperature.mul_(decay)
            if self.temperature.item() < min_temp:
                self.temperature.fill_(min_temp)
    
    def compute_pareto_weights(self, error_vectors: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        safe_errors = torch.nan_to_num(error_vectors, posinf=1e6, neginf=1e6)
        safe_errors = safe_errors + epsilon
        normalized = F.normalize(safe_errors, dim=-1)
        raw = self.weight_optimizer(normalized)
        weights = F.softmax(raw / self.temperature, dim=-1)
        # Floor to avoid exact zeros
        floor = 0.01
        weights = weights * (1 - floor * self.num_objectives) + floor
        # Renormalize to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = torch.nan_to_num(weights, nan=1.0/self.num_objectives)
        return weights
    
    def forward(self, state_input: torch.Tensor, error_vectors: torch.Tensor):
        # Compute actions for each objective
        objective_actions = []
        for net in self.objective_networks:
            obj_action = net(state_input)
            objective_actions.append(obj_action)
        objective_actions = torch.stack(objective_actions, dim=1)
        
        # Compute stable weights
        weights = self.compute_pareto_weights(error_vectors)
        
        # Weighted sum
        action = torch.sum(objective_actions * weights.unsqueeze(-1), dim=1)
        return action, weights
