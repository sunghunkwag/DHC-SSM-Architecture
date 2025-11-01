"""
Layer 3: Slow Strategic Reasoner - Enhanced v2.1

Based on CausalReasoningModule from Autonomous-Self-Organizing-AI.
Provides causal relationship discovery and strategic planning through GNN processing.

Enhancements v2.1:
- Comprehensive error handling and fallback mechanisms
- torch_geometric import safety with fallback GNN implementation
- Enhanced graph stability with adaptive edge thresholding
- Device consistency and NaN protection
- Improved numerical stability in causal computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

# Conditional import with comprehensive fallback
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    import torch_geometric.data as pyg_data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Using fallback GNN implementation.")

# Import validation utilities with fallback
try:
    from ..utils.shape_validator import ShapeValidator, DimensionAligner
    from ..utils.device import ensure_device
except ImportError:
    class ShapeValidator:
        @staticmethod
        def validate_tensor(*args, **kwargs): pass
        @staticmethod
        def validate_batch_compatible(*args, **kwargs): pass
    
    def ensure_device(tensor, device): return tensor.to(device)


class CausalGraphBuilder(nn.Module):
    """
    Builds causal graphs from state sequences with enhanced stability.
    """
    
    def __init__(self, state_dim: int, max_nodes: int = 10, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Node feature extraction with dropout
        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(state_dim // 2, state_dim // 4, device=device),
            nn.LayerNorm(state_dim // 4, device=device)
        )
        
        # Edge prediction with enhanced stability
        self.edge_predictor = nn.Sequential(
            nn.Linear(state_dim // 2, 64, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Sigmoid()
        )
        
        # Causal strength with clipping
        self.strength_estimator = nn.Sequential(
            nn.Linear(state_dim // 2, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Tanh()
        )
        
    def forward(self, state_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build causal graph with comprehensive error handling.
        """
        try:
            # Device and shape safety
            state_sequence = ensure_device(state_sequence, self.device)
            ShapeValidator.validate_tensor(state_sequence, 3, "state_sequence")
            
            batch_size, seq_len, _ = state_sequence.shape
            
            # Adaptive node selection based on sequence length
            if seq_len < self.max_nodes:
                # Pad if too short
                padding = torch.zeros(batch_size, self.max_nodes - seq_len, self.state_dim, device=self.device)
                state_sequence = torch.cat([state_sequence, padding], dim=1)
                seq_len = self.max_nodes
            
            # Extract representative nodes
            node_indices = torch.linspace(0, seq_len-1, self.max_nodes, dtype=torch.long, device=self.device)
            nodes = state_sequence[:, node_indices]
            
            # Encode with NaN protection
            node_features = self.node_encoder(nodes)
            node_features = torch.nan_to_num(node_features, nan=0.0)
            
            # Build adjacency and strength matrices with stability
            batch_adjacency = []
            batch_strengths = []
            
            for batch_idx in range(batch_size):
                adj_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=self.device)
                str_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=self.device)
                
                for i in range(self.max_nodes):
                    for j in range(self.max_nodes):
                        if i != j:  # No self-loops
                            try:
                                node_pair = torch.cat([
                                    node_features[batch_idx, i], 
                                    node_features[batch_idx, j]
                                ], dim=0)
                                
                                # Predictions with error handling
                                edge_prob = self.edge_predictor(node_pair.unsqueeze(0)).squeeze()
                                causal_strength = self.strength_estimator(node_pair.unsqueeze(0)).squeeze()
                                
                                # NaN protection
                                edge_prob = torch.nan_to_num(edge_prob, nan=0.1)
                                causal_strength = torch.nan_to_num(causal_strength, nan=0.0)
                                
                                adj_matrix[i, j] = edge_prob
                                str_matrix[i, j] = causal_strength
                                
                            except Exception:
                                # Safe fallback values
                                adj_matrix[i, j] = 0.1
                                str_matrix[i, j] = 0.0
                
                batch_adjacency.append(adj_matrix)
                batch_strengths.append(str_matrix)
            
            adjacency_tensor = torch.stack(batch_adjacency)
            causal_strengths_tensor = torch.stack(batch_strengths)
            
            # Apply edge thresholding for graph stability
            edge_threshold = 0.3
            adjacency_tensor = torch.where(
                adjacency_tensor > edge_threshold, 
                adjacency_tensor, 
                torch.zeros_like(adjacency_tensor)
            )
            
            return node_features, adjacency_tensor, causal_strengths_tensor
            
        except Exception as e:
            # Complete fallback
            batch_size = state_sequence.size(0) if state_sequence is not None else 1
            return (
                torch.zeros(batch_size, self.max_nodes, self.state_dim // 4, device=self.device),
                torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=self.device),
                torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=self.device)
            )


class FallbackGNNReasoner(nn.Module):
    """
    Fallback GNN when torch_geometric is unavailable.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        self.node_processor = nn.Sequential(
            nn.Linear(node_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device)
        )
        
        self.global_reasoner = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device)
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.node_processor(node_features)
        graph_representation = torch.mean(x, dim=0, keepdim=True)
        return self.global_reasoner(graph_representation)


class GraphNeuralReasoner(nn.Module):
    """
    Graph neural network with torch_geometric fallback capability.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Real GNN layers
            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(GCNConv(node_dim, hidden_dim).to(device))
            
            for _ in range(num_layers - 2):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim).to(device))
            
            self.gnn_layers.append(GCNConv(hidden_dim, output_dim).to(device))
            self.use_fallback = False
            
        else:
            # Fallback implementation
            self.fallback_gnn = FallbackGNNReasoner(node_dim, hidden_dim, output_dim, device)
            self.use_fallback = True
        
        # Global reasoning head
        self.global_reasoner = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device),
            nn.LayerNorm(output_dim, device=device)
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
               batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GNN reasoning with comprehensive error handling.
        """
        try:
            if self.use_fallback or not TORCH_GEOMETRIC_AVAILABLE:
                return self.fallback_gnn(node_features, edge_index)
            
            x = node_features
            
            # GNN layers with error handling
            for i, layer in enumerate(self.gnn_layers):
                try:
                    x = layer(x, edge_index)
                    if i < len(self.gnn_layers) - 1:
                        x = F.relu(x)
                except Exception:
                    # Skip problematic layer
                    continue
            
            # Global pooling with fallback
            try:
                if batch is not None and TORCH_GEOMETRIC_AVAILABLE:
                    graph_representation = global_mean_pool(x, batch)
                else:
                    graph_representation = torch.mean(x, dim=0, keepdim=True)
            except Exception:
                graph_representation = torch.mean(x, dim=0, keepdim=True)
            
            strategic_output = self.global_reasoner(graph_representation)
            return strategic_output
            
        except Exception:
            # Complete fallback
            return self.fallback_gnn(node_features, edge_index)


class GoalFormationModule(nn.Module):
    """
    Strategic goal formation with enhanced stability.
    """
    
    def __init__(self, causal_dim: int, goal_dim: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        self.goal_generator = nn.Sequential(
            nn.Linear(causal_dim, causal_dim * 2, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(causal_dim * 2, goal_dim, device=device),
            nn.LayerNorm(goal_dim, device=device)
        )
        
        self.priority_network = nn.Sequential(
            nn.Linear(causal_dim, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Softplus()
        )
        
    def forward(self, causal_reasoning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate goals with comprehensive error protection.
        """
        try:
            causal_reasoning = ensure_device(causal_reasoning, self.device)
            
            goals = self.goal_generator(causal_reasoning)
            priorities = self.priority_network(causal_reasoning)
            
            # NaN and stability protection
            goals = torch.nan_to_num(goals, nan=0.0)
            priorities = torch.nan_to_num(priorities, nan=1.0)
            
            return goals, priorities
            
        except Exception:
            # Safe fallback
            batch_size = causal_reasoning.size(0)
            goal_dim = self.goal_generator[-2].out_features
            
            return (
                torch.zeros(batch_size, goal_dim, device=self.device),
                torch.ones(batch_size, 1, device=self.device)
            )


class SlowStrategicReasoner(nn.Module):
    """
    Layer 3: Enhanced Strategic Reasoner with comprehensive robustness.
    
    Performs causal analysis with extensive error handling and fallback mechanisms.
    """
    
    def __init__(self, state_dim: int = 128, causal_dim: int = 64, goal_dim: int = 32, 
                 max_nodes: int = 10, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.causal_dim = causal_dim
        self.goal_dim = goal_dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Enhanced components
        self.graph_builder = CausalGraphBuilder(state_dim, max_nodes, device)
        self.gnn_reasoner = GraphNeuralReasoner(
            node_dim=state_dim // 4,
            hidden_dim=causal_dim,
            output_dim=causal_dim,
            device=device
        )
        self.goal_former = GoalFormationModule(causal_dim, goal_dim, device)
        
        # Strategic prediction with stability
        self.strategic_head = nn.Sequential(
            nn.Linear(causal_dim + goal_dim, causal_dim, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(causal_dim, causal_dim, device=device),
            nn.LayerNorm(causal_dim, device=device)
        )
        
    def forward(self, state_buffer: torch.Tensor, step_count: int, 
                async_interval: int = 5) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Strategic reasoning with comprehensive error handling.
        """
        # Asynchronous processing check
        if step_count % async_interval != 0:
            return None, None, None
            
        try:
            # Safety checks
            state_buffer = ensure_device(state_buffer, self.device)
            ShapeValidator.validate_tensor(state_buffer, 3, "state_buffer")
            
            batch_size = state_buffer.size(0)
            
            # Build causal graph with error handling
            node_features, adjacency_matrix, causal_strengths = self.graph_builder(state_buffer)
            
            # Process each batch with individual error handling
            strategic_outputs = []
            causal_infos = []
            
            for batch_idx in range(batch_size):
                try:
                    # Adaptive edge thresholding for stability
                    adj = adjacency_matrix[batch_idx]
                    
                    # Dynamic threshold based on graph characteristics
                    edge_threshold = max(0.3, torch.quantile(adj, 0.7).item())
                    
                    # Find significant edges
                    edge_mask = adj > edge_threshold
                    edge_indices = torch.nonzero(edge_mask, as_tuple=False).t()
                    
                    if edge_indices.size(1) > 0 and not self.gnn_reasoner.use_fallback:
                        # GNN reasoning with real edges
                        nodes = node_features[batch_idx]
                        strategic_reasoning = self.gnn_reasoner(nodes, edge_indices)
                    else:
                        # Fallback: mean pooling + MLP
                        nodes = node_features[batch_idx]
                        strategic_reasoning = torch.mean(nodes, dim=0, keepdim=True)
                        strategic_reasoning = self.gnn_reasoner.global_reasoner(strategic_reasoning)
                    
                    strategic_outputs.append(strategic_reasoning)
                    
                    # Collect diagnostics
                    with torch.no_grad():
                        num_edges = edge_indices.size(1) if edge_indices.size(1) > 0 else 0
                        avg_strength = causal_strengths[batch_idx].mean().item()
                        density = num_edges / max(1, self.max_nodes * (self.max_nodes - 1))
                        
                        causal_infos.append({
                            'num_causal_edges': num_edges,
                            'avg_causal_strength': avg_strength,
                            'graph_density': density,
                            'edge_threshold': edge_threshold,
                            'reasoning_successful': True
                        })
                        
                except Exception as batch_error:
                    # Batch-level fallback
                    strategic_outputs.append(torch.zeros(1, self.causal_dim, device=self.device))
                    causal_infos.append({
                        'batch_error': str(batch_error),
                        'fallback_used': True,
                        'reasoning_successful': False
                    })
            
            # Safely stack outputs
            try:
                strategic_prediction = torch.stack(strategic_outputs, dim=0)
                if strategic_prediction.dim() == 3:
                    strategic_prediction = strategic_prediction.squeeze(1)
            except Exception:
                strategic_prediction = torch.zeros(batch_size, self.causal_dim, device=self.device)
            
            # Generate strategic goals with protection
            try:
                goals, priorities = self.goal_former(strategic_prediction)
            except Exception:
                goals = torch.zeros(batch_size, self.goal_dim, device=self.device)
                priorities = torch.ones(batch_size, 1, device=self.device)
            
            # Final strategic output with error handling
            try:
                combined_input = torch.cat([strategic_prediction, goals], dim=-1)
                final_strategic_output = self.strategic_head(combined_input)
                final_strategic_output = torch.nan_to_num(final_strategic_output, nan=0.0)
            except Exception:
                final_strategic_output = torch.zeros(batch_size, self.causal_dim, device=self.device)
            
            # Aggregate causal information
            aggregated_info = {
                'processing_step': step_count,
                'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE,
                'successful_batches': sum(1 for info in causal_infos if info.get('reasoning_successful', False)),
                'failed_batches': sum(1 for info in causal_infos if not info.get('reasoning_successful', True)),
                'avg_edges_per_graph': np.mean([info.get('num_causal_edges', 0) for info in causal_infos]),
                'avg_causal_strength': np.mean([info.get('avg_causal_strength', 0) for info in causal_infos]),
                'avg_graph_density': np.mean([info.get('graph_density', 0) for info in causal_infos]),
            }
            
            return final_strategic_output, goals, aggregated_info
            
        except Exception as e:
            # Complete system fallback
            batch_size = state_buffer.size(0) if state_buffer is not None else 1
            
            return (
                torch.zeros(batch_size, self.causal_dim, device=self.device),
                torch.zeros(batch_size, self.goal_dim, device=self.device),
                {
                    'complete_fallback': True,
                    'error': str(e),
                    'reasoning_successful': False
                }
            )
    
    def analyze_causal_structure(self, state_buffer: torch.Tensor) -> Dict[str, Any]:
        """
        Causal structure analysis with comprehensive error handling.
        """
        try:
            with torch.no_grad():
                state_buffer = ensure_device(state_buffer, self.device)
                
                node_features, adjacency_matrix, causal_strengths = self.graph_builder(state_buffer)
                
                return {
                    'causal_graph_nodes': node_features.shape[1],
                    'avg_node_activation': torch.mean(node_features).item(),
                    'causal_edge_density': torch.mean(adjacency_matrix > 0.3).item(),
                    'strongest_connection': torch.max(causal_strengths).item(),
                    'weakest_connection': torch.min(causal_strengths).item(),
                    'network_complexity': torch.sum(adjacency_matrix > 0.3).item(),
                    'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE,
                    'analysis_successful': True
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'analysis_successful': False,
                'fallback_analysis': True,
                'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE
            }


class StrategicPlanningModule(nn.Module):
    """
    Long-term strategic planning with enhanced stability.
    """
    
    def __init__(self, causal_dim: int, goal_dim: int, plan_horizon: int = 10, device: str = 'cpu'):
        super().__init__()
        self.causal_dim = causal_dim
        self.goal_dim = goal_dim
        self.plan_horizon = plan_horizon
        self.device = device
        
        # Enhanced planner
        self.planner = nn.Sequential(
            nn.Linear(causal_dim + goal_dim, causal_dim * 2, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(causal_dim * 2, causal_dim, device=device),
            nn.ReLU(),
            nn.Linear(causal_dim, plan_horizon * goal_dim, device=device)
        )
        
        # Plan evaluator
        self.evaluator = nn.Sequential(
            nn.Linear(plan_horizon * goal_dim, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 1, device=device),
            nn.Sigmoid()
        )
        
    def forward(self, causal_reasoning: torch.Tensor, goals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate strategic plans with error protection.
        """
        try:
            causal_reasoning = ensure_device(causal_reasoning, self.device)
            goals = ensure_device(goals, self.device)
            
            combined_input = torch.cat([causal_reasoning, goals], dim=-1)
            
            plan_flat = self.planner(combined_input)
            strategic_plan = plan_flat.view(-1, self.plan_horizon, self.goal_dim)
            
            plan_quality = self.evaluator(plan_flat)
            
            # NaN protection
            strategic_plan = torch.nan_to_num(strategic_plan, nan=0.0)
            plan_quality = torch.nan_to_num(plan_quality, nan=0.5)
            
            return strategic_plan, plan_quality
            
        except Exception:
            # Safe fallback planning
            batch_size = causal_reasoning.size(0)
            return (
                torch.zeros(batch_size, self.plan_horizon, self.goal_dim, device=self.device),
                torch.ones(batch_size, 1, device=self.device) * 0.5
            )