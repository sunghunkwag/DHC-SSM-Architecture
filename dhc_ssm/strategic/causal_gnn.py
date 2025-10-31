"""
Layer 3: Slow Strategic Reasoner

Based on CausalReasoningModule from Autonomous-Self-Organizing-AI.
Provides causal relationship discovery and strategic planning through GNN processing.

Key Features:
- Graph neural network causal reasoning
- Asynchronous processing (every N steps)
- Causal structure learning
- Strategic goal formation
- "Why does this happen?" analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch_geometric.data as pyg_data


class CausalGraphBuilder(nn.Module):
    """
    Builds causal graphs from state sequences.
    Discovers "what causes what" relationships in the data.
    """
    
    def __init__(self, state_dim: int, max_nodes: int = 10, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Node feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(state_dim // 2, state_dim // 4, device=device),
            nn.LayerNorm(state_dim // 4, device=device)
        )
        
        # Edge prediction network for causal relationships
        self.edge_predictor = nn.Sequential(
            nn.Linear(state_dim // 2, 64, device=device),  # Concatenated node pairs
            nn.ReLU(),
            nn.Linear(64, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Sigmoid()  # Probability of causal edge
        )
        
        # Causal strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(state_dim // 2, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Tanh()  # Causal strength [-1, 1]
        )
        
    def forward(self, state_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build causal graph from state sequence.
        
        Args:
            state_sequence: Sequence of states [batch_size, seq_len, state_dim]
            
        Returns:
            node_features: Graph node features [batch_size, max_nodes, node_dim]
            adjacency_matrix: Causal adjacency matrix [batch_size, max_nodes, max_nodes]
            causal_strengths: Edge strengths [batch_size, max_nodes, max_nodes]
        """
        batch_size, seq_len, _ = state_sequence.shape
        
        # Extract representative nodes from sequence (using clustering-like approach)
        # For simplicity, we'll use uniform sampling of the sequence
        node_indices = torch.linspace(0, seq_len-1, self.max_nodes, dtype=torch.long, device=self.device)
        nodes = state_sequence[:, node_indices]  # [batch_size, max_nodes, state_dim]
        
        # Encode nodes
        node_features = self.node_encoder(nodes)  # [batch_size, max_nodes, node_dim]
        
        # Build adjacency matrix and causal strengths
        batch_adjacency = []
        batch_strengths = []
        
        for batch_idx in range(batch_size):
            adjacency_row = []
            strength_row = []
            
            for i in range(self.max_nodes):
                adj_col = []
                strength_col = []
                
                for j in range(self.max_nodes):
                    if i != j:  # No self-loops
                        # Concatenate node pair for edge prediction
                        node_pair = torch.cat([node_features[batch_idx, i], node_features[batch_idx, j]], dim=0)
                        
                        # Predict edge existence
                        edge_prob = self.edge_predictor(node_pair.unsqueeze(0))
                        
                        # Predict causal strength
                        causal_strength = self.strength_estimator(node_pair.unsqueeze(0))
                        
                        adj_col.append(edge_prob)
                        strength_col.append(causal_strength)
                    else:
                        adj_col.append(torch.tensor(0.0, device=self.device))
                        strength_col.append(torch.tensor(0.0, device=self.device))
                        
                adjacency_row.append(torch.stack(adj_col))
                strength_row.append(torch.stack(strength_col))
                
            batch_adjacency.append(torch.stack(adjacency_row))
            batch_strengths.append(torch.stack(strength_row))
            
        adjacency_matrix = torch.stack(batch_adjacency)
        causal_strengths = torch.stack(batch_strengths)
        
        return node_features, adjacency_matrix.squeeze(-1), causal_strengths.squeeze(-1)


class GraphNeuralReasoner(nn.Module):
    """
    Graph neural network for causal reasoning on discovered graph structures.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        
        # GNN layers for causal reasoning
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(GCNConv(node_dim, hidden_dim).to(device))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim).to(device))
            
        # Output layer
        self.gnn_layers.append(GCNConv(hidden_dim, output_dim).to(device))
        
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
        Perform causal reasoning on graph structure.
        
        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Strategic reasoning output [batch_size, output_dim]
        """
        x = node_features
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                
        # Global pooling to get graph-level representation
        if batch is not None:
            graph_representation = global_mean_pool(x, batch)
        else:
            graph_representation = torch.mean(x, dim=0, keepdim=True)
            
        # Apply global reasoning
        strategic_output = self.global_reasoner(graph_representation)
        
        return strategic_output


class GoalFormationModule(nn.Module):
    """
    Forms strategic goals based on causal understanding.
    """
    
    def __init__(self, causal_dim: int, goal_dim: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Goal generation network
        self.goal_generator = nn.Sequential(
            nn.Linear(causal_dim, causal_dim * 2, device=device),
            nn.ReLU(),
            nn.Linear(causal_dim * 2, goal_dim, device=device),
            nn.LayerNorm(goal_dim, device=device)
        )
        
        # Goal priority network
        self.priority_network = nn.Sequential(
            nn.Linear(causal_dim, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, 1, device=device),
            nn.Softplus()  # Ensure positive priorities
        )
        
    def forward(self, causal_reasoning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate strategic goals from causal reasoning.
        
        Args:
            causal_reasoning: Output from causal GNN [batch_size, causal_dim]
            
        Returns:
            goals: Strategic goals [batch_size, goal_dim]
            priorities: Goal priorities [batch_size, 1]
        """
        goals = self.goal_generator(causal_reasoning)
        priorities = self.priority_network(causal_reasoning)
        
        return goals, priorities


class SlowStrategicReasoner(nn.Module):
    """
    Layer 3: Complete Slow Strategic Reasoner
    
    Performs deep causal analysis and strategic planning.
    Operates asynchronously (every N steps) for computational efficiency.
    """
    
    def __init__(self, state_dim: int = 128, causal_dim: int = 64, 
                 goal_dim: int = 32, max_nodes: int = 10, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.causal_dim = causal_dim
        self.goal_dim = goal_dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Causal graph builder
        self.graph_builder = CausalGraphBuilder(state_dim, max_nodes, device)
        
        # GNN reasoner
        self.gnn_reasoner = GraphNeuralReasoner(
            node_dim=state_dim // 4,  # From graph builder
            hidden_dim=causal_dim,
            output_dim=causal_dim,
            device=device
        )
        
        # Goal formation
        self.goal_former = GoalFormationModule(causal_dim, goal_dim, device)
        
        # Strategic prediction head
        self.strategic_head = nn.Sequential(
            nn.Linear(causal_dim + goal_dim, causal_dim, device=device),
            nn.ReLU(),
            nn.Linear(causal_dim, causal_dim, device=device),
            nn.LayerNorm(causal_dim, device=device)
        )
        
    def forward(self, state_buffer: torch.Tensor, 
               step_count: int, async_interval: int = 5) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Perform strategic causal reasoning.
        
        Args:
            state_buffer: Historical state sequence [batch_size, buffer_len, state_dim]
            step_count: Current step number
            async_interval: Process every N steps
            
        Returns:
            strategic_prediction: Strategic prediction [batch_size, causal_dim] (or None if not processing)
            goal_context: Strategic goals [batch_size, goal_dim] (or None)
            causal_info: Causal graph analysis information (or None)
        """
        # Only process every async_interval steps
        if step_count % async_interval != 0:
            return None, None, None
            
        batch_size = state_buffer.size(0)
        
        # Build causal graph
        node_features, adjacency_matrix, causal_strengths = self.graph_builder(state_buffer)
        
        # Convert to PyTorch Geometric format for GNN processing
        strategic_outputs = []
        causal_infos = []
        
        for batch_idx in range(batch_size):
            # Extract edges for this batch
            adj = adjacency_matrix[batch_idx]
            edge_threshold = 0.5  # Threshold for edge existence
            
            # Find edges above threshold
            edge_indices = torch.nonzero(adj > edge_threshold, as_tuple=False).t()
            
            if edge_indices.size(1) > 0:  # If edges exist
                # Create PyTorch Geometric data
                nodes = node_features[batch_idx]
                
                # Perform GNN reasoning
                strategic_reasoning = self.gnn_reasoner(nodes, edge_indices)
            else:
                # No edges found - use mean pooling
                nodes = node_features[batch_idx]
                strategic_reasoning = torch.mean(nodes, dim=0, keepdim=True)
                strategic_reasoning = self.gnn_reasoner.global_reasoner(strategic_reasoning)
                
            strategic_outputs.append(strategic_reasoning)
            
            # Collect causal information
            with torch.no_grad():
                num_edges = edge_indices.size(1) if edge_indices.size(1) > 0 else 0
                avg_causal_strength = causal_strengths[batch_idx].mean().item()
                graph_density = num_edges / (self.max_nodes * (self.max_nodes - 1))
                
                causal_infos.append({
                    'num_causal_edges': num_edges,
                    'avg_causal_strength': avg_causal_strength,
                    'graph_density': graph_density,
                    'reasoning_triggered': True
                })
                
        # Stack outputs
        strategic_prediction = torch.stack(strategic_outputs, dim=0)
        
        # Generate strategic goals
        goals, priorities = self.goal_former(strategic_prediction)
        
        # Combine reasoning and goals for final strategic prediction
        combined_input = torch.cat([strategic_prediction, goals], dim=-1)
        final_strategic_output = self.strategic_head(combined_input)
        
        # Aggregate causal information
        aggregated_causal_info = {
            'batch_causal_info': causal_infos,
            'avg_edges_per_graph': np.mean([info['num_causal_edges'] for info in causal_infos]),
            'avg_causal_strength': np.mean([info['avg_causal_strength'] for info in causal_infos]),
            'avg_graph_density': np.mean([info['graph_density'] for info in causal_infos]),
            'processing_step': step_count
        }
        
        return final_strategic_output, goals, aggregated_causal_info
    
    def analyze_causal_structure(self, state_buffer: torch.Tensor) -> Dict[str, Any]:
        """
        Detailed analysis of causal structure in state buffer.
        
        Args:
            state_buffer: State sequence [batch_size, seq_len, state_dim]
            
        Returns:
            Detailed causal analysis
        """
        with torch.no_grad():
            node_features, adjacency_matrix, causal_strengths = self.graph_builder(state_buffer)
            
            analysis = {
                'causal_graph_nodes': node_features.shape[1],
                'avg_node_activation': torch.mean(node_features).item(),
                'causal_edge_density': torch.mean(adjacency_matrix > 0.5).item(),
                'strongest_causal_connection': torch.max(causal_strengths).item(),
                'weakest_causal_connection': torch.min(causal_strengths).item(),
                'causal_network_complexity': torch.sum(adjacency_matrix > 0.5).item()
            }
            
        return analysis


class StrategicPlanningModule(nn.Module):
    """
    Long-term strategic planning based on causal understanding.
    """
    
    def __init__(self, causal_dim: int, goal_dim: int, plan_horizon: int = 10, device: str = 'cpu'):
        super().__init__()
        self.causal_dim = causal_dim
        self.goal_dim = goal_dim
        self.plan_horizon = plan_horizon
        self.device = device
        
        # Strategic planner network
        self.planner = nn.Sequential(
            nn.Linear(causal_dim + goal_dim, causal_dim * 2, device=device),
            nn.ReLU(),
            nn.Linear(causal_dim * 2, causal_dim, device=device),
            nn.ReLU(),
            nn.Linear(causal_dim, plan_horizon * goal_dim, device=device)  # Plans for multiple steps
        )
        
        # Plan evaluation network
        self.evaluator = nn.Sequential(
            nn.Linear(plan_horizon * goal_dim, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 1, device=device),
            nn.Sigmoid()  # Plan quality score
        )
        
    def forward(self, causal_reasoning: torch.Tensor, goals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate strategic plans.
        
        Args:
            causal_reasoning: Causal analysis [batch_size, causal_dim]
            goals: Strategic goals [batch_size, goal_dim]
            
        Returns:
            strategic_plan: Multi-step plan [batch_size, plan_horizon, goal_dim]
            plan_quality: Quality assessment [batch_size, 1]
        """
        # Combine causal reasoning and goals
        combined_input = torch.cat([causal_reasoning, goals], dim=-1)
        
        # Generate strategic plan
        plan_flat = self.planner(combined_input)
        strategic_plan = plan_flat.view(-1, self.plan_horizon, self.goal_dim)
        
        # Evaluate plan quality
        plan_quality = self.evaluator(plan_flat)
        
        return strategic_plan, plan_quality
