"""
Layer 1: Spatial Encoder Backbone

Based on HierarchicalCNN-ReasoningFramework's Enhanced CNN components.
Provides O(n) complexity spatial feature extraction with multi-scale processing.

Key Features:
- Multi-scale feature extraction
- Dynamic convolution adaptation
- Spatial attention mechanisms
- Input-adaptive processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class DynamicConv2D(nn.Module):
    """
    Input-adaptive convolution layer that adjusts kernel weights based on input characteristics.
    Enables the model to adapt convolution operations to different input patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 adaptation_dim: int = 64, device: str = 'cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
        
        # Base convolution weights
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                 padding=kernel_size//2, device=device)
        
        # Adaptation network that generates weight modifications
        self.adaptation_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_channels, adaptation_dim, device=device),
            nn.ReLU(),
            nn.Linear(adaptation_dim, out_channels * in_channels * kernel_size * kernel_size, device=device),
            nn.Tanh()  # Keep adaptation weights bounded
        )
        
        # Scaling factor for adaptation strength
        self.adaptation_scale = nn.Parameter(torch.tensor(0.1, device=device))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic weight adaptation.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, out_channels, height, width]
        """
        batch_size = x.size(0)
        
        # Generate adaptation weights based on input
        adaptation_weights = self.adaptation_net(x)  # [batch_size, weight_dim]
        adaptation_weights = adaptation_weights.view(
            batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        
        # Get base weights and adapt them
        base_weights = self.base_conv.weight.unsqueeze(0)  # [1, out_ch, in_ch, k, k]
        adapted_weights = base_weights + self.adaptation_scale * adaptation_weights
        
        # Apply adapted convolution for each sample in batch
        outputs = []
        for i in range(batch_size):
            # Perform convolution with adapted weights for sample i
            sample_output = F.conv2d(
                x[i:i+1], adapted_weights[i], bias=self.base_conv.bias,
                padding=self.kernel_size//2
            )
            outputs.append(sample_output)
            
        return torch.cat(outputs, dim=0)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for selective feature processing.
    Helps the model focus on relevant spatial locations.
    """
    
    def __init__(self, channels: int, reduction: int = 16, device: str = 'cpu'):
        super().__init__()
        self.channels = channels
        self.device = device
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, device=device),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, device=device),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, device=device),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial and channel attention.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Attention-weighted tensor [batch_size, channels, height, width]
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_features)
        x = x * spatial_att
        
        return x


class MultiScaleFeatures(nn.Module):
    """
    Multi-scale feature extraction using different receptive fields.
    Captures both fine-grained and coarse-grained spatial patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        branch_channels = out_channels // 4
        
        # Different scale branches
        self.branch_1x1 = nn.Conv2d(in_channels, branch_channels, 1, device=device)
        self.branch_3x3 = nn.Conv2d(in_channels, branch_channels, 3, padding=1, device=device)
        self.branch_5x5 = nn.Conv2d(in_channels, branch_channels, 5, padding=2, device=device)
        self.branch_7x7 = nn.Conv2d(in_channels, branch_channels, 7, padding=3, device=device)
        
        # Batch normalization for each branch
        self.bn_1x1 = nn.BatchNorm2d(branch_channels, device=device)
        self.bn_3x3 = nn.BatchNorm2d(branch_channels, device=device)
        self.bn_5x5 = nn.BatchNorm2d(branch_channels, device=device)
        self.bn_7x7 = nn.BatchNorm2d(branch_channels, device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Multi-scale features [batch_size, out_channels, height, width]
        """
        # Extract features at different scales
        feat_1x1 = F.relu(self.bn_1x1(self.branch_1x1(x)))
        feat_3x3 = F.relu(self.bn_3x3(self.branch_3x3(x)))
        feat_5x5 = F.relu(self.bn_5x5(self.branch_5x5(x)))
        feat_7x7 = F.relu(self.bn_7x7(self.branch_7x7(x)))
        
        # Concatenate all scale features
        multi_scale_features = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_7x7], dim=1)
        
        return multi_scale_features


class SpatialEncoderBackbone(nn.Module):
    """
    Layer 1: Complete Spatial Encoder Backbone
    
    Integrates all spatial processing components:
    - Multi-scale feature extraction
    - Dynamic convolution adaptation  
    - Spatial attention mechanisms
    
    Complexity: O(n) where n is input size
    """
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 256, 
                 hidden_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, hidden_dim, 7, padding=3, device=device)
        self.initial_bn = nn.BatchNorm2d(hidden_dim, device=device)
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatures(hidden_dim, hidden_dim, device)
        
        # Dynamic convolution layers
        self.dynamic_conv1 = DynamicConv2D(hidden_dim, hidden_dim, 3, device=device)
        self.dynamic_conv2 = DynamicConv2D(hidden_dim, hidden_dim, 3, device=device)
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(hidden_dim, device=device)
        
        # Feature projection to output dimension
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(hidden_dim * 64, feature_dim, device=device),  # 8*8 = 64
            nn.ReLU(),
            nn.LayerNorm(feature_dim, device=device)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from input.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature vector f_t [batch_size, feature_dim]
        """
        # Initial convolution
        features = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Multi-scale processing
        features = self.multi_scale(features)
        
        # Dynamic convolution layers
        features = F.relu(self.dynamic_conv1(features))
        features = F.relu(self.dynamic_conv2(features))
        
        # Spatial attention
        features = self.spatial_attention(features)
        
        # Project to output feature dimension
        feature_vector = self.feature_projection(features)
        
        return feature_vector
    
    def get_spatial_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate spatial feature maps for visualization.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Dictionary of intermediate feature maps
        """
        maps = {}
        
        # Initial features
        features = F.relu(self.initial_bn(self.initial_conv(x)))
        maps['initial'] = features
        
        # Multi-scale features
        features = self.multi_scale(features)
        maps['multi_scale'] = features
        
        # After dynamic convolutions
        features = F.relu(self.dynamic_conv1(features))
        maps['dynamic_conv1'] = features
        
        features = F.relu(self.dynamic_conv2(features))
        maps['dynamic_conv2'] = features
        
        # After attention
        features = self.spatial_attention(features)
        maps['attention'] = features
        
        return maps
