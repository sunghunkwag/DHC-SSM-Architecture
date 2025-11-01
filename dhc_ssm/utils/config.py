"""
Configuration Management

Provides flexible configuration system with validation and defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import yaml
from pathlib import Path

@dataclass
class SpatialConfig:
    """Configuration for spatial encoder."""
    input_channels: int = 3
    feature_dim: int = 256
    num_scales: int = 3
    use_attention: bool = True
    dropout: float = 0.1
    
@dataclass
class TacticalConfig:
    """Configuration for tactical processor."""
    state_dim: int = 128
    prediction_dim: int = 64
    num_layers: int = 4
    use_selective_scan: bool = True
    dropout: float = 0.1
    
@dataclass
class StrategicConfig:
    """Configuration for strategic reasoner."""
    causal_dim: int = 64
    num_gnn_layers: int = 3
    max_nodes: int = 20
    async_interval: int = 5
    use_temporal_edges: bool = True
    dropout: float = 0.1
    
@dataclass
class DeterministicConfig:
    """Configuration for deterministic learning engine."""
    action_dim: int = 32
    num_objectives: int = 4
    learning_rate: float = 0.001
    pareto_alpha: float = 0.5
    intrinsic_weight: float = 1.0
    
@dataclass
class SystemConfig:
    """Configuration for system-level parameters."""
    device: str = 'cpu'
    buffer_size: int = 50
    temporal_context_length: int = 10
    checkpoint_interval: int = 100
    validation_interval: int = 10
    
@dataclass
class DHCSSMConfig:
    """Complete DHC-SSM configuration."""
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
    strategic: StrategicConfig = field(default_factory=StrategicConfig)
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Model metadata
    version: str = "2.0.0"
    architecture_type: str = "DHC-SSM"
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = asdict(self)
        path_obj = Path(path)
        
        if path_obj.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path_obj.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    
    @classmethod
    def load(cls, path: str) -> 'DHCSSMConfig':
        """Load configuration from file."""
        path_obj = Path(path)
        
        if path_obj.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path_obj.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
            
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Validate dimensions are positive
        if self.spatial.feature_dim <= 0:
            errors.append("spatial.feature_dim must be positive")
        if self.tactical.state_dim <= 0:
            errors.append("tactical.state_dim must be positive")
        if self.strategic.causal_dim <= 0:
            errors.append("strategic.causal_dim must be positive")
        if self.deterministic.action_dim <= 0:
            errors.append("deterministic.action_dim must be positive")
            
        # Validate rates and probabilities
        if not 0 < self.deterministic.learning_rate < 1:
            errors.append("learning_rate must be between 0 and 1")
        if not 0 <= self.spatial.dropout <= 1:
            errors.append("dropout rates must be between 0 and 1")
            
        # Validate device
        valid_devices = ['cpu', 'cuda', 'mps']
        if self.system.device not in valid_devices and not self.system.device.startswith('cuda:'):
            errors.append(f"device must be one of {valid_devices} or 'cuda:N'")
            
        return errors


def get_default_config() -> DHCSSMConfig:
    """Get default configuration for DHC-SSM."""
    return DHCSSMConfig()

def get_small_config() -> DHCSSMConfig:
    """Get small configuration for testing."""
    return DHCSSMConfig(
        spatial=SpatialConfig(
            feature_dim=64,
            num_scales=2
        ),
        tactical=TacticalConfig(
            state_dim=32,
            prediction_dim=16
        ),
        strategic=StrategicConfig(
            causal_dim=16,
            max_nodes=8,
            num_gnn_layers=2
        ),
        deterministic=DeterministicConfig(
            action_dim=8
        ),
        system=SystemConfig(
            buffer_size=20,
            temporal_context_length=5
        )
    )

def get_large_config() -> DHCSSMConfig:
    """Get large configuration for production."""
    return DHCSSMConfig(
        spatial=SpatialConfig(
            feature_dim=512,
            num_scales=4
        ),
        tactical=TacticalConfig(
            state_dim=256,
            prediction_dim=128
        ),
        strategic=StrategicConfig(
            causal_dim=128,
            max_nodes=30,
            num_gnn_layers=4
        ),
        deterministic=DeterministicConfig(
            action_dim=64
        ),
        system=SystemConfig(
            buffer_size=100,
            temporal_context_length=20
        )
    )
