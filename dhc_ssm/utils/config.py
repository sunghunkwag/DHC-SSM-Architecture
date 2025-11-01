"""
Configuration Management v2.1

Provides flexible configuration system with comprehensive validation and defaults.
Enhanced with CPU/CUDA presets and stability features.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

# Optional YAML support with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


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
    pareto_epsilon: float = 0.01
    learning_rate: float = 0.001
    use_adaptive_weights: bool = True
    gradient_clip: float = 1.0
    temperature_init: float = 2.0


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    checkpoint_interval: int = 10
    validation_interval: int = 5
    early_stopping_patience: int = 20


@dataclass
class SystemConfig:
    """Configuration for system-wide settings."""
    device: str = 'cpu'
    buffer_size: int = 50
    seed: int = 42
    num_workers: int = 4
    log_level: str = 'INFO'
    experiment_name: str = 'dhc_ssm_experiment'
    output_dir: str = './outputs'
    temporal_context_length: int = 10


@dataclass
class DHCSSMConfig:
    """Complete DHC-SSM configuration with comprehensive validation."""
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
    strategic: StrategicConfig = field(default_factory=StrategicConfig)
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Model metadata
    version: str = "2.1.0"
    architecture_type: str = "DHC-SSM"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Optional[str] = None) -> str:
        """
        Convert configuration to JSON with error handling.
        
        Args:
            path: Optional path to save JSON file
            
        Returns:
            JSON string
        """
        try:
            json_str = json.dumps(self.to_dict(), indent=2)
            
            if path:
                Path(path).write_text(json_str)
                
            return json_str
        except Exception as e:
            raise ValueError(f"Failed to convert to JSON: {e}")
    
    def to_yaml(self, path: Optional[str] = None) -> str:
        """
        Convert configuration to YAML (if available).
        
        Args:
            path: Optional path to save YAML file
            
        Returns:
            YAML string
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Use pip install PyYAML")
            
        try:
            yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
            
            if path:
                Path(path).write_text(yaml_str)
                
            return yaml_str
        except Exception as e:
            raise ValueError(f"Failed to convert to YAML: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DHCSSMConfig':
        """
        Create configuration from dictionary with comprehensive error handling.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            DHCSSMConfig instance
        """
        try:
            spatial = SpatialConfig(**config_dict.get('spatial', {}))
            tactical = TacticalConfig(**config_dict.get('tactical', {}))
            strategic = StrategicConfig(**config_dict.get('strategic', {}))
            deterministic = DeterministicConfig(**config_dict.get('deterministic', {}))
            training = TrainingConfig(**config_dict.get('training', {}))
            system = SystemConfig(**config_dict.get('system', {}))
            
            return cls(
                spatial=spatial,
                tactical=tactical,
                strategic=strategic,
                deterministic=deterministic,
                training=training,
                system=system,
                version=config_dict.get('version', '2.1.0'),
                architecture_type=config_dict.get('architecture_type', 'DHC-SSM')
            )
        except Exception as e:
            raise ValueError(f"Failed to create config from dict: {e}")
    
    @classmethod
    def from_json(cls, path: str) -> 'DHCSSMConfig':
        """
        Load configuration from JSON file with error handling.
        
        Args:
            path: Path to JSON file
            
        Returns:
            DHCSSMConfig instance
        """
        try:
            config_dict = json.loads(Path(path).read_text())
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")
    
    @classmethod
    def from_yaml(cls, path: str) -> 'DHCSSMConfig':
        """
        Load configuration from YAML file with error handling.
        
        Args:
            path: Path to YAML file
            
        Returns:
            DHCSSMConfig instance
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Use: pip install PyYAML")
            
        try:
            config_dict = yaml.safe_load(Path(path).read_text())
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")
    
    def validate(self) -> List[str]:
        """
        Comprehensive configuration validation.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate spatial config
        if self.spatial.input_channels < 1:
            errors.append("spatial.input_channels must be positive")
        if self.spatial.feature_dim < 1:
            errors.append("spatial.feature_dim must be positive")
        if not 0 <= self.spatial.dropout <= 1:
            errors.append("spatial.dropout must be between 0 and 1")
        if self.spatial.num_scales < 1:
            errors.append("spatial.num_scales must be positive")
            
        # Validate tactical config
        if self.tactical.state_dim < 1:
            errors.append("tactical.state_dim must be positive")
        if self.tactical.prediction_dim < 1:
            errors.append("tactical.prediction_dim must be positive")
        if self.tactical.num_layers < 1:
            errors.append("tactical.num_layers must be positive")
        if not 0 <= self.tactical.dropout <= 1:
            errors.append("tactical.dropout must be between 0 and 1")
            
        # Validate strategic config
        if self.strategic.causal_dim < 1:
            errors.append("strategic.causal_dim must be positive")
        if self.strategic.async_interval < 1:
            errors.append("strategic.async_interval must be positive")
        if self.strategic.max_nodes < 2:
            errors.append("strategic.max_nodes must be >= 2")
        if self.strategic.num_gnn_layers < 1:
            errors.append("strategic.num_gnn_layers must be positive")
        if not 0 <= self.strategic.dropout <= 1:
            errors.append("strategic.dropout must be between 0 and 1")
            
        # Validate deterministic config
        if self.deterministic.action_dim < 1:
            errors.append("deterministic.action_dim must be positive")
        if self.deterministic.num_objectives < 1:
            errors.append("deterministic.num_objectives must be positive")
        if self.deterministic.learning_rate <= 0:
            errors.append("deterministic.learning_rate must be positive")
        if not 0 < self.deterministic.pareto_epsilon < 1:
            errors.append("deterministic.pareto_epsilon must be between 0 and 1")
        if self.deterministic.gradient_clip <= 0:
            errors.append("deterministic.gradient_clip must be positive")
        if self.deterministic.temperature_init <= 0:
            errors.append("deterministic.temperature_init must be positive")
            
        # Validate system config
        if self.system.device not in ['cpu', 'cuda', 'mps'] and not self.system.device.startswith('cuda:'):
            errors.append("system.device must be 'cpu', 'cuda', 'mps', or 'cuda:N'")
        if self.system.buffer_size < 1:
            errors.append("system.buffer_size must be positive")
        if self.system.temporal_context_length < 1:
            errors.append("system.temporal_context_length must be positive")
            
        return errors
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))


def get_default_config() -> DHCSSMConfig:
    """
    Get default configuration optimized for general use.
    
    Returns:
        Default DHCSSMConfig instance
    """
    return DHCSSMConfig()


def get_small_config() -> DHCSSMConfig:
    """
    Get small model configuration for fast experimentation and testing.
    
    Returns:
        Small DHCSSMConfig instance
    """
    return DHCSSMConfig(
        spatial=SpatialConfig(
            feature_dim=128,
            num_scales=2
        ),
        tactical=TacticalConfig(
            state_dim=64,
            prediction_dim=32,
            num_layers=3
        ),
        strategic=StrategicConfig(
            causal_dim=32,
            max_nodes=8,
            num_gnn_layers=2,
            async_interval=8
        ),
        deterministic=DeterministicConfig(
            action_dim=16,
            learning_rate=0.01
        ),
        system=SystemConfig(
            buffer_size=30,
            temporal_context_length=8
        )
    )


def get_large_config() -> DHCSSMConfig:
    """
    Get large model configuration for maximum capacity and performance.
    
    Returns:
        Large DHCSSMConfig instance
    """
    return DHCSSMConfig(
        spatial=SpatialConfig(
            feature_dim=512,
            num_scales=4
        ),
        tactical=TacticalConfig(
            state_dim=256,
            prediction_dim=128,
            num_layers=6
        ),
        strategic=StrategicConfig(
            causal_dim=128,
            max_nodes=25,
            num_gnn_layers=4,
            async_interval=3
        ),
        deterministic=DeterministicConfig(
            action_dim=64,
            learning_rate=0.0005
        ),
        system=SystemConfig(
            buffer_size=100,
            temporal_context_length=20
        )
    )


def get_cpu_optimized_config() -> DHCSSMConfig:
    """
    Get CPU-optimized configuration for better stability on CPU-only systems.
    
    Returns:
        CPU-optimized DHCSSMConfig instance
    """
    config = get_small_config()
    
    # CPU-specific optimizations
    config.system.device = 'cpu'
    config.strategic.max_nodes = 6  # Reduce for CPU efficiency
    config.strategic.async_interval = 10  # Less frequent strategic processing
    config.strategic.num_gnn_layers = 2  # Lighter GNN
    config.training.batch_size = 16  # Smaller batch for CPU
    config.deterministic.gradient_clip = 0.5  # More conservative
    
    return config


def get_cuda_optimized_config() -> DHCSSMConfig:
    """
    Get CUDA-optimized configuration for GPU acceleration and performance.
    
    Returns:
        CUDA-optimized DHCSSMConfig instance
    """
    config = get_large_config()
    
    # CUDA-specific optimizations
    config.system.device = 'cuda'
    config.strategic.max_nodes = 25  # More nodes for GPU
    config.strategic.async_interval = 3  # More frequent processing
    config.strategic.num_gnn_layers = 4  # Deeper GNN
    config.training.batch_size = 64  # Larger batch for GPU
    config.deterministic.temperature_init = 1.5  # Adjusted for GPU
    
    return config


def get_debug_config() -> DHCSSMConfig:
    """
    Get debug configuration with minimal parameters for quick testing.
    
    Returns:
        Debug DHCSSMConfig instance
    """
    return DHCSSMConfig(
        spatial=SpatialConfig(
            feature_dim=32,
            num_scales=1
        ),
        tactical=TacticalConfig(
            state_dim=16,
            prediction_dim=8,
            num_layers=2
        ),
        strategic=StrategicConfig(
            causal_dim=8,
            max_nodes=4,
            num_gnn_layers=1,
            async_interval=20
        ),
        deterministic=DeterministicConfig(
            action_dim=4,
            num_objectives=2,
            learning_rate=0.1
        ),
        system=SystemConfig(
            buffer_size=10,
            temporal_context_length=3
        )
    )


def create_config_from_template(template: str = 'default', **overrides) -> DHCSSMConfig:
    """
    Create configuration from template with custom overrides.
    
    Args:
        template: Template name ('default', 'small', 'large', 'cpu', 'cuda', 'debug')
        **overrides: Configuration overrides
        
    Returns:
        Configured DHCSSMConfig instance
    """
    # Get base config
    if template == 'small':
        config = get_small_config()
    elif template == 'large':
        config = get_large_config()
    elif template == 'cpu':
        config = get_cpu_optimized_config()
    elif template == 'cuda':
        config = get_cuda_optimized_config()
    elif template == 'debug':
        config = get_debug_config()
    else:
        config = get_default_config()
    
    # Apply overrides
    for key, value in overrides.items():
        if '.' in key:
            # Nested attribute (e.g., 'spatial.feature_dim')
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Top-level attribute
            setattr(config, key, value)
    
    # Validate after overrides
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed after overrides:\n" + "\n".join(errors))
    
    return config