"""
DHC-SSM Enhanced: Deterministic Hierarchical Causal State Space Model

A production-ready implementation of the DHC-SSM architecture combining:
- Enhanced CNN for spatial processing (O(n))
- Selective State Space Models for temporal processing (O(n))
- Graph Neural Networks for causal reasoning
- Pareto optimization for deterministic multi-objective learning

This enhanced version includes:
- Robust error handling and validation
- Comprehensive testing infrastructure
- Production-ready features
- Extensive documentation
- Modular and extensible design

Author: Sung Hun Kwag (Enhanced v2.0)
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Sung Hun Kwag"
__license__ = "MIT"

# Import main components for easy access
from .integration.dhc_ssm_model import DHCSSMArchitecture
from .spatial.enhanced_cnn import SpatialEncoderBackbone
from .tactical.ssm_processor import FastTacticalProcessor
from .strategic.causal_gnn import SlowStrategicReasoner
from .deterministic.pareto_navigator import DeterministicLearningEngine

# Import utilities
from .utils.config import DHCSSMConfig, get_default_config, get_small_config, get_large_config
from .utils.shape_validator import ShapeValidator, DimensionAligner, FlexibleConcatenation

__all__ = [
    "DHCSSMArchitecture",
    "SpatialEncoderBackbone",
    "FastTacticalProcessor",
    "SlowStrategicReasoner", 
    "DeterministicLearningEngine",
    "DHCSSMConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config",
    "ShapeValidator",
    "DimensionAligner",
    "FlexibleConcatenation",
]

# Version info
version_info = {
    'version': __version__,
    'architecture': 'DHC-SSM',
    'complexity': 'O(n)',
    'learning_type': 'deterministic',
    'probabilistic_sampling': False,
    'enhancements': {
        'shape_validation': True,
        'dimension_alignment': True,
        'config_management': True,
        'error_handling': True,
        'device_consistency': True
    }
}
