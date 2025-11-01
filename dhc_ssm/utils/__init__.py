"""
DHC-SSM Utilities Package

Provides essential utilities for the DHC-SSM architecture:
- Configuration management
- Shape validation and dimension alignment
- Error handling utilities
"""

from .config import (
    DHCSSMConfig,
    SpatialConfig,
    TacticalConfig, 
    StrategicConfig,
    DeterministicConfig,
    SystemConfig,
    get_default_config,
    get_small_config,
    get_large_config
)

from .shape_validator import (
    ShapeValidator,
    DimensionAligner,
    FlexibleConcatenation,
    ShapeReporter,
    validate_dhc_ssm_flow
)

__all__ = [
    # Configuration
    "DHCSSMConfig",
    "SpatialConfig",
    "TacticalConfig",
    "StrategicConfig", 
    "DeterministicConfig",
    "SystemConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config",
    
    # Validation
    "ShapeValidator",
    "DimensionAligner",
    "FlexibleConcatenation",
    "ShapeReporter",
    "validate_dhc_ssm_flow"
]
