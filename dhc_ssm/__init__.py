"""
DHC-SSM: Deterministic Hierarchical Causal State Space Model

A revolutionary AI architecture that eliminates probabilistic sampling uncertainty
while achieving O(n) computational efficiency through:

1. Spatial Encoder Backbone (Enhanced CNN)
2. Fast Tactical Processor (SSM O(n))
3. Slow Strategic Reasoner (Causal GNN)
4. Deterministic Learning Engine (Information-Theoretic)

Author: Sung Hun Kwag
License: MIT
"""

from .integration.dhc_ssm_model import DHCSSMArchitecture
from .spatial.enhanced_cnn import SpatialEncoderBackbone
from .tactical.ssm_processor import FastTacticalProcessor
from .strategic.causal_gnn import SlowStrategicReasoner
from .deterministic.pareto_navigator import DeterministicLearningEngine

__version__ = "1.0.0"
__author__ = "Sung Hun Kwag"
__license__ = "MIT"

__all__ = [
    "DHCSSMArchitecture",
    "SpatialEncoderBackbone", 
    "FastTacticalProcessor",
    "SlowStrategicReasoner",
    "DeterministicLearningEngine",
]
