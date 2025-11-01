# DHC-SSM Enhanced Architecture v2.1

## Deterministic Hierarchical Causal State Space Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![O(n) Complexity](https://img.shields.io/badge/Complexity-O(n)-brightgreen.svg)](#)
[![Deterministic](https://img.shields.io/badge/Learning-Deterministic-blue.svg)](#)

An AI architecture combining spatial processing, temporal modeling, and causal reasoning with O(n) complexity and deterministic learning.

---

## Performance Results (v2.1)

**Benchmark Results (2025-11-01)**
- Forward Pass Success Rate: 100.0% (20/20 trials)
- Learning Steps Success Rate: 100.0% (20/20 trials)
- Configurations tested: debug + cpu_optimized

**Improvements from v2.0**
- Forward Pass: +30.0% improvement (70% → 100%)
- Learning Steps: +20.0% improvement (80% → 100%)
- Average: +25.0% total enhancement

**Reproduce Results**
```bash
python tests/benchmark_comprehensive.py
# Results saved to: benchmark_results_v2_1_comprehensive.json
```

---

## Architecture Overview

**Four-Layer Design:**
- **Layer 1:** CNN (O(n)) - Spatial feature extraction
- **Layer 2:** State Space Model (O(n)) - Temporal processing
- **Layer 3:** Causal GNN - Strategic reasoning with fallbacks
- **Layer 4:** Deterministic Engine - Pareto optimization

All layers include error handling, dimension alignment, and device consistency.

---

## v2.1 Enhancements

**Updated components:**
- Shape Validator - Tensor validation
- Pareto Navigator - Stability + NaN protection
- Causal GNN - torch_geometric fallbacks + error handling
- Integration Layer - Fusion + dimension alignment
- Config System - CPU/CUDA optimized presets
- Error Handling - Graceful degradation
- Device Consistency - Automatic tensor alignment
- NaN Protection - Numerical stability

---

## Quick Start

```bash
# Install
pip install -e .

# Run demo
python examples/demo.py

# Test benchmark
python tests/benchmark_comprehensive.py

# Different configurations:
python -c "from dhc_ssm.utils.config import get_cpu_optimized_config; print('CPU optimized loaded')"
python -c "from dhc_ssm.utils.config import get_cuda_optimized_config; print('CUDA optimized loaded')"
```

**Configuration Presets:**
- `get_debug_config()` - Minimal for testing
- `get_cpu_optimized_config()` - CPU-only systems  
- `get_cuda_optimized_config()` - GPU acceleration
- `get_small_config()` - Fast experimentation
- `get_large_config()` - Maximum capacity

---

## Stability Testing

**Tested Configurations:**
- Debug Config: 100% success rate (small model)
- CPU Optimized: 100% success rate (medium model)
- Both pathways stable: Fast tactical + slow strategic
- Error scenarios handled: Fallback mechanisms included

**Current Status:**
- Handles edge cases gracefully
- Maintains O(n) efficiency
- Deterministic learning without probabilistic sampling
- Runtime crash elimination

---

## Technical Features

**Key Components:**
- **O(n) Complexity:** Linear time complexity
- **Deterministic Learning:** No probabilistic sampling uncertainty
- **Multi-pathway Processing:** Fast + slow reasoning with fusion
- **Causal Understanding:** Graph-based reasoning
- **Information-theoretic Motivation:** Intrinsic signals
- **Stability:** Error handling throughout

---

## Benchmarking

**Current Results (v2.1):**
- Forward Pass: 100.0%
- Learning Steps: 100.0%
- Status: Stable

**Benchmark Files:**
- `benchmark_results_v2_1_comprehensive.json` - Results
- `tests/benchmark_comprehensive.py` - Test runner

---

## Repository Structure

```
dhc_ssm/
├── spatial/          # Layer 1: CNN
├── tactical/         # Layer 2: SSM processor
├── strategic/        # Layer 3: Causal GNN
├── deterministic/    # Layer 4: Learning engine
├── integration/      # System integration
├── utils/           # Validation, config, device management
examples/            # Demos and usage examples
tests/               # Benchmarks
```

---

## Contributing

Enhancements welcome:
- Additional configuration presets
- Extended benchmarking scenarios
- Integration with other frameworks
- Performance optimizations

---

## License

MIT License

---

## Summary

**DHC-SSM v2.1 features:**
- 100% benchmark success rate
- +25% improvement from v2.0
- O(n) efficiency maintained
- Deterministic learning approach
- Error handling implemented
- Multiple optimized configurations