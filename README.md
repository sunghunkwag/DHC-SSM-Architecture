# 🧠 DHC-SSM Enhanced Architecture v2.0.1 (Honest Results)

## Deterministic Hierarchical Causal State Space Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![O(n) Complexity](https://img.shields.io/badge/Complexity-O(n)-brightgreen.svg)](#)
[![Deterministic](https://img.shields.io/badge/Learning-Deterministic-blue.svg)](#)
[![Development Status](https://img.shields.io/badge/Status-Beta-yellow.svg)](#)

**Research prototype. This README shows only measured facts.**

---

## 📊 Real Performance Results

### Measured on 2025-11-01 (v2.0 baseline)
- Forward Pass Success Rate: **70.0%** (7/10)
- Learning Steps Success Rate: **80.0%** (8/10)
- Status: **Not production-ready**

### Improvements Applied (v2.0.1)
- DimensionAligner: edge-case handling, device safety, NaN protection
- ParetoOptimizer: NaN protection, temperature scheduling, weight floor/renorm
- GradientComputer: None-grad fallback, gradient clipping
- DeviceManager: centralized device handling

### How to Reproduce Real Results
```bash
# Baseline (v2.0)
python tests/benchmark_real.py

# After fixes (v2.0.1)
python tests/benchmark_after_fixes.py
```

The JSON outputs (benchmark_results_real.json, benchmark_results_after_fixes.json) are the source of truth.

---

## 🏗️ Architecture Overview

- Layer 1: Enhanced CNN (O(n))
- Layer 2: State Space Model (O(n)) + Temporal Fusion (fixed)
- Layer 3: Causal GNN (async)
- Layer 4: Deterministic Engine (Intrinsic + Pareto)

All cross-layer interfaces now have validation and dimension alignment.

---

## ⚠️ Current Status (Facts Only)
- v2.0 baseline: 70%/80% (measured)
- v2.0.1 contains stability patches and new benchmark scripts
- Production readiness depends on your environment (run the scripts above)
- README will only reflect numbers from the JSON files you generate

---

## 🚀 Run
```bash
pip install -e .
python examples/demo.py
python tests/benchmark_real.py
python tests/benchmark_after_fixes.py
```

---

## 🛠️ Next Steps (Evidence-Driven)
- If your after-fixes JSON < 90%:
  - Forward: add enhanced tensor preprocessor for complex shapes
  - Learning: stabilize causal graph (edge thresholding, degree clipping)
  - Broaden validation and error recovery

---

## 📄 License
MIT
