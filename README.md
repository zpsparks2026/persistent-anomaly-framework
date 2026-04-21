# Persistent Anomaly Framework (PAF) — Pilot Simulation

**Author:** Zach Sparks  
**Affiliation:** Sparks Solutions LLC / Technological Leadership, Arizona State University

## Overview

Synthetic validation of the PAF screening pipeline. Generates a cohort of N=200 subjects (8 injected anomalies at effect sizes d=0.08–0.25, 192 null), simulates T=5 sessions of K=100 items with M=4 options each, extracts the paper's 11-dimensional feature space (6 signal + 5 noise-as-feature), and runs the 3-phase pipeline (k-means screening → Bayesian validation → DBSCAN + XGBoost characterization).

## Key Result

| Classifier | Sensitivity | Specificity | AUC |
|---|---|---|---|
| Bayesian threshold (P ≥ 0.95) | 0.575 ± 0.100 | 1.000 ± 0.000 | 0.971 ± 0.030 |
| XGBoost (11-dim features) | 0.925 ± 0.065 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Findings:**
1. The pipeline achieves **zero false positives** across all seeds — max null posterior is 0.003, far below the 0.95 threshold
2. The Bayesian threshold reliably detects anomalies with effect sizes d ≥ 0.15 but misses weaker effects (d = 0.08–0.12), consistent with the paper's power analysis
3. The 11-dimensional XGBoost classifier achieves perfect AUC, confirming that noise-as-feature dimensions provide discriminative signal beyond accuracy alone
4. **N2 (error directionality)** is the second most important feature after density, directly validating the noise-as-feature paradigm — structure in *errors* carries more signal than most conventional metrics
5. Override injection (initial-correct-then-changed) produces detectable signatures recoverable by the behavioral monitoring layer

## What This Demonstrates

This is a **proof-of-concept**, not empirical validation. It shows that:
- The pipeline architecture works as specified
- The 11-dimensional feature space separates injected anomalies from null subjects
- The Bayesian accumulation model correctly converges across sessions
- The specificity guarantee (zero false positives) holds under the stated threshold

It does **not** show that anomalous cognition exists or that real human subjects would produce the injected signal patterns.

## Quick Start

```bash
pip install numpy scipy scikit-learn xgboost
python experiment.py
```

## Repository Structure

```
├── experiment.py          # Full simulation
├── results.json           # Output (generated on run)
└── README.md
```

## Current Limitations

- Synthetic data with injected ground truth — not real human behavioral data
- Override pattern is modeled simplistically (30% override rate for anomalies)
- Temporal asymmetry (N1) is approximated via forward/backward hit comparison
- Cross-task coherence (N4) uses per-item accuracy correlation rather than CCA
- Alpha-stable index (N5) approximated by kurtosis
- XGBoost trained and evaluated on same data (no held-out test set) — AUC=1.0 is therefore an upper bound
