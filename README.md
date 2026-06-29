# DQN-RLMOEA for UAM Vertiport Location

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Q-Network integrated multi-objective evolutionary algorithm (**DQN-RLMOEA**) for Urban Air Mobility (UAM) vertiport maximal covering location, with a Shanghai empirical case study.

## Overview

This repository contains the implementation of **DQN-RLMOEA**, a reinforcement-learning-enhanced multi-objective evolutionary algorithm that solves the Multi-Objective Optimization Vertiport Location Problem (MOOVLP). The algorithm integrates:

- **Target Network** — stabilizes Q-learning by decoupling action selection from target computation
- **Experience Replay** — breaks temporal correlations via mini-batch sampling from a transition buffer
- **Hypervolume-based Reward** — replaces simple Pareto-size counting with the hypervolume indicator for principled multi-objective feedback
- **Adaptive Crossover Selection** — uses an ε-greedy DQN to select among single-point, two-point, and uniform crossover operators based on population state

## Repository Structure

```
├── RLMOEA2_shanghai_MOSMCLP.py       # Main algorithm + Shanghai case study
├── four_algorithms_com_zdt_theory.py  # ZDT benchmark comparison (NSGA-II, MOEA/D, SPEA3, RL-MOEA1)
├── sensitivity_analysis_RLMOEA2.py   # One-Factor-at-a-Time sensitivity analysis (90 experiments)
├── spatial_strategy_analysis.py      # Four-experiment spatial differentiation framework
├── ZDT前沿数据/                      # True Pareto front data for ZDT1–ZDT4, ZDT6
├── shanghai_problem.npz              # Pre-computed coverage matrix (200 × 3,902)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ZDT benchmark comparison

```bash
python four_algorithms_com_zdt_theory.py
```

Compares DQN-RLMOEA against NSGA-II, MOEA/D, SPEA3, and RL-MOEA1 on the five-problem ZDT benchmark suite (ZDT1–ZDT4, ZDT6) using IGD and HV metrics. True Pareto front data is loaded from `ZDT前沿数据/前沿数据/`.

### 3. Run the main Shanghai experiment

```bash
python RLMOEA2_shanghai_MOSMCLP.py
```

This loads the pre-computed coverage matrix from `shanghai_problem.npz`, runs DQN-RLMOEA for 300 generations (population size 80), extracts the Pareto front, and outputs four representative schemes with spatial distribution analysis.

### 4. Run the spatial strategy analysis

```bash
python spatial_strategy_analysis.py
```

Produces the four-experiment spatial differentiation report and figures (urban score composite panel, Moran scatter plots, CDF curves, permutation test distributions).

### 5. Run the sensitivity analysis

```bash
python sensitivity_analysis_RLMOEA2.py
```

Executes 90 independent experiments (6 parameters × 5 levels × 3 repeats) using the OFAT method. Results are cached to `sensitivity_results.json`. The tuned configuration is **P = 80, G = 300, p_m = 0.05, ε₀ = 0.85, η = 0.0005, γ = 0.80**.

## Data

The `shanghai_problem.npz` file contains the pre-computed coverage matrix (Haversine distance with a 5 km binary threshold) for 200 candidate vertiport sites and 3,902 community demand points in Shanghai. Community population weights are derived from 24-hour average mobile phone signaling data.

The original raw data files (building footprints, POI, demand Shapefile, administrative boundaries) are excluded from this repository. To regenerate `shanghai_problem.npz` from raw data, place the source files under `shanghai_data_processer/` and run the experiment script.

## Model

**MOOVLP** (Multi-Objective Optimization Vertiport Location Problem):

$$\min \; \sum_{j \in J} w_j (1 - y_j), \quad \min \; \sum_{i \in I} c_i x_i$$

Subject to coverage, busyness-reliability, and facility quantity constraints. Decision variables: $x_i \in \{0,1\}$ (facility selection), $y_j \in \{0,1\}$ (demand coverage).

## Key Results

| Metric | Value |
|--------|-------|
| Pareto-optimal solutions | 4 representative schemes (S1–S4) |
| Strategy spectrum | Systematic peripheral infill (S1, p < 0.0001, Cohen's d = −5.95) → proportional representation (S3/S4, p > 0.7) |
| Dominant hyperparameter | Mutation probability (HV range 9.83 × 10⁹, 5–18× others) |
| Tuned configuration | HV improved by 7.4% over baseline |

## Citation

If you use this code, please cite:

```
Zhou, X., Zhang, C., Wang, H., Zhao, D., & Wu, C. (2025).
A Deep Q-Network Integrated Multi-Objective Covering Model for
Urban Air Mobility Vertiport Location: A Case Study of Shanghai.
```

## License

No License
