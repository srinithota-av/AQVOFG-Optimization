# Benchmark Datasets and Numerical Results

This directory contains the numerical benchmark data and raw optimization results for the **Adaptive Quantum Variable-Order Fractional Gradient (AQVOFG)** optimization algorithm.

---

## 📂 Directory Structure

```text
data/
├── README.md                  # Documentation
├── benchmarks/                # Test function configuration metadata
│   ├── sphere.json
│   ├── rosenbrock.json
│   ├── rastrigin.json
│   ├── ackley.json
│   └── griewank.json
└── results/                   # Raw output CSVs for reproducibility
    ├── convergence_curves.csv
    └── statistical_summary.csv
