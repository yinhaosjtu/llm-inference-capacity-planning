# Strategic Capacity Planning for Large Language Model Inference Services under Demand Uncertainty

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official code and benchmark instances for the paper:

> **Strategic Capacity Planning for Large Language Model Inference Services under Demand Uncertainty**  
> *International Journal of Production Economics* (Accepted / In Press)

---

## Overview

This repository provides the exact MILP formulations, solvers, benchmark instances, experiment scripts, and plotting pipeline accompanying the paper:

- **Model**: A risk-averse two-stage stochastic program for GPU cluster procurement and KV-cache memory reservation under joint demand, input-length, and output-length uncertainty, linearized via exact McCormick envelopes and incorporating Conditional Value-at-Risk ($\text{CVaR}_{0.95}$), power budgets, and carbon budgets.
- **Exact Solvers**:
  - **Branch-and-Cut (`solvers/gurobi_solver.py`)**: Monolithic extensive-form MILP strengthened *a priori* with four families of problem-specific aggregate valid inequalities and a mean-value warm start.
  - **Callback-Based Multi-Cut Benders Decomposition (`solvers/benders_solver.py`)**: Single-tree lazy-constraint Benders decomposition with energy-gradient-augmented optimality cuts, multithreaded LP subproblem solves, and an outer convergence-certification loop.
- **Benchmark Instances (`instances/`)**: 25 pre-generated, gzip-compressed benchmark instances across Small (S), Medium (M), and Large (L) scales (see [`instances/README.md`](instances/README.md) for the complete data dictionary).

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── RunAllExperiments.py                  # Master runner executing all experiments and figure generation
├── Plot.py                               # Generates all 9 PDF result figures from plot_data.json
├── solvers/
│   ├── __init__.py
│   ├── gurobi_solver.py                  # Extensive-form Branch-and-Cut solver (with valid inequalities)
│   └── benders_solver.py                 # Callback-based multi-cut Benders decomposition solver
├── experiments/
│   ├── comparative_experiment.py         # Algorithmic comparison: B&C vs. Benders (Tables 3 & 4)
│   ├── vss_and_evpi.py                   # Risk-adjusted VSS & EVPI evaluation (Tables 5 & 6)
│   └── sensitivity_analysis.py           # Sensitivity, risk aversion, and robustness analyses (Figs. 3–9)
└── instances/
    ├── README.md                         # Complete schema and parameter mapping to Table 1
    ├── S_01.json.gz ... S_10.json.gz     # 10 small instances  (|K|=10, |J|=10, |S|=1,000)
    ├── M_01.json.gz ... M_10.json.gz     # 10 medium instances (|K|=15, |J|=15, |S|=2,000)
    └── L_01.json.gz ... L_05.json.gz     # 5 large instances   (|K|=30, |J|=30, |S|=3,000)
```

---

## Requirements and Installation

### Prerequisites

- **Python**: 3.10 or higher (tested with Python 3.12).
- **Gurobi Optimizer**: A valid academic or commercial Gurobi license is required to run `gurobipy` on the benchmark instances (tested with Gurobi 13.0). Free academic licenses are available from [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/).
- **Hardware**: The paper benchmarks were run on an Intel Xeon machine with 64 GB RAM. Running the L-class extensive-form MILP (`|K|=30, |J|=30, |S|=3,000`) requires substantial RAM (recommended $\ge$ 32–64 GB).

### Setup

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Verify that Python and your Gurobi license are configured properly:

```bash
python -c "import gurobipy as gp; print('Gurobi version:', gp.gurobi.version())"
```

---

## Reproducing the Paper Results

### 1. Run the Full Pipeline

To execute all experiments and regenerate all tables and figures sequentially:

```bash
python RunAllExperiments.py
```

> **Runtime Note**: Running all experiments from scratch (without pre-populated caches) takes roughly **14–15 hours** on a multi-core workstation (approximately 10–20 minutes for VSS/EVPI, ~8 hours for the full 10-seed sensitivity and robustness suites, and ~6 hours for the 25-instance B&C vs. Benders benchmark). Intermediate results are cached automatically under `cache/` so interrupted runs can resume cleanly.

### 2. Run Individual Experiment Modules

Each experiment script can be executed independently from the repository root:

#### (a) Algorithmic Performance Comparison (Tables 3 & 4)

```bash
python experiments/comparative_experiment.py
```

- Solves all 25 instances (`S_01`–`S_10`, `M_01`–`M_10`, `L_01`–`L_05`) using both Branch-and-Cut (`EF` with valid inequalities) and Benders Decomposition (`BD`) with `TIME_LIMIT = 3600` s and `GAP_TOL = 0`.
- Writes sheets `Instance_Configs` and `Algorithm_Comparison` into `results.xlsx`.
- Caches per-instance solve results under `cache/comparative/`.

#### (b) Risk-Adjusted VSS and EVPI (Tables 5 & 6)

```bash
python experiments/vss_and_evpi.py --xlsx results.xlsx
```

- Evaluates `RP`, `EV`, `EEV`, and `WS` under the unified mean–$\text{CVaR}_{0.95}$ risk criterion on `S_01`–`S_05` and `M_01`–`M_05`.
- Optional CLI flags:
  - `--scales S M` (subset of `{S, M, L}`)
  - `--instances 5` (number of instances per scale)
  - `--ws-workers 4` (worker processes for parallel wait-and-see single-scenario solves; reduce to `1` if your Gurobi license restricts concurrent environments)
  - `--output results_vss_evpi_risk_consistent.json` (full-precision JSON output)
  - `--xlsx results.xlsx` (upserts `VSS` and `EVPI` sheets into `results.xlsx`)
- **Quick smoke test** (1 small instance, ~20–40 seconds):
  ```bash
  python experiments/vss_and_evpi.py --scales S --instances 1 --ws-workers 2
  ```

#### (c) Sensitivity, Risk Aversion, and Robustness Analyses (Figs. 3–9)

```bash
python experiments/sensitivity_analysis.py
python Plot.py
```

- `experiments/sensitivity_analysis.py` loads `instances/M_01.json.gz` as the base structure and executes:
  1. One-at-a-time sensitivity sweeps over $\gamma^{\text{cloud}}$, $\gamma^{\text{mem}}$ ($\mu^{\text{kv}}$), and $\gamma^{\text{cv}}$ across 10 seeds (`lambda = 0.3`).
  2. Risk-aversion sweep over $\lambda \in \{0.0, 0.1, 0.25, 0.5, 0.75, 1.0\}$ across 10 seeds.
  3. Correlation sweep over $\rho_j \in \{0.0, 0.3, 0.6, 0.9\}$ across 10 seeds, $\rho_j \times \lambda$ risk frontier across 3 seeds, and out-of-sample cross-family transfer between log-normal and moment-matched Gamma scenarios (3,000 evaluation scenarios per seed across 10 seeds).
- Writes aggregated plotting statistics to `plot_data.json` and caches individual solves under `cache/sensitivity/`.
- `Plot.py` reads `plot_data.json` and generates all 9 publication-ready PDF figures in the repository root.

---

## Mapping Between Outputs and Paper Tables / Figures

| Paper Reference | Description | Script | Output File / Sheet |
| :--- | :--- | :--- | :--- |
| **Table 3** | Test instance configurations | `experiments/comparative_experiment.py` | `results.xlsx` (`Instance_Configs`) |
| **Table 4** | Branch-and-Cut vs. Benders comparison | `experiments/comparative_experiment.py` | `results.xlsx` (`Algorithm_Comparison`) |
| **Table 5** | Risk-adjusted VSS | `experiments/vss_and_evpi.py` | `results.xlsx` (`VSS`), `results_vss_evpi_risk_consistent.json` |
| **Table 6** | Risk-adjusted EVPI | `experiments/vss_and_evpi.py` | `results.xlsx` (`EVPI`), `results_vss_evpi_risk_consistent.json` |
| **Fig. 3(a)** | Sensitivity to elastic cloud cost $\gamma^{\text{cloud}}$ | `sensitivity_analysis.py` + `Plot.py` | `fig_sens_cloud_cost.pdf` |
| **Fig. 3(b)** | Sensitivity to workload variability $\gamma^{\text{cv}}$ | `sensitivity_analysis.py` + `Plot.py` | `fig_sens_demand_cv.pdf` |
| **Fig. 4** | Sensitivity to KV-cache memory coefficient $\gamma^{\text{mem}}$ | `sensitivity_analysis.py` + `Plot.py` | `fig_sens_mukv.pdf` |
| **Fig. 5(a)** | Mean–$\text{CVaR}_{0.95}$ Pareto profile | `sensitivity_analysis.py` + `Plot.py` | `fig_risk_pareto.pdf` |
| **Fig. 5(b)** | Upper-tail exceedance probability (ECDF) | `sensitivity_analysis.py` + `Plot.py` | `fig_risk_tail_ecdf.pdf` |
| **Fig. 6** | Risk-aversion impact on costs & GPU tiers | `sensitivity_analysis.py` + `Plot.py` | `fig_risk_decisions.pdf` |
| **Fig. 7** | Robustness to demand–output correlation $\rho_j$ | `sensitivity_analysis.py` + `Plot.py` | `fig_robust_rho.pdf` |
| **Fig. 8** | Mean–risk frontier across correlation levels | `sensitivity_analysis.py` + `Plot.py` | `fig_robust_rho_frontier.pdf` |
| **Fig. 9** | Out-of-sample transfer (Log-normal vs. Gamma) | `sensitivity_analysis.py` + `Plot.py` | `fig_robust_dist_oos.pdf` |

*(Figures 1 and 2 in the paper are schematic diagrams rather than data plots.)*

---

## Using the Solvers Programmatically

Both solvers share a consistent Python interface:

```python
from experiments.comparative_experiment import load_instance
from solvers import gurobi_solver, benders_solver

inst = load_instance("instances/S_01.json.gz")

# Extensive-form Branch-and-Cut with problem-specific valid inequalities
res_bc = gurobi_solver.solve(
    inst, time_limit=3600, gap_tol=0, valid_inequalities=True, return_first_stage=True
)

# Callback-based multi-cut Benders decomposition
res_bd = benders_solver.solve(
    inst, time_limit=3600, gap_tol=0, return_first_stage=True
)

print("B&C obj:", res_bc["obj_val"], "time:", res_bc["time"])
print("BD  obj:", res_bd["obj_val"], "time:", res_bd["time"])
```

---

## Implementation Notes

- **Probability Normalization**: Scenario probabilities stored in `instances/*.json.gz` are rounded to 6 decimal places on disk and normalized via `math.fsum` to sum to exactly `1.0` upon loading in all experiment scripts and solvers.
- **Hardware Incompatibility Set ($\Psi_{\text{incomp}}$)**: Both solvers enforce $x_{jk,s} = 0$ and $e_{jk,s} = 0$ for any incompatible pair $(j, k) \in \Psi_{\text{incomp}}$ (reflecting that a request exceeding device HBM cannot run on GPU type $k$ either on-premise or in the cloud). In all 25 provided benchmark instances and sensitivity sweeps, $\Psi_{\text{incomp}}$ is empty under the baseline GQA calibration.
- **Caching**: Experiment scripts cache completed runs in `cache/comparative/`, `cache/vss/`, and `cache/sensitivity/`. If you modify solver tolerances, instance files, or experiment constants, delete the corresponding `cache/` subdirectory before rerunning so stale entries are not reused.
- **Timing and Hardware Variability**: Wall-clock solve times and minor floating-point tie-breaks depend on CPU architecture, thread count, and Gurobi version; certified optimal objective values agree across platforms.

---

## Citation

If you find this repository or the benchmark instances useful in your work, please cite the accompanying paper (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{llm_capacity_planning_ijpe,
  title   = {Strategic Capacity Planning for Large Language Model Inference Services under Demand Uncertainty},
  author  = {The Authors},
  journal = {International Journal of Production Economics},
  year    = {2026},
  note    = {In Press}
}
```

---

## License

This project (including the source code and benchmark instances) is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
