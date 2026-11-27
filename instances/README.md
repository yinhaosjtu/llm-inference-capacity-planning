# Benchmark Instances

This directory contains the 25 gzip-compressed JSON benchmark instances (`.json.gz`) used in all numerical experiments of the paper *Strategic Capacity Planning for Large Language Model Inference Services under Demand Uncertainty* (*International Journal of Production Economics*).

## Instance Summary

| Scale Class | Files | Count | $\|K\|$ (GPU Types) | $\|J\|$ (Service Classes) | $\|S\|$ (Scenarios) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **S-class** (Small) | `S_01.json.gz` – `S_10.json.gz` | 10 | 10 | 10 | 1,000 |
| **M-class** (Medium) | `M_01.json.gz` – `M_10.json.gz` | 10 | 15 | 15 | 2,000 |
| **L-class** (Large) | `L_01.json.gz` – `L_05.json.gz` | 5 | 30 | 30 | 3,000 |

- **Comparative benchmark** (`experiments/comparative_experiment.py`): uses all 25 instances (`S_01`–`S_10`, `M_01`–`M_10`, `L_01`–`L_05`).
- **Risk-adjusted VSS & EVPI** (`experiments/vss_and_evpi.py`): uses the first 5 small instances (`S_01`–`S_05`) and first 5 medium instances (`M_01`–`M_05`).
- **Sensitivity, risk aversion, and robustness analyses** (`experiments/sensitivity_analysis.py`): uses `M_01.json.gz` as the base instance structure and resamples scenario sets across 10 random seeds (`rep = 1..10`, and `rep = 1..3` for the $\rho \times \lambda$ frontier).

## File Format and Loading Conventions

Each `.json.gz` file stores a UTF-8 JSON object. Because JSON requires string dictionary keys:

- Integer-indexed maps (e.g., `C_cap`, `D_bar`, `p`) are serialized with stringified integer keys (`"0"`, `"1"`, …).
- Pair-indexed scenario maps (`D`, `L_out`, `L_in`) are serialized with comma-separated keys (`"j,s"`).
- `Psi_incomp` is stored as a list of `[j, k]` pairs.
- **Per-slot service capacity (`nu_scen`) is omitted on disk** to keep archive sizes compact (~15 MB total) and is reconstructed automatically upon loading via:
  $$\nu_{jk,s} = \begin{cases} 0, & (j, k) \in \Psi_{\text{incomp}}, \\ \operatorname{round}\!\left(\chi \dfrac{T^{\text{plan}}}{L^{\text{out}}_{j,s} / \Theta_k + \tau^{\text{overhead}}},\, 2\right), & \text{otherwise.} \end{cases}$$
- **Scenario probabilities (`p`)** are stored rounded to 6 decimal places on disk and automatically normalized by `math.fsum` to sum to exactly `1.0` in all experiment loaders and solvers.

You can load any instance directly with the built-in loader:

```python
from experiments.comparative_experiment import load_instance

inst = load_instance("instances/S_01.json.gz")
print(len(inst["K"]), len(inst["J"]), len(inst["S"]), sum(inst["p"].values()))
```

## Data Dictionary

All 46 fields stored in each instance file are documented below, mapped to the notation in Table 1 of the paper.

### Sets and Compatibility

| JSON Key | Paper Symbol | JSON Format | Description |
| :--- | :---: | :--- | :--- |
| `K` | $K$ | `list[int]` | Indices of available GPU types (`0 .. \|K\|-1`). |
| `J` | $J$ | `list[int]` | Indices of LLM inference service classes (`0 .. \|J\|-1`). |
| `S` | $S$ | `list[int]` | Indices of workload scenarios (`0 .. \|S\|-1`). |
| `Psi_incomp` | $\Psi_{\text{incomp}}$ | `list[[int, int]]` | Incompatible `(j, k)` service–GPU pairs exceeding the single-request HBM feasibility threshold. |

### GPU Hardware and Cost Parameters (indexed by `k`)

| JSON Key | Paper Symbol | Unit | Description |
| :--- | :---: | :--- | :--- |
| `C_cap` | $C_k^{\text{cap}}$ | USD / quarter | Amortized capital procurement cost per GPU of type $k$. |
| `C_elast` | $C_k^{\text{elast}}$ | USD / request | Elastic cloud offloading cost per request on GPU type $k$. |
| `M` | $M_k$ | GB | Effective HBM memory available for KV cache on GPU type $k$. |
| `N_bar` | $\bar{N}_k$ | units | Maximum procurement supply limit for GPU type $k$. |
| `Theta` | $\Theta_k$ | tokens / s | Effective steady-state decode token generation throughput of GPU type $k$. |
| `P_idle` | $P_k^{\text{idle}}$ | kW | Idle power consumption of GPU type $k$. |
| `P_peak` | $P_k^{\text{peak}}$ | kW | Peak power consumption of GPU type $k$. |
| `gpu_tiers` | — | string | Hardware tier label (`ultra-high`, `high-end`, `mid-range`, or `standard`) for fleet composition reporting. |

### Service Class Parameters (indexed by `j`)

| JSON Key | Paper Symbol | Unit | Description |
| :--- | :---: | :--- | :--- |
| `D_bar` | $\bar{D}_j$ | requests / quarter | Calibrated baseline mean demand for service class $j$. |
| `demand_cv` | $\sigma_j^{D}$ | dimensionless | Calibrated log-scale dispersion parameter of demand for class $j$ (used in sensitivity resampling). |
| `L_in_mean` | $\bar{L}_j^{\text{in}}$ | tokens | Calibrated baseline mean input sequence length for class $j$. |
| `L_in_logsigma` | $\sigma_j^{I}$ | dimensionless | Calibrated log-scale standard deviation of input sequence length for class $j$. |
| `L_out_bar` | $\bar{L}_j^{\text{out}}$ | tokens | Calibrated baseline mean output sequence length for class $j$. |
| `L_out_mu` | $\mu_j^{L}$ | log-tokens | Calibrated log-normal location parameter of output sequence length for class $j$. |
| `L_out_sigma` | $\sigma_j^{L}$ | dimensionless | Calibrated log-normal scale parameter of output sequence length for class $j$. |
| `sigma_in` | $\sigma_j^{\text{in}}$ | tokens | Marginal standard deviation of input sequence length for class $j$. |
| `sigma_out` | $\sigma_j^{\text{out}}$ | tokens | Marginal standard deviation of output sequence length for class $j$. |
| `sigma_tot` | $\sigma_j^{\text{tot}}$ | tokens | Deterministic estimate of total sequence-length standard deviation entering the safety-capacity constraint. |
| `C_rej` | $C_j^{\text{rej}}$ | USD / request | Penalty cost for rejecting a request of class $j$. |

### System, Sustainability, and Risk Parameters (scalars)

| JSON Key | Paper Symbol | Unit / Baseline Value | Description |
| :--- | :---: | :--- | :--- |
| `mu_kv` | $\mu^{\text{kv}}$ | GB / token (`0.00032768`) | Per-token KV-cache memory footprint under FP16 GQA calibration. |
| `xi` | $\xi$ | dimensionless (`0.75`) | Effective time-averaged KV-cache memory utilization factor. |
| `beta` | $\beta$ | dimensionless (`[2.0, 2.5]`) | Safety factor in the distribution-free intra-period memory overflow constraint. |
| `Z_min` | $Z_{\min}$ | dimensionless (`0.05`) | Lower bound on the memory reservation ratio $z_k$. |
| `Z_max` | $Z_{\max}$ | dimensionless (`0.30`) | Upper bound on the memory reservation ratio $z_k$. |
| `tau_overhead` | $\tau^{\text{overhead}}$ | s / request (`0.05`) | Prefill latency and scheduling overhead per request. |
| `T_plan` | $T^{\text{plan}}$ | s (`7776000`) | Strategic planning horizon duration (one 90-day quarter). |
| `T_slot` | $T^{\text{slot}}$ | s (`3600`) | Tactical peak-load time slot duration (one hour). |
| `T_op` | $T^{\text{op}}$ | hours (`2160`) | Operational hours per planning period (90 days × 24 h). |
| `nu_util_factor` | $\chi$ | dimensionless (`0.036`) | Sustained-utilization factor discounting raw generation turnover in $\nu_{jk,s}$. |
| `C_energy` | $C^{\text{energy}}$ | USD / kWh (`0.10`) | Unit electricity cost. |
| `rho` | $\rho$ | dimensionless (`1.5`) | Data-center Power Usage Effectiveness (PUE). |
| `kappa` | $\kappa$ | kg CO$_2$ / kWh (`0.5`) | Grid carbon intensity. |
| `W_bar` | $\bar{W}$ | kW | Total cluster peak power budget. |
| `E_bar` | $\bar{E}$ | kg CO$_2$ / quarter | Planning-period carbon emission budget. |
| `alpha` | $\alpha$ | dimensionless (`0.95`) | Confidence level for Conditional Value-at-Risk ($\text{CVaR}_\alpha$). |
| `lambda_risk` | $\lambda$ | dimensionless (`[0.20, 0.40]`) | Risk-aversion weight balancing expected recourse and $\text{CVaR}_\alpha$. |
| `rho_corr` | $\rho_j$ | dimensionless (`0.3`) | Baseline demand–output Gaussian copula correlation parameter. |

### Scenario Realizations

| JSON Key | Paper Symbol | JSON Key Format / Unit | Description |
| :--- | :---: | :--- | :--- |
| `p` | $p_s$ | `"s"` / probability | Scenario probability (normalized to sum to 1 upon loading). |
| `D` | $D_{j,s}$ | `"j,s"` / requests | Realized quarterly request demand for class $j$ in scenario $s$. |
| `L_out` | $L_{j,s}^{\text{out}}$ | `"j,s"` / tokens | Realized output sequence length for class $j$ in scenario $s$. |
| `L_in` | $L_{j,s}^{\text{in}}$ | `"j,s"` / tokens | Realized input sequence length for class $j$ in scenario $s$. |
| `random_input` | — | `bool` (`true`) | Indicates that `L_in` is scenario-indexed `(j, s)` rather than class-constant `j`. |
