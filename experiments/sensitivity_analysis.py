# Sensitivity, risk-aversion, and robustness experiments (Figs. 3-9).
import os
import gc
import gzip
import glob
import json
import copy
import math
import sys
from typing import Dict, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.stats import gamma as _gamma, norm as _norm

from solvers import gurobi_solver
from solvers import benders_solver

PLOT_JSON = os.path.join(_PROJECT_ROOT, 'plot_data.json')

_CACHE_DEBUG = os.environ.get('CACHE_DEBUG', '0') == '1'

INSTANCES_DIR = os.path.join(_PROJECT_ROOT, 'instances')
BASE_SCALE = 'M'
TIME_LIMIT = 3600
GAP_TOL = 1e-6

_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'cache', 'sensitivity')
_CACHE_FIELDS = ('obj_val', 'gap', 'time', 'status', 'n', 'z', 'y',
                 'n_vals', 'z_vals', 'capital_cost', 'energy_cost',
                 'expected_recourse', 'scenario_recourse', 'cvar_95',
                 'total_gpus', 'avg_z', 'cloud_burst_ratio', 'tier_counts')
_INT_KEY_DICTS = ('n', 'z', 'y', 'n_vals', 'z_vals')
_NONCACHEABLE_STATUS = {1, 11, 14}


def _is_cacheable(res):
    return res.get('status') not in _NONCACHEABLE_STATUS


def _cache_get(key, announce=True):
    path = os.path.join(_CACHE_DIR, key + '.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            res = json.load(f)
    except Exception:
        return None
    for fld in _INT_KEY_DICTS:
        if isinstance(res.get(fld), dict):
            res[fld] = {int(k): v for k, v in res[fld].items()}
    if _CACHE_DEBUG and announce:
        print(f"[cache] HIT  {key}", flush=True)
    return res


def _cache_put(key, res):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    slim = {f: res.get(f) for f in _CACHE_FIELDS if f in res}
    path = os.path.join(_CACHE_DIR, key + '.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(slim, f)
    os.replace(tmp, path)
    if _CACHE_DEBUG:
        print(f"[cache] WRITE {key}", flush=True)


def _cached(key, solve_callable):
    if key is not None:
        hit = _cache_get(key)
        if hit is not None:
            return hit
        if _CACHE_DEBUG:
            print(f"[cache] MISS {key}", flush=True)
    res = solve_callable()
    if res.get('status') == 11:
        raise KeyboardInterrupt("Interrupted by user during solve (GRB.INTERRUPTED)")
    if key is not None and _is_cacheable(res):
        _cache_put(key, res)
    return res


def _probe_all(keys):
    out = {}
    for k in keys:
        hit = _cache_get(k, announce=False)
        if hit is None:
            return None
        out[k] = hit
    return out


_DEFAULT_SCENARIO_SEED = 0

RHO_BASELINE = 0.3
RHO_IO = 0.4
_TAIL_TRUNC_Z = 2.326348
LAMBDA = 0.3

SCENARIO_REPS = list(range(1, 11))
CLOUD_GAMMAS = [0.5, 0.75, 1.0, 1.5, 2.0]
MUKV_GAMMAS = [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
DEMAND_CV_GAMMAS = [0.5, 0.75, 1.0, 1.25, 1.5]
LAMBDA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

RHO_VALUES = [0.0, 0.3, 0.6, 0.9]
RHO_REPS = list(range(1, 11))
FRONTIER_RHO_VALUES = [0.0, 0.3, 0.6]
FRONTIER_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FRONTIER_REPS = list(range(1, 4))
DIST_REPS = list(range(1, 11))
OOS_EVAL_SEED_OFFSET = 1000
OOS_EVAL_SCEN = 3000

# Ten structured workload regimes: (label, demand_mult, length_mult, weight).
SCENARIO_TYPES = [
    ('low_demand_short',   0.6, 0.8,  1.5),
    ('low_demand_normal',  0.7, 1.0,  2.0),
    ('normal_short',       1.0, 0.85, 2.5),
    ('baseline',           1.0, 1.0,  4.0),
    ('normal_long',        1.0, 1.2,  2.5),
    ('high_demand_normal', 1.3, 1.0,  2.0),
    ('high_demand_long',   1.4, 1.3,  1.5),
    ('peak_moderate',      1.5, 1.1,  1.0),
    ('extreme_demand',     1.7, 1.0,  0.5),
    ('extreme_length',     1.2, 1.6,  0.5),
]

_TUPLE_KEY_FIELDS = {'D', 'L_out', 'nu_scen'}
_INT_KEY_FIELDS = {'C_cap', 'C_elast', 'M', 'N_bar', 'Theta', 'P_idle', 'P_peak',
                   'L_out_bar', 'sigma_out', 'sigma_in', 'sigma_tot', 'L_in_logsigma', 'C_rej', 'D_bar', 'p', 'gpu_tiers',
                   'L_in_mean', 'demand_cv', 'L_out_mu', 'L_out_sigma'}
_AUTO_KEY_FIELDS = {'L_in'}


def _open_json(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def restore_keys(inst):
    out = {}
    for field, val in inst.items():
        if isinstance(val, dict):
            nd = {}
            if field in _TUPLE_KEY_FIELDS:
                for k, v in val.items():
                    nd[tuple(int(x) for x in k.split(','))] = v
            elif field in _INT_KEY_FIELDS:
                for k, v in val.items():
                    nd[int(k)] = v
            elif field in _AUTO_KEY_FIELDS:
                for k, v in val.items():
                    if ',' in k:
                        nd[tuple(int(x) for x in k.split(','))] = v
                    else:
                        nd[int(k)] = v
            else:
                nd = val
            out[field] = nd
        elif field == 'Psi_incomp' and isinstance(val, list):
            out[field] = [tuple(x) for x in val]
        else:
            out[field] = val
    return out


def _ensure_nu_scen(inst):
    if inst.get('nu_scen'):
        return inst
    incomp = set(map(tuple, inst['Psi_incomp']))
    T_plan, tau, Theta = inst['T_plan'], inst['tau_overhead'], inst['Theta']
    nu_f = inst.get('nu_util_factor', 1.0)
    L_out = inst['L_out']
    K, J, S = inst['K'], inst['J'], inst['S']
    nu_scen = {}
    for s in S:
        for j in J:
            lo = L_out[(j, s)]
            for k in K:
                nu_scen[(j, k, s)] = 0.0 if (j, k) in incomp else round(
                    nu_f * T_plan / (lo / Theta[k] + tau), 2)
    inst['nu_scen'] = nu_scen
    return inst


def _normalize_probabilities(inst):
    S = inst['S']
    probs = {s: float(inst['p'][s]) for s in S}
    total = math.fsum(probs.values())
    if total > 0:
        inst['p'] = {s: v / total for s, v in probs.items()}
    return inst


# Load base instance structure from .json(.gz) and normalize probabilities.
def load_instance(path):
    with _open_json(path) as f:
        return _normalize_probabilities(_ensure_nu_scen(restore_keys(json.load(f))))


def _instance_stem(path):
    base = os.path.basename(path)
    for ext in ('.json.gz', '.json'):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def _index_of(path):
    stem = _instance_stem(path)
    try:
        return int(stem.split('_')[-1])
    except ValueError:
        return 10 ** 9


def find_base_file(instances_dir=INSTANCES_DIR, scale=BASE_SCALE):
    by_stem = {}
    for ext in ('.json', '.json.gz'):
        for p in glob.glob(os.path.join(instances_dir, f"{scale}_*{ext}")):
            by_stem[_instance_stem(p)] = p
    paths = list(by_stem.values())
    return min(paths, key=_index_of) if paths else None


def _scenario_split(num_scenarios):
    num_types = len(SCENARIO_TYPES)
    per_type = max(1, num_scenarios // num_types)
    counts, s_idx = [], 0
    for type_idx in range(num_types):
        n_this = ((num_scenarios - s_idx) if type_idx == num_types - 1
                  else min(per_type, num_scenarios - s_idx))
        n_this = max(0, n_this)
        counts.append(n_this)
        s_idx += n_this
    return counts


# Normalize probabilities and update dispersions, Psi_incomp, and nu_scen.
def _finalize(inst, S, p, D, L_out, cv_scale, scenario_seed, family, L_in=None, rho_io=RHO_IO):
    total_p = math.fsum(p.values())
    p = {s: v / total_p for s, v in p.items()}
    inst.update({'p': p, 'D': D, 'L_out': L_out})
    if L_in is not None:
        inst['L_in'] = L_in
        inst['random_input'] = True

    incomp_set = set(map(tuple, inst['Psi_incomp']))
    T_plan, tau, Theta = inst['T_plan'], inst['tau_overhead'], inst['Theta']
    nu_f = inst.get('nu_util_factor', 1.0)
    K, J = inst['K'], inst['J']

    L_in_logsigma = inst['L_in_logsigma']
    L_in_mean = inst['L_in_mean']
    L_out_mu = inst['L_out_mu']
    L_out_sigma = inst['L_out_sigma']
    sigma_in = inst.get('sigma_in', {j: 0.0 for j in J})
    if abs(cv_scale - 1.0) > 1e-12:
        sigma_out = {}
        sigma_in = {}
        L_out_bar = {}
        for j in J:
            sigma_L0 = L_out_sigma[j]
            sigma_L = max(sigma_L0 * cv_scale, 0.05)
            L_out_mean = float(np.exp(L_out_mu[j] + 0.5 * sigma_L0 ** 2))
            L_out_bar[j] = int(L_out_mean)
            sigma_out[j] = round(L_out_mean * np.sqrt(np.exp(sigma_L ** 2) - 1.0), 2)
            s_in = max(L_in_logsigma[j] * cv_scale, 1e-9)
            cv_in = np.sqrt(np.exp(s_in ** 2) - 1.0)
            sigma_in[j] = round(cv_in * L_in_mean[j], 2)
        inst['L_out_bar'] = L_out_bar
        inst['sigma_out'] = sigma_out
        inst['sigma_in'] = sigma_in
    else:
        sigma_out = inst['sigma_out']

    if L_in is not None:
        sigma_tot = {}
        for j in J:
            so = sigma_out[j]
            si = sigma_in.get(j, 0.0)
            sigma_tot[j] = round(np.sqrt(si ** 2 + 2 * rho_io * si * so + so ** 2), 2)
        inst['sigma_tot'] = sigma_tot
        if abs(cv_scale - 1.0) > 1e-12:
            mu_kv = inst['mu_kv']
            M = inst['M']
            Psi = []
            for j in J:
                max_mem = (L_in_mean[j] + 3 * sigma_in[j]
                           + inst['L_out_bar'][j] + 3 * sigma_out[j]) * mu_kv
                for k in K:
                    if max_mem > M[k] * 0.9:
                        Psi.append((j, k))
            inst['Psi_incomp'] = Psi
            incomp_set = set(map(tuple, Psi))
    else:
        inst['sigma_tot'] = dict(sigma_out)

    nu_scen = {}
    for s in S:
        for j in J:
            for k in K:
                nu_scen[(j, k, s)] = 0.0 if (j, k) in incomp_set else round(
                    nu_f * T_plan / (L_out[(j, s)] / Theta[k] + tau), 2)
    inst['nu_scen'] = nu_scen

    inst['cv_scale'] = cv_scale
    inst['scenario_seed'] = scenario_seed
    inst['dist_family'] = family
    return inst


# Generate moment-matched Gamma scenarios with Gaussian-copula dependence.
def add_scenarios_gamma(structure, num_scenarios, scenario_seed, cv_scale=1.0, rho=None, rho_io=RHO_IO):
    inst = copy.deepcopy(structure)
    rng = np.random.default_rng(scenario_seed)
    K, J = inst['K'], inst['J']
    D_bar = inst['D_bar']
    demand_cv = inst['demand_cv']
    L_out_mu = inst['L_out_mu']
    L_out_sigma = inst['L_out_sigma']
    if rho is None:
        rho = inst.get('rho_corr', RHO_BASELINE)
    chol = np.sqrt(max(0.0, 1.0 - rho ** 2))

    S = list(range(num_scenarios))
    inst['S'] = S
    counts = _scenario_split(num_scenarios)
    p, D, L_out, L_in = {}, {}, {}, {}
    L_in_mean = inst['L_in_mean']
    L_in_logsigma = inst['L_in_logsigma']
    chol_io = np.sqrt(max(0.0, 1.0 - rho_io ** 2))
    s_idx = 0
    for (_, d_mult, l_mult, weight), n_this in zip(SCENARIO_TYPES, counts):
        for _ in range(n_this):
            p[s_idx] = weight * (1 + rng.uniform(-0.1, 0.1))
            for j in J:
                sigma_D = max(demand_cv[j] * cv_scale, 0.05)
                sigma_L0 = L_out_sigma[j]
                sigma_L = max(sigma_L0 * cv_scale, 0.05)
                D_mean = D_bar[j] * d_mult
                mean_L = float(np.exp(L_out_mu[j] + 0.5 * sigma_L0 ** 2)) * l_mult
                vfD = float(np.expm1(sigma_D ** 2))
                kD, thD = 1.0 / vfD, vfD * D_mean
                vfL = float(np.expm1(sigma_L ** 2))
                kL, thL = 1.0 / vfL, vfL * mean_L
                mean_in = float(L_in_mean[j])
                s_in_g = max(L_in_logsigma[j] * cv_scale, 1e-9)
                vfI = float(np.expm1(s_in_g ** 2))
                kI, thI = 1.0 / vfI, vfI * mean_in
                w1, w2, w3 = rng.standard_normal(3)
                z1 = w1
                z2 = rho * w1 + chol * w2
                z3 = rho_io * z2 + chol_io * w3
                z1 = min(max(z1, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)
                z2 = min(max(z2, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)
                z3 = min(max(z3, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)
                u1 = min(max(float(_norm.cdf(z1)), 1e-10), 1 - 1e-10)
                u2 = min(max(float(_norm.cdf(z2)), 1e-10), 1 - 1e-10)
                u3 = min(max(float(_norm.cdf(z3)), 1e-10), 1 - 1e-10)
                d_val = float(_gamma.ppf(u1, kD, scale=thD))
                l_val = float(_gamma.ppf(u2, kL, scale=thL))
                in_val = float(_gamma.ppf(u3, kI, scale=thI))
                D[(j, s_idx)] = round(max(1000.0, d_val), 0)
                L_out[(j, s_idx)] = round(max(20.0, l_val), 2)
                L_in[(j, s_idx)] = round(max(1.0, in_val), 2)
            s_idx += 1
    return _finalize(inst, S, p, D, L_out, cv_scale, scenario_seed, 'gamma', L_in=L_in, rho_io=rho_io)


# Build a perturbed instance for a given multiplier, correlation, and distribution family.
def make_instance(base, scenario_seed=None, cv_scale=1.0,
                  cloud_gamma=1.0, mukv_gamma=1.0, rho=None, family='lognormal',
                  kappa_in=1.0, rho_io=RHO_IO, num_scen_override=None):
    if scenario_seed is None:
        scenario_seed = _DEFAULT_SCENARIO_SEED
    struct = copy.deepcopy(base)
    if rho is not None:
        struct['rho_corr'] = rho

    if cloud_gamma != 1.0:
        struct['C_elast'] = {k: round(base['C_elast'][k] * cloud_gamma, 6) for k in struct['K']}

    if mukv_gamma != 1.0:
        new_mu = base['mu_kv'] * mukv_gamma
        struct['mu_kv'] = round(new_mu, 10)
        Psi = []
        for j in struct['J']:
            max_mem = (struct['L_in_mean'][j] + 3 * struct['sigma_in'][j]
                       + struct['L_out_bar'][j] + 3 * struct['sigma_out'][j]) * new_mu
            for k in struct['K']:
                if max_mem > struct['M'][k] * 0.9:
                    Psi.append((j, k))
        struct['Psi_incomp'] = Psi

    num_scen = num_scen_override or base.get('num_scenarios', len(base['S']))
    if family == 'gamma':
        inst = add_scenarios_gamma(struct, num_scen, scenario_seed, cv_scale=cv_scale,
                                   rho=rho, rho_io=rho_io)
    else:
        inst = add_scenarios_random_input(struct, num_scen, scenario_seed,
                                          kappa_in=kappa_in, rho_io=rho_io,
                                          cv_scale=cv_scale)
    inst['Psi_incomp'] = [tuple(x) for x in inst['Psi_incomp']]
    return inst


# Generate Gaussian-copula log-normal scenarios for (D, L_out, L_in).
def add_scenarios_random_input(structure: Dict[str, Any], num_scenarios: int,
                               scenario_seed: int, kappa_in: float, rho_io: float,
                               cv_scale: float = 1.0) -> Dict[str, Any]:
    inst = copy.deepcopy(structure)
    rng = np.random.default_rng(scenario_seed)

    K = inst['K']
    J = inst['J']
    D_bar = inst['D_bar']
    demand_cv = inst['demand_cv']
    L_out_mu = inst['L_out_mu']
    L_out_sigma = inst['L_out_sigma']
    Theta = inst['Theta']
    L_in_mean = inst['L_in_mean']
    rho = inst.get('rho_corr', RHO_BASELINE)
    chol = np.sqrt(max(0.0, 1.0 - rho ** 2))
    chol_io = np.sqrt(max(0.0, 1.0 - rho_io ** 2))

    mu_in_of, sigma_in_of = {}, {}
    L_in_logsigma = inst['L_in_logsigma']
    for j in J:
        s_in_base = max(L_in_logsigma[j] * cv_scale, 1e-9)
        cv_in_base = np.sqrt(np.exp(s_in_base ** 2) - 1.0)
        cv_in_j = max(kappa_in * cv_in_base, 1e-9)
        s_in = np.sqrt(np.log(1.0 + cv_in_j ** 2))
        sigma_in_of[j] = s_in
        mu_in_of[j] = np.log(max(1.0, float(L_in_mean[j]))) - 0.5 * s_in ** 2

    S = list(range(num_scenarios))
    inst['S'] = S
    num_types = len(SCENARIO_TYPES)
    per_type = max(1, num_scenarios // num_types)
    p, D, L_out, L_in = {}, {}, {}, {}
    s_idx = 0

    for type_idx, (_, d_mult, l_mult, weight) in enumerate(SCENARIO_TYPES):
        n_this = ((num_scenarios - s_idx) if type_idx == num_types - 1
                  else min(per_type, num_scenarios - s_idx))
        for _ in range(n_this):
            if s_idx >= num_scenarios:
                break
            p[s_idx] = weight * (1 + rng.uniform(-0.1, 0.1))
            for j in J:
                sigma_D = max(demand_cv[j] * cv_scale, 0.05)
                sigma_L = max(L_out_sigma[j] * cv_scale, 0.05)
                D_mean = D_bar[j] * d_mult
                mu_D = np.log(D_mean) - 0.5 * sigma_D ** 2
                sigma_L0 = L_out_sigma[j]
                L_out_mean = float(np.exp(L_out_mu[j] + 0.5 * sigma_L0 ** 2)) * l_mult
                mu_L = np.log(L_out_mean) - 0.5 * sigma_L ** 2

                w1, w2, w3 = rng.standard_normal(3)
                z1 = w1
                z2 = rho * w1 + chol * w2
                z3 = rho_io * z2 + chol_io * w3

                z1 = min(max(z1, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)
                z2 = min(max(z2, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)
                z3 = min(max(z3, -_TAIL_TRUNC_Z), _TAIL_TRUNC_Z)

                D[(j, s_idx)] = round(max(1000.0, float(np.exp(mu_D + sigma_D * z1))), 0)
                L_out[(j, s_idx)] = round(max(20.0, float(np.exp(mu_L + sigma_L * z2))), 2)
                L_in[(j, s_idx)] = round(max(1.0, float(np.exp(mu_in_of[j] + sigma_in_of[j] * z3))), 2)
            s_idx += 1

    total_p = math.fsum(p.values())
    p = {s: v / total_p for s, v in p.items()}
    inst.update({'p': p, 'D': D, 'L_out': L_out, 'L_in': L_in})

    if abs(cv_scale - 1.0) > 1e-12:
        mu_kv = inst['mu_kv']
        M = inst['M']
        L_in_mean = inst['L_in_mean']
        sigma_out, L_out_bar = {}, {}
        for j in J:
            sigma_L = max(L_out_sigma[j] * cv_scale, 0.05)
            sigma_L0 = L_out_sigma[j]
            L_out_mean = float(np.exp(L_out_mu[j] + 0.5 * sigma_L0 ** 2))
            L_out_bar[j] = int(L_out_mean)
            sigma_out[j] = round(L_out_mean * np.sqrt(np.exp(sigma_L ** 2) - 1.0), 2)
        inst['sigma_out'] = sigma_out
        inst['L_out_bar'] = L_out_bar

    sigma_out_eff = inst['sigma_out']
    L_in_mean = inst['L_in_mean']
    sigma_in_eff, sigma_tot_eff = {}, {}
    for j in J:
        s_in_base = max(L_in_logsigma[j] * cv_scale, 1e-9)
        cv_in_base = np.sqrt(np.exp(s_in_base ** 2) - 1.0)
        cv_in_j = max(kappa_in * cv_in_base, 1e-9)
        so = sigma_out_eff[j]
        si = round(cv_in_j * float(L_in_mean[j]), 2)
        sigma_in_eff[j] = si
        sigma_tot_eff[j] = round(
            np.sqrt(si ** 2 + 2 * rho_io * si * so + so ** 2), 2)
    inst['sigma_in'] = sigma_in_eff
    inst['sigma_tot'] = sigma_tot_eff

    mu_kv_eff = inst['mu_kv']
    M = inst['M']
    Psi = []
    for j in J:
        max_mem = (L_in_mean[j] + 3 * sigma_in_eff[j]
                   + inst['L_out_bar'][j] + 3 * sigma_out_eff[j]) * mu_kv_eff
        for k in K:
            if max_mem > M[k] * 0.9:
                Psi.append((j, k))
    inst['Psi_incomp'] = Psi

    incomp_set = set(map(tuple, inst['Psi_incomp']))
    T_plan = inst['T_plan']
    tau = inst['tau_overhead']
    nu_f = inst.get('nu_util_factor', 1.0)
    nu_scen = {}
    for s in S:
        for j in J:
            for k in K:
                nu_scen[(j, k, s)] = (0.0 if (j, k) in incomp_set
                                      else round(nu_f * T_plan / (L_out[(j, s)] / Theta[k] + tau), 2))
    inst['nu_scen'] = nu_scen

    inst['cv_scale'] = cv_scale
    inst['scenario_seed'] = scenario_seed
    inst['kappa_in'] = kappa_in
    inst['rho_io'] = rho_io
    inst['random_input'] = True
    return inst


def _free_solve(inst, lam, details=False, first=False):
    return benders_solver.solve(
        inst, time_limit=TIME_LIMIT, gap_tol=GAP_TOL,
        lambda_override=lam, return_details=details, return_first_stage=first)


# Use EF LP when stage 1 is fixed; otherwise solve the full two-stage MILP via Benders.
def _solve(inst, lam=LAMBDA, fix=None, details=False, first=False, vi=True,
           cache_key=None):
    if fix is not None:
        return _cached(cache_key, lambda: gurobi_solver.solve(
            inst, time_limit=TIME_LIMIT, gap_tol=GAP_TOL,
            lambda_override=lam, valid_inequalities=vi,
            fix_first_stage=fix, return_details=details,
            return_first_stage=first, scenario_input=True))
    return _cached(cache_key, lambda: _free_solve(inst, lam, details=details, first=first))


# One-at-a-time sensitivity sweeps over cloud cost, mu_kv, and workload variability.
def run_sensitivity(base, state):
    sens = state['sens']

    def solve(inst, key):
        return _cached(key, lambda: _free_solve(inst, LAMBDA, details=True))

    print("\n[Sensitivity 1/3] Elastic cloud cost multiplier gamma^cloud")
    for g in CLOUD_GAMMAS:
        sens['cloud_cost'][g] = []
        for sd in SCENARIO_REPS:
            key = f"cloud__g{g}__rep{sd}"
            r = _cache_get(key)
            if r is None:
                inst = make_instance(base, scenario_seed=sd, cv_scale=1.0, cloud_gamma=g)
                r = solve(inst, key)
                del inst
            sens['cloud_cost'][g].append(r)
            print(f"  cloud g={g} rep={sd}: Obj={r['obj_val']:.1f}" if r['obj_val'] is not None else
                  f"  cloud g={g} rep={sd}: FAILED")
            del r
            gc.collect()

    print("\n[Sensitivity 2/3] KV-cache memory coefficient multiplier gamma^mem")
    for g in MUKV_GAMMAS:
        sens['mukv'][g] = []
        for sd in SCENARIO_REPS:
            key = f"mukv__g{g}__rep{sd}"
            r = _cache_get(key)
            if r is None:
                inst = make_instance(base, scenario_seed=sd, cv_scale=1.0, mukv_gamma=g)
                r = solve(inst, key)
                del inst
            sens['mukv'][g].append(r)
            print(f"  mukv g={g} rep={sd}: Obj={r['obj_val']:.1f}" if r['obj_val'] is not None else
                  f"  mukv g={g} rep={sd}: FAILED")
            del r
            gc.collect()

    print("\n[Sensitivity 3/3] Workload variability scaling factor gamma^cv")
    for g in DEMAND_CV_GAMMAS:
        sens['demand_cv'][g] = []
        for sd in SCENARIO_REPS:
            key = f"demandcv__g{g}__rep{sd}"
            r = _cache_get(key)
            if r is None:
                inst = make_instance(base, scenario_seed=sd, cv_scale=g)
                r = solve(inst, key)
                del inst
            sens['demand_cv'][g].append(r)
            print(f"  cv g={g} rep={sd}: Obj={r['obj_val']:.1f}" if r['obj_val'] is not None else
                  f"  cv g={g} rep={sd}: FAILED")
            del r
            gc.collect()
    save_plot_json(state, announce=False)


# Risk-aversion weight (lambda) sweep for mean-CVaR trade-off and fleet decisions.
def run_risk(base, state):
    risk = state['risk']
    print("\n[Risk Analysis] Risk-aversion weight lambda sweep")
    for lam in LAMBDA_VALUES:
        risk[lam] = []
        for sd in SCENARIO_REPS:
            key = f"risk__lam{lam}__rep{sd}"
            r = _cache_get(key)
            if r is None:
                inst = make_instance(base, scenario_seed=sd, cv_scale=1.0)
                r = _cached(key, lambda inst=inst, lam=lam:
                            _free_solve(inst, lam, details=True))
                del inst
            risk[lam].append(r)
            print(f"  lambda={lam} rep={sd}: Obj={r['obj_val']:.1f}" if r['obj_val'] is not None else
                  f"  lambda={lam} rep={sd}: FAILED")
            del r
            gc.collect()
    save_plot_json(state, announce=False)


# Robustness sweep over demand-output correlation rho_j.
def run_rho_sweep(base, state):
    print("\n[Robustness 1/3] Correlation rho sweep  (lambda=%.2f, cv=1.0)" % LAMBDA)
    for rho in RHO_VALUES:
        for sd in RHO_REPS:
            key = f"rho__r{rho}__rep{sd}"
            r = _cache_get(key)
            if r is None:
                inst = make_instance(base, scenario_seed=sd, cv_scale=1.0, rho=rho)
                r = _solve(inst, details=True, cache_key=key)
                del inst
            if r['obj_val'] is None:
                print(f"  rho={rho} rep={sd}: FAILED")
                del r
                gc.collect()
                continue
            tc = r['tier_counts']
            state['rho'].append([
                rho, sd, round(r['obj_val'], 2), round(r['capital_cost'], 2),
                round(r['energy_cost'], 2), round(r['expected_recourse'], 2),
                round(r['cvar_95'], 2), r['total_gpus'], round(r['avg_z'], 4),
                round(r['cloud_burst_ratio'] * 100, 2),
                tc.get('ultra-high', 0), tc.get('high-end', 0),
                tc.get('mid-range', 0), tc.get('standard', 0), round(r['time'], 2)])
            print(f"  rho={rho} rep={sd}: Obj={r['obj_val']:.1f} "
                  f"GPUs={r['total_gpus']} avg_z={r['avg_z']:.3f} CVaR={r['cvar_95']:.0f}")
            del r
            gc.collect()
    save_plot_json(state)


# Compute mean-CVaR frontiers across correlation levels rho_j.
def run_rho_frontier(base, state):
    print("\n[Robustness 2/3] rho x lambda risk frontier")
    for rho in FRONTIER_RHO_VALUES:
        for lam in FRONTIER_LAMBDAS:
            for sd in FRONTIER_REPS:
                key = f"frontier__r{rho}__lam{lam}__rep{sd}"
                r = _cache_get(key)
                if r is None:
                    inst = make_instance(base, scenario_seed=sd, cv_scale=1.0, rho=rho)
                    r = _solve(inst, lam=lam, details=True, cache_key=key)
                    del inst
                if r['obj_val'] is None:
                    print(f"  rho={rho} lam={lam} rep={sd}: FAILED")
                    del r
                    gc.collect()
                    continue
                state['frontier'].append([
                    rho, lam, sd, round(r['expected_recourse'], 2), round(r['cvar_95'], 2),
                    round(r['obj_val'], 2), round(r['capital_cost'], 2),
                    r['total_gpus'], round(r['avg_z'], 4), round(r['time'], 2)])
                print(f"  rho={rho} lam={lam} rep={sd}: "
                      f"E_Rec={r['expected_recourse']:.0f} CVaR={r['cvar_95']:.0f}")
                del r
                gc.collect()
    save_plot_json(state)


# Out-of-sample cross-family transfer test between log-normal and moment-matched Gamma.
def run_distribution_oos(base, state):
    print("\n[Robustness 3/3] Out-of-sample distribution transfer  "
          "(lognormal <-> moment-matched Gamma, rho=%.2f, lambda=%.2f)"
          % (RHO_BASELINE, LAMBDA))
    for sd in DIST_REPS:
        k_ln = f"dist__rep{sd}__ln"
        k_ga = f"dist__rep{sd}__ga"
        k_lne = f"dist__rep{sd}__lnEVAL"
        k_gae = f"dist__rep{sd}__gaEVAL"
        k_lng = f"dist__rep{sd}__lnON_ga"
        k_gal = f"dist__rep{sd}__gaON_ln"
        cached = _probe_all([k_ln, k_ga, k_lne, k_gae, k_lng, k_gal])
        if cached is not None:
            rln, rga = cached[k_ln], cached[k_ga]
            ln_own = cached[k_lne]['obj_val']
            ga_own = cached[k_gae]['obj_val']
            lnplan_ga = cached[k_lng]['obj_val']
            gaplan_ln = cached[k_gal]['obj_val']
        else:
            ln = make_instance(base, scenario_seed=sd, cv_scale=1.0,
                               rho=RHO_BASELINE, family='lognormal')
            ga = make_instance(base, scenario_seed=sd, cv_scale=1.0,
                               rho=RHO_BASELINE, family='gamma')
            rln = _solve(ln, details=True, first=True, cache_key=k_ln)
            rga = _solve(ga, details=True, first=True, cache_key=k_ga)
            del ln, ga
            if rln['obj_val'] is None or rga['obj_val'] is None:
                print(f"  rep={sd}: training solve FAILED")
                del rln, rga
                gc.collect()
                continue
            plan_ln = {'n': rln['n'], 'y': rln['y']}
            plan_ga = {'n': rga['n'], 'y': rga['y']}

            eval_seed = sd + OOS_EVAL_SEED_OFFSET
            ln_eval = make_instance(base, scenario_seed=eval_seed, cv_scale=1.0,
                                    rho=RHO_BASELINE, family='lognormal',
                                    num_scen_override=OOS_EVAL_SCEN)
            ga_eval = make_instance(base, scenario_seed=eval_seed, cv_scale=1.0,
                                    rho=RHO_BASELINE, family='gamma',
                                    num_scen_override=OOS_EVAL_SCEN)
            ln_own = _solve(ln_eval, fix=plan_ln, vi=False, cache_key=k_lne)['obj_val']
            ga_own = _solve(ga_eval, fix=plan_ga, vi=False, cache_key=k_gae)['obj_val']
            lnplan_ga = _solve(ga_eval, fix=plan_ln, vi=False, cache_key=k_lng)['obj_val']
            gaplan_ln = _solve(ln_eval, fix=plan_ga, vi=False, cache_key=k_gal)['obj_val']
            del ln_eval, ga_eval

        if rln['obj_val'] is None or rga['obj_val'] is None:
            print(f"  rep={sd}: training solve FAILED")
            del rln, rga
            gc.collect()
            continue
        if None in (ln_own, ga_own, lnplan_ga, gaplan_ln):
            print(f"  rep={sd}: evaluation solve FAILED")
            del rln, rga
            gc.collect()
            continue

        reg_ln = (lnplan_ga - ga_own) / ga_own * 100 if ga_own > 1e-6 else None
        reg_ga = (gaplan_ln - ln_own) / ln_own * 100 if ln_own > 1e-6 else None

        state['dist'].append([
            sd, round(ln_own, 2), round(ga_own, 2),
            round(lnplan_ga, 2), (round(reg_ln, 3) if reg_ln is not None else None),
            round(gaplan_ln, 2), (round(reg_ga, 3) if reg_ga is not None else None),
            rln['total_gpus'], rga['total_gpus'],
            round(rln['avg_z'], 4), round(rga['avg_z'], 4)])
        rl = f"{reg_ln:.2f}%" if reg_ln is not None else "NA"
        rg = f"{reg_ga:.2f}%" if reg_ga is not None else "NA"
        print(f"  rep={sd}: LN_own={ln_own:.0f} GA_own={ga_own:.0f} | "
              f"LNplan@GA={lnplan_ga:.0f}(reg {rl}) GAplan@LN={gaplan_ln:.0f}(reg {rg})")
        del rln, rga
        gc.collect()
    save_plot_json(state)


def _mean(xs):
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _std(xs):
    xs = [v for v in xs if v is not None]
    return float(np.std(xs)) if xs else None


def _upsert_json(updates, path=PLOT_JSON):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(updates)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


# Aggregate experiment outputs into plot_data.json for Plot.py.
def save_plot_json(state, announce=False):
    out = {}
    sens = state['sens']

    for exp in ('cloud_cost', 'mukv', 'demand_cv'):
        if sens[exp]:
            gammas = sorted(sens[exp].keys())
            totals = [[r['obj_val'] for r in sens[exp][g] if r['obj_val'] is not None]
                      for g in gammas]
            out[f'sens_{exp}'] = {
                'gammas': gammas,
                'total_mean': [float(np.mean(t)) if t else None for t in totals],
                'total_std': [float(np.std(t)) if t else None for t in totals],
            }

    risk = state['risk']
    if risk:
        lams = sorted(risk.keys())

        def rk(lam, field):
            return [r[field] for r in risk[lam] if r['obj_val'] is not None]

        out['risk_pareto'] = {
            'lambda': lams,
            'e_recourse_mean': [_mean(rk(l, 'expected_recourse')) for l in lams],
            'cvar_mean': [_mean(rk(l, 'cvar_95')) for l in lams],
        }

        tail = {}
        for l in lams:
            vals = []
            for r in risk[l]:
                if r['obj_val'] is not None and r.get('scenario_recourse'):
                    vals.extend(r['scenario_recourse'])
            tail[str(l)] = vals
        out['risk_tail_ecdf'] = {'by_lambda': tail}

        out['risk_decisions'] = {
            'lambda': lams,
            'capital_mean': [_mean(rk(l, 'capital_cost')) for l in lams],
            'e_recourse_mean': [_mean(rk(l, 'expected_recourse')) for l in lams],
            'tier_means': {
                t: [_mean([r['tier_counts'].get(name, 0)
                           for r in risk[l] if r['obj_val'] is not None])
                    for l in lams]
                for t, name in [('ultra', 'ultra-high'), ('high', 'high-end'),
                                ('mid', 'mid-range'), ('standard', 'standard')]
            },
        }

    rho_rows = state['rho']
    if rho_rows:
        rhos = sorted({row[0] for row in rho_rows})

        def rho_col(rho, idx):
            return [row[idx] for row in rho_rows if row[0] == rho]

        out['robust_rho'] = {
            'rhos': rhos,
            'total_mean': [_mean(rho_col(r, 2)) for r in rhos],
            'total_std': [_std(rho_col(r, 2)) for r in rhos],
            'cvar_mean': [_mean(rho_col(r, 6)) for r in rhos],
            'cvar_std': [_std(rho_col(r, 6)) for r in rhos],
            'tier_means': {
                'ultra': [_mean(rho_col(r, 10)) for r in rhos],
                'high': [_mean(rho_col(r, 11)) for r in rhos],
                'mid': [_mean(rho_col(r, 12)) for r in rhos],
                'standard': [_mean(rho_col(r, 13)) for r in rhos],
            },
        }

    fr_rows = state['frontier']
    if fr_rows:
        by_rho = {}
        for row in fr_rows:
            by_rho.setdefault(row[0], {}).setdefault(row[1], {'e_rec': [], 'cvar': []})
            by_rho[row[0]][row[1]]['e_rec'].append(row[3])
            by_rho[row[0]][row[1]]['cvar'].append(row[4])
        frontier_out = {}
        for rho in sorted(by_rho):
            lam_keys = sorted(by_rho[rho])
            frontier_out[str(rho)] = {
                'lambdas': lam_keys,
                'e_rec_mean': [_mean(by_rho[rho][l]['e_rec']) for l in lam_keys],
                'cvar_mean': [_mean(by_rho[rho][l]['cvar']) for l in lam_keys],
            }
        out['robust_rho_frontier'] = {'by_rho': frontier_out}

    if state['dist']:
        out['robust_dist_oos'] = {
            'rows': [{'rep': row[0], 'ln_opt': row[1], 'ga_opt': row[2],
                      'lnplan_ga': row[3], 'reg_ln': row[4],
                      'gaplan_ln': row[5], 'reg_ga': row[6]}
                     for row in state['dist']]
        }

    if out:
        _upsert_json(out, path=PLOT_JSON)
        if announce:
            print(f"[OK] plot data saved -> {PLOT_JSON}")


def summarize_sensitivity_risk(state):
    sens = state['sens']
    risk = state['risk']
    print("\n--- Sensitivity summary (mean over seeds) ---")
    for exp, label in [('cloud_cost', 'Cloud'), ('mukv', 'mu_kv'), ('demand_cv', 'Demand CV')]:
        print(f"  {label}: ", end='')
        for g in sorted(sens[exp].keys()):
            valid = [r['obj_val'] for r in sens[exp][g] if r['obj_val'] is not None]
            if valid:
                print(f"g={g}:{np.mean(valid)/1000:.1f}k ", end='')
        print()
    print("--- Risk aversion summary (mean over seeds) ---")
    for lam in sorted(risk.keys()):
        valid = [r for r in risk[lam] if r['obj_val'] is not None]
        if valid:
            print(f"  lambda={lam}: Obj={np.mean([r['obj_val'] for r in valid])/1000:.1f}k "
                  f"Cap={np.mean([r['capital_cost'] for r in valid])/1000:.1f}k "
                  f"CVaR={np.mean([r['cvar_95'] for r in valid])/1000:.1f}k "
                  f"avg_z={np.mean([r['avg_z'] for r in valid]):.3f}")


def summarize_robustness(state):
    print("\n===== Summary =====")

    if state['rho']:
        print("\n[A] rho sweep (mean over seeds):")
        by_rho = {}
        for row in state['rho']:
            by_rho.setdefault(row[0], []).append(row)
        base_obj = None
        for rho in sorted(by_rho):
            rows = by_rho[rho]
            obj = np.mean([r[2] for r in rows])
            cap = np.mean([r[3] for r in rows])
            gpus = np.mean([r[7] for r in rows])
            z = np.mean([r[8] for r in rows])
            cvar = np.mean([r[6] for r in rows])
            if base_obj is None and abs(rho - RHO_BASELINE) < 1e-9:
                base_obj = obj
            print(f"  rho={rho}: Obj={obj/1000:.1f}k Cap={cap/1000:.1f}k "
                  f"GPUs={gpus:.1f} avg_z={z:.3f} CVaR={cvar/1000:.1f}k")
        if base_obj:
            objs = [np.mean([r[2] for r in by_rho[rho]]) for rho in sorted(by_rho)]
            spread = (max(objs) - min(objs)) / base_obj * 100
            print(f"  -> max relative total-cost variation across rho ~ {spread:.1f}% "
                  f"(relative to rho={RHO_BASELINE} baseline)")

    if state['dist']:
        print("\n[C] Distribution-family out-of-sample (mean over seeds):")
        reg_ln = [r[4] for r in state['dist'] if r[4] is not None]
        reg_ga = [r[6] for r in state['dist'] if r[6] is not None]
        ln_opt = np.mean([r[1] for r in state['dist']])
        ga_opt = np.mean([r[2] for r in state['dist']])
        print(f"  LN mean self-optimum={ln_opt/1000:.1f}k  GA mean self-optimum={ga_opt/1000:.1f}k "
              f"(same moments -> magnitudes should be close)")
        if reg_ln:
            print(f"  regret of LN plan on Gamma: mean {np.mean(reg_ln):.2f}%  "
                  f"max {np.max(reg_ln):.2f}%")
        if reg_ga:
            print(f"  regret of Gamma plan on LN: mean {np.mean(reg_ga):.2f}%  "
                  f"max {np.max(reg_ga):.2f}%")
        print("  -> smaller regret indicates the conclusions are driven by the first two "
              "moments and are insensitive to the distribution family.")


# Run all sensitivity, risk-aversion, and robustness experiments sequentially.
def main():
    base_file = find_base_file(INSTANCES_DIR, BASE_SCALE)
    if base_file is None:
        print(f"[ERROR] no {BASE_SCALE}_*.json(.gz) found in {INSTANCES_DIR}/")
        return
    base = load_instance(base_file)
    base_name = _instance_stem(base_file)
    num_scen = base.get('num_scenarios', len(base['S']))
    print(f"base structure = {base_name} ({BASE_SCALE}-scale first instance)  "
          f"|K|={len(base['K'])} |J|={len(base['J'])} |S|={num_scen}  "
          f"lambda={LAMBDA} gap={GAP_TOL}", flush=True)

    state = {
        'sens': {'cloud_cost': {}, 'mukv': {}, 'demand_cv': {}},
        'risk': {},
        'rho': [], 'frontier': [], 'dist': [],
    }

    run_sensitivity(base, state)
    run_risk(base, state)
    summarize_sensitivity_risk(state)

    run_rho_sweep(base, state)
    run_rho_frontier(base, state)
    run_distribution_oos(base, state)
    summarize_robustness(state)

    save_plot_json(state, announce=True)


if __name__ == "__main__":
    main()
