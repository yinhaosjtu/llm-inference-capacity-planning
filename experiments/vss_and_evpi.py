# Risk-adjusted VSS and EVPI evaluation under a unified mean-CVaR criterion (Tables 5 & 6).
import os
import gc
import gzip
import time
import json
import glob
import copy
import math
import argparse
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gurobipy as gp
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

from solvers import gurobi_solver
from solvers import benders_solver

INSTANCES_DIR = os.path.join(_PROJECT_ROOT, 'instances')
SCALES = ['S', 'M']
N_INSTANCES = 5
WS_WORKERS = min(4, os.cpu_count() or 1)

GAP_TOL = 0
TIME_LIMIT = gp.GRB.INFINITY
GAP_TRUST_THRESHOLD = 1e-6
CACHE_SCHEMA = 'risk-consistent-vss-evpi-v2'

_XLSX_HEADER_FILL = '4472C4'

_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'cache', 'vss')


def _is_cacheable(res):
    obj, gap = res.get('obj_val'), res.get('gap')
    return (res.get('status') == gp.GRB.OPTIMAL and obj is not None
            and math.isfinite(obj) and gap is not None
            and 0.0 <= gap <= GAP_TRUST_THRESHOLD)


def _cache_get(key):
    path = os.path.join(_CACHE_DIR, key + '.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'result' in data:
            if data.get('schema') not in (None, CACHE_SCHEMA):
                return None
            result = data.get('result')
        else:
            result = data
        return result if isinstance(result, dict) and _is_cacheable(result) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _cache_put(key, obj):
    if not _is_cacheable(obj):
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, key + '.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'schema': CACHE_SCHEMA, 'result': obj},
                  f, allow_nan=False)
    os.replace(tmp, path)


def scenario_probabilities(inst):
    S = inst['S']
    if not S or len(set(S)) != len(S):
        raise ValueError('Scenario set must be nonempty and have unique indices')
    probs = {s: float(inst['p'][s]) for s in S}
    if any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError('Scenario probabilities must be finite and nonnegative')
    total = math.fsum(probs.values())
    if total <= 0:
        raise ValueError('Scenario probabilities must have positive total mass')
    return {s: p / total for s, p in probs.items()}


def _normalized_instance(inst):
    out = copy.deepcopy(inst)
    out.setdefault('_original_probability_sum', math.fsum(inst['p'][s] for s in inst['S']))
    out['p'] = scenario_probabilities(inst)
    return out


def _scenario_input(inst):
    return bool(inst.get('random_input', False)) or any(
        isinstance(key, tuple) for key in inst['L_in'])


# Evaluate (1 - lambda) * E[C] + lambda * CVaR_alpha[C] over scenario total costs.
def risk_aggregate(costs, probs, lam, alpha):
    if not costs or len(costs) != len(probs):
        raise ValueError('Costs and probabilities must be nonempty and equal length')
    if not 0 <= lam <= 1 or not 0 <= alpha < 1:
        raise ValueError('Require lambda in [0,1] and alpha in [0,1)')
    if any(not math.isfinite(c) for c in costs):
        raise ValueError('Scenario costs must be finite')
    if any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError('Probabilities must be finite and nonnegative')
    total = math.fsum(probs)
    if total <= 0:
        raise ValueError('Probabilities must have positive total mass')
    weights = [p / total for p in probs]
    expected = math.fsum(p * c for p, c in zip(weights, costs))
    cvar = gurobi_solver._weighted_cvar(costs, weights, alpha)
    return {'expected': expected, 'cvar': cvar,
            'risk_adjusted': (1 - lam) * expected + lam * cvar}


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


# Load a .json(.gz) instance, rebuild nu_scen, and normalize probabilities.
def load_instance(path):
    with _open_json(path) as f:
        inst = _ensure_nu_scen(restore_keys(json.load(f)))
    inst['_original_probability_sum'] = math.fsum(inst['p'][s] for s in inst['S'])
    inst['p'] = scenario_probabilities(inst)
    return inst


def _instance_stem(fname):
    base = os.path.basename(fname)
    for ext in ('.json.gz', '.json'):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def _index_of(fname):
    base = _instance_stem(fname)
    try:
        return int(base.split('_')[-1])
    except ValueError:
        return 10 ** 9


def list_instance_paths(scale, instances_dir=INSTANCES_DIR, limit=None):
    by_stem = {}
    for ext in ('.json', '.json.gz'):
        for p in glob.glob(os.path.join(instances_dir, f"{scale}_*{ext}")):
            by_stem[_instance_stem(p)] = p
    paths = sorted(by_stem.values(), key=_index_of)
    return paths[:limit] if limit is not None else paths


def _rebuild_nu(inst, K, J):
    incomp = set(map(tuple, inst['Psi_incomp']))
    T_plan, tau, Theta, L_out = inst['T_plan'], inst['tau_overhead'], inst['Theta'], inst['L_out']
    nu_f = inst.get('nu_util_factor', 1.0)
    nu = {}
    for j in J:
        for k in K:
            nu[(j, k, 0)] = 0.0 if (j, k) in incomp else round(
                nu_f * T_plan / (L_out[(j, 0)] / Theta[k] + tau), 2)
    return nu


# Build the single-scenario EV problem at probability-weighted workload means.
def build_mean_instance(inst):
    ev = copy.deepcopy(inst)
    J, K, S = inst['J'], inst['K'], inst['S']
    p = scenario_probabilities(inst)
    ev['S'] = [0]
    ev['p'] = {0: 1.0}
    ev['D'] = {(j, 0): math.fsum(p[s] * inst['D'][(j, s)] for s in S) for j in J}
    ev['L_out'] = {(j, 0): math.fsum(p[s] * inst['L_out'][(j, s)] for s in S)
                   for j in J}
    if _scenario_input(inst):
        ev['L_in'] = {(j, 0): math.fsum(p[s] * inst['L_in'][(j, s)] for s in S)
                      for j in J}
    ev['nu_scen'] = _rebuild_nu(ev, K, J)
    return ev


# Extract a single-scenario instance for wait-and-see (WS) perfect-information solving.
def build_single_scenario_instance(inst, s):
    stochastic = {'S', 'p', 'D', 'L_out', 'L_in', 'nu_scen'}
    ws = {key: copy.deepcopy(value) for key, value in inst.items() if key not in stochastic}
    J, K = inst['J'], inst['K']
    ws['S'] = [0]
    ws['p'] = {0: 1.0}
    ws['D'] = {(j, 0): inst['D'][(j, s)] for j in J}
    ws['L_out'] = {(j, 0): inst['L_out'][(j, s)] for j in J}
    if _scenario_input(inst):
        ws['L_in'] = {(j, 0): inst['L_in'][(j, s)] for j in J}
    else:
        ws['L_in'] = copy.deepcopy(inst['L_in'])
    ws['nu_scen'] = {(j, k, 0): inst['nu_scen'][(j, k, s)] for j in J for k in K}
    return ws


def _solve_record(res):
    return {key: res.get(key) for key in ('obj_val', 'gap', 'status', 'time')}


def _require_optimal(res, label):
    if not _is_cacheable(res):
        raise RuntimeError(f'{label} not certified within numerical tolerance: '
                           f"status={res.get('status')}, gap={res.get('gap')}, "
                           f"obj={res.get('obj_val')}")


# Solve the two-stage recourse problem (RP) via Benders decomposition.
def solve_rp(inst, name=None):
    inst = _normalized_instance(inst)
    hit = _cache_get(f'{name}__rp') if name is not None else None
    if hit is not None:
        return hit
    res = benders_solver.solve(inst, time_limit=TIME_LIMIT, gap_tol=GAP_TOL)
    out = _solve_record(res)
    _require_optimal(out, f'{name} RP')
    if name is not None:
        _cache_put(f'{name}__rp', out)
    return out


def _recover_first_stage(inst, res):
    raw_n = {int(k): v for k, v in res['n'].items()}
    raw_y = {int(k): v for k, v in res['y'].items()}
    n, y, z = {}, {}, {}
    for k in inst['K']:
        nk = int(round(raw_n[k]))
        tol = 1e-7 * max(1, nk)
        if abs(raw_n[k] - nk) > 1e-7 or not 0 <= nk <= inst['N_bar'][k]:
            raise RuntimeError(f'EV integer decision invalid at GPU {k}')
        lo, hi = inst['Z_min'] * nk, inst['Z_max'] * nk
        if not math.isfinite(raw_y[k]) or not lo - tol <= raw_y[k] <= hi + tol:
            raise RuntimeError(f'EV reservation invalid at GPU {k}')
        n[k] = nk
        y[k] = min(hi, max(lo, raw_y[k]))
        z[k] = y[k] / nk if nk else inst['Z_min']
    return n, z, y


# Solve the EV problem and evaluate its first-stage plan across all scenarios (EEV).
def compute_eev(inst, name=None):
    inst = _normalized_instance(inst)
    si = _scenario_input(inst)
    ev = _cache_get(f'{name}__ev') if name is not None else None
    if ev is None:
        res = gurobi_solver.solve(build_mean_instance(inst), time_limit=TIME_LIMIT,
                                  gap_tol=GAP_TOL, return_first_stage=True,
                                  scenario_input=si)
        _require_optimal(res, f'{name} EV')
        ev_n, ev_z, ev_y = _recover_first_stage(inst, res)
        ev = dict(_solve_record(res), n=ev_n, z=ev_z, y=ev_y)
        if name is not None:
            _cache_put(f'{name}__ev', ev)
    ev_n, ev_z, ev_y = _recover_first_stage(inst, ev)
    eev = _cache_get(f'{name}__eev') if name is not None else None
    if eev is None:
        res = gurobi_solver.solve(inst, time_limit=TIME_LIMIT, gap_tol=GAP_TOL,
                                  fix_first_stage={'n': ev_n, 'y': ev_y},
                                  scenario_input=si)
        eev = _solve_record(res)
        _require_optimal(eev, f'{name} EEV')
        if name is not None:
            _cache_put(f'{name}__eev', eev)
    return dict(eev, ev_obj=ev['obj_val'], ev_status=ev['status'],
                ev_gap=ev['gap'])


def _init_ws_worker():
    gp.setParam('OutputFlag', 0)
    gp.setParam('Threads', 1)


def _solve_ws_scenario(single):
    res = gurobi_solver.solve(single, time_limit=TIME_LIMIT, gap_tol=GAP_TOL,
                              scenario_input=_scenario_input(single))
    return _solve_record(res)


# Solve all single-scenario WS subproblems in parallel and apply mean-CVaR aggregation.
def compute_ws(inst, name=None):
    inst = _normalized_instance(inst)
    S, p = inst['S'], inst['p']
    records, missing = {}, []
    t0 = time.time()
    for s in S:
        res = _cache_get(f'{name}__ws_{s}') if name is not None else None
        if res is None:
            missing.append(s)
        else:
            records[s] = res

    def store(s, res):
        _require_optimal(res, f'{name} WS scenario {s}')
        if name is not None:
            _cache_put(f'{name}__ws_{s}', res)
        records[s] = res
        if len(S) > 200 and len(records) % max(1, len(S) // 5) == 0:
            print(f'        WS {len(records)}/{len(S)}  ({time.time() - t0:.0f}s)', flush=True)

    if WS_WORKERS > 1 and len(missing) > WS_WORKERS:
        remaining = iter(missing)
        with ProcessPoolExecutor(max_workers=WS_WORKERS, initializer=_init_ws_worker) as pool:
            pending = {}

            def submit_next():
                s = next(remaining, None)
                if s is not None:
                    pending[pool.submit(_solve_ws_scenario,
                                        build_single_scenario_instance(inst, s))] = s

            for _ in range(2 * WS_WORKERS):
                submit_next()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    s = pending.pop(future)
                    store(s, future.result())
                    submit_next()
    else:
        for s in missing:
            store(s, _solve_ws_scenario(build_single_scenario_instance(inst, s)))
    costs = [records[s]['obj_val'] for s in S]
    risk = risk_aggregate(costs, [p[s] for s in S], inst['lambda_risk'], inst['alpha'])
    return dict(risk, n_fail=0, max_gap=max(records[s]['gap'] for s in S),
                n_optimal=len(S), scenario_costs=dict(zip(S, costs)))


# Compute RP, EEV, WS, VSS, and EVPI for one benchmark instance.
def solve_instance(scale, name, inst):
    K, J, S = inst['K'], inst['J'], inst['S']
    inst = copy.deepcopy(inst)
    inst.setdefault('_original_probability_sum', math.fsum(inst['p'][s] for s in S))
    inst['p'] = scenario_probabilities(inst)
    print(f'  [{name}] |K|={len(K)} |J|={len(J)} |S|={len(S)} '
          f"lambda={inst['lambda_risk']}, alpha={inst['alpha']}", flush=True)
    t0 = time.time()
    rp = solve_rp(inst, name=name)
    rp_val = rp['obj_val']
    print(f"    RP   = {rp_val:12.2f}  (gap {rp['gap'] * 100:.6f}%, "
          f'{time.time() - t0:.0f}s)', flush=True)
    t1 = time.time()
    eev = compute_eev(inst, name=name)
    eev_val = eev['obj_val']
    print(f'    EEV  = {eev_val:12.2f}  ({time.time() - t1:.0f}s)', flush=True)
    t2 = time.time()
    ws = compute_ws(inst, name=name)
    ws_val = ws['risk_adjusted']
    print(f'    WS_r = {ws_val:12.2f}  ({time.time() - t2:.0f}s, fail=0)', flush=True)
    vss, evpi = eev_val - rp_val, rp_val - ws_val
    tol = 1e-5 * max(1.0, abs(rp_val), abs(eev_val), abs(ws_val))
    if vss < -tol or evpi < -tol:
        raise RuntimeError(f'{name}: WS <= RP <= EEV violated beyond tolerance')
    return {
        'scale': scale, 'name': name, 'nk': len(K), 'nj': len(J), 'ns': len(S),
        'lambda_risk': inst['lambda_risk'], 'alpha': inst['alpha'],
        'original_probability_sum': inst['_original_probability_sum'],
        'rp': rp_val, 'rp_gap': rp['gap'],
        'rp_status': rp['status'], 'eev': eev_val, 'eev_status': eev['status'],
        'eev_gap': eev['gap'], 'ev_obj': eev['ev_obj'],
        'ev_status': eev['ev_status'], 'ev_gap': eev['ev_gap'],
        'ws': ws_val, 'ws_expected': ws['expected'], 'ws_cvar': ws['cvar'],
        'ws_fail': ws['n_fail'], 'ws_max_gap': ws['max_gap'],
        'ws_n_optimal': ws['n_optimal'], 'vss': vss, 'evpi': evpi,
        'vss_pct': vss / rp_val * 100 if rp_val > 1e-6 else None,
        'evpi_pct': evpi / rp_val * 100 if rp_val > 1e-6 else None,
        'untrusted': False,
    }


def _r2(v, p=2):
    return round(v, p) if v is not None else None


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    return sum(vals) / len(vals), min(vals), max(vals)


def _class_mean(results):
    rp_m, _, _ = _stats([r['rp'] for r in results])
    ws_m, _, _ = _stats([r['ws'] for r in results])
    eev_m, _, _ = _stats([r['eev'] for r in results])
    vss_m, _, _ = _stats([r['vss'] for r in results])
    evpi_m, _, _ = _stats([r['evpi'] for r in results])
    vss_pct_m = (vss_m / rp_m * 100.0) if (vss_m is not None and rp_m and rp_m > 1e-6) else None
    evpi_pct_m = (evpi_m / rp_m * 100.0) if (evpi_m is not None and rp_m and rp_m > 1e-6) else None
    return rp_m, ws_m, eev_m, vss_m, evpi_m, vss_pct_m, evpi_pct_m


def _vss_sheet(all_results):
    rows, bold = [], set()
    for scale, results in all_results.items():
        for r in results:
            rows.append([scale, r['name'], _r2(r['rp'], 0), _r2(r['eev'], 0),
                         _r2(r['vss'], 0), _r2(r['vss_pct'], 2)])
        rp_m, _, eev_m, vss_m, _, vss_pct_m, _ = _class_mean(results)
        bold.add(len(rows))
        rows.append(['', 'Mean', _r2(rp_m, 0), _r2(eev_m, 0),
                     _r2(vss_m, 0), _r2(vss_pct_m, 2)])
    return {
        'header': ['Class', 'Instance', 'RP', 'EEV', 'VSS', 'VSS%'],
        'rows': rows,
        'number_formats': {3: '#,##0', 4: '#,##0', 5: '#,##0', 6: '0.00'},
        'bold_rows': bold,
    }


def _evpi_sheet(all_results):
    rows, bold = [], set()
    for scale, results in all_results.items():
        for r in results:
            rows.append([scale, r['name'], _r2(r['ws'], 0), _r2(r['rp'], 0),
                         _r2(r['evpi'], 0), _r2(r['evpi_pct'], 2)])
        rp_m, ws_m, _, _, evpi_m, _, evpi_pct_m = _class_mean(results)
        bold.add(len(rows))
        rows.append(['', 'Mean', _r2(ws_m, 0), _r2(rp_m, 0),
                     _r2(evpi_m, 0), _r2(evpi_pct_m, 2)])
    return {
        'header': ['Class', 'Instance', 'WS', 'RP', 'EVPI', 'EVPI%'],
        'rows': rows,
        'number_formats': {3: '#,##0', 4: '#,##0', 5: '#,##0', 6: '0.00'},
        'bold_rows': bold,
    }


def _upsert_sheets(sheets, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        wb = load_workbook(path)
    else:
        wb = Workbook()
        wb.remove(wb.active)

    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor=_XLSX_HEADER_FILL)
    center = Alignment(horizontal='center', vertical='center')
    bold = Font(bold=True)

    for name, spec in sheets.items():
        if name in wb.sheetnames:
            wb.remove(wb[name])
        ws = wb.create_sheet(name)

        header = spec['header']
        ws.append(header)
        for c in range(1, len(header) + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center

        nfmt = spec.get('number_formats', {})
        bold_rows = spec.get('bold_rows', set())
        for r_off, row in enumerate(spec['rows']):
            excel_row = r_off + 2
            ws.append(row)
            is_bold = r_off in bold_rows
            for c in range(1, len(row) + 1):
                cell = ws.cell(row=excel_row, column=c)
                if c in nfmt and isinstance(cell.value, (int, float)):
                    cell.number_format = nfmt[c]
                if is_bold:
                    cell.font = bold
        ws.freeze_panes = 'A2'

    tmp = path + '.tmp'
    wb.save(tmp)
    os.replace(tmp, path)


def _fmt(v, w=12, p=2):
    return f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}"


# Atomically write full-precision VSS/EVPI summary JSON.
def save_json(all_results, path, complete=False):
    class_means = {}
    for scale, rows in all_results.items():
        rp, ws, eev, vss, evpi, vss_pct, evpi_pct = _class_mean(rows)
        class_means[scale] = dict(n=len(rows), rp=rp, ws=ws, eev=eev, vss=vss,
                                  evpi=evpi, vss_pct=vss_pct, evpi_pct=evpi_pct)
    payload = {
        'schema': CACHE_SCHEMA, 'complete': complete,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'definitions': {
            'EV': 'probability-weighted mean D/input/output; rebuild throughput',
            'risk': '(1-lambda)*E[total cost] + lambda*CVaR_alpha[total cost]',
            'VSS': 'risk-adjusted EEV - RP',
            'EVPI': 'RP - risk-adjusted perfect-information total cost',
            'class_percentage': 'ratio of mean costs, not mean of individual ratios',
        },
        'python': platform.python_version(), 'gurobi': gp.gurobi.version(),
        'gap_tol': GAP_TOL, 'certification_gap_threshold': GAP_TRUST_THRESHOLD,
        'ws_workers': WS_WORKERS,
        'results': all_results, 'class_means': class_means,
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


# CLI entry point for batch risk-adjusted VSS and EVPI evaluation.
def main():
    global WS_WORKERS
    parser = argparse.ArgumentParser(description='Risk-consistent finite-scenario VSS/EVPI')
    parser.add_argument('--scales', nargs='+', choices=['S', 'M', 'L'], default=SCALES)
    parser.add_argument('--instances', type=int, default=N_INSTANCES)
    parser.add_argument('--ws-workers', type=int, default=WS_WORKERS)
    parser.add_argument('--output', default=os.path.join(_PROJECT_ROOT, 'results_vss_evpi_risk_consistent.json'))
    parser.add_argument('--xlsx', default=None, help='Optional workbook; no workbook touched by default')
    args = parser.parse_args()
    if args.instances <= 0 or args.ws_workers <= 0:
        parser.error('--instances and --ws-workers must be positive')
    WS_WORKERS = args.ws_workers
    hdr = (f"{'Class':>5} {'Instance':>10} | {'WS_risk':>12} {'RP':>12} {'EEV':>12} "
           f"| {'EVPI':>12} {'EVPI%':>7} | {'VSS':>12} {'VSS%':>8}")
    print('[Experiment] Risk-adjusted VSS & EVPI (weighted workload means; total-cost CVaR)', flush=True)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    all_results = {}
    for scale in args.scales:
        paths = list_instance_paths(scale, INSTANCES_DIR, args.instances)
        if len(paths) != args.instances:
            raise RuntimeError(f'{scale}: expected {args.instances} instances, found {len(paths)}')
        print(f'\n=== {scale}-class : {len(paths)} instance(s) ===', flush=True)
        results = []
        all_results[scale] = results
        for path in paths:
            name = f'{scale}_{_index_of(path)}'
            r = solve_instance(scale, name, load_instance(path))
            results.append(r)
            print(f"{scale:>5} {name:>10} | {_fmt(r['ws'])} {r['rp']:>12.2f} "
                  f"{_fmt(r['eev'])} | {_fmt(r['evpi'])} {_fmt(r['evpi_pct'], 7)} "
                  f"| {_fmt(r['vss'])} {_fmt(r['vss_pct'], 8)}", flush=True)
            save_json(all_results, args.output)
            if args.xlsx:
                _upsert_sheets({'VSS': _vss_sheet(all_results),
                                'EVPI': _evpi_sheet(all_results)}, args.xlsx)
            gc.collect()
        rp, ws, eev, vss, evpi, vss_pct, evpi_pct = _class_mean(results)
        print(f"{scale:>5} {'Mean':>10} | {_fmt(ws)} {_fmt(rp)} {_fmt(eev)} "
              f'| {_fmt(evpi)} {_fmt(evpi_pct, 7)} | {_fmt(vss)} {_fmt(vss_pct, 8)}', flush=True)
        print('-' * len(hdr), flush=True)
    save_json(all_results, args.output, complete=True)
    print(f'[OK] full-precision results saved -> {args.output}', flush=True)


if __name__ == "__main__":
    main()
