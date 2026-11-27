# Benchmark comparison between Branch-and-Cut (EF) and Benders Decomposition (Tables 3 & 4).
import os
import gc
import gzip
import json
import glob
import math
import time
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gurobipy import GRB
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

from solvers import benders_solver, gurobi_solver

_CACHE_DEBUG = os.environ.get('CACHE_DEBUG', '0') == '1'

TIME_LIMIT = 3600
GAP_TOL = 0

INSTANCES_DIR = os.path.join(_PROJECT_ROOT, 'instances')
SCALES = ['S', 'M', 'L']
INSTANCE_LIMIT = None

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results.xlsx')
_XLSX_HEADER_FILL = '4472C4'

_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'cache', 'comparative')
_CACHE_FIELDS = ('obj_val', 'gap', 'time', 'status', 'nk', 'nj', 'ns')
_NONCACHEABLE_STATUS = {1, 11, 14}


def _is_cacheable(res):
    return res.get('status') not in _NONCACHEABLE_STATUS


def _cache_get(key):
    path = os.path.join(_CACHE_DIR, key + '.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(key, res):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    slim = {f: res.get(f) for f in _CACHE_FIELDS}
    path = os.path.join(_CACHE_DIR, key + '.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(slim, f)
    os.replace(tmp, path)


# Run solver with end-to-end wall-clock timing and JSON result caching.
def _run_cached(algo, inst, name, tl, gap, dims=None):
    key = f"{name}__{algo['key']}"
    hit = _cache_get(key)
    if hit is not None:
        if _CACHE_DEBUG:
            hit = dict(hit)
            hit['_cache'] = 'HIT'
        return hit

    t0 = time.time()
    res = algo['run'](inst, tl, gap)
    res = dict(res)
    res['time'] = time.time() - t0

    if res.get('status') == 11:
        raise KeyboardInterrupt("Interrupted by user during solve (GRB.INTERRUPTED)")

    if _is_cacheable(res):
        to_store = dict(res)
        if dims is not None:
            to_store['nk'], to_store['nj'], to_store['ns'] = dims
        _cache_put(key, to_store)
        if _CACHE_DEBUG:
            res = dict(res)
            res['_cache'] = 'WRITE'
    elif _CACHE_DEBUG:
        res = dict(res)
        res['_cache'] = 'MISS'
    return res


def _probe_cached(name, algorithms):
    out = {}
    for a in algorithms:
        hit = _cache_get(f"{name}__{a['key']}")
        if hit is None:
            return None
        if _CACHE_DEBUG:
            hit = dict(hit)
            hit['_cache'] = 'HIT'
        out[a['key']] = hit
    return out


def _dims_from_cache(cached):
    for res in cached.values():
        if res.get('nk') is not None:
            return res['nk'], res['nj'], res['ns']
    return None


def _open_json(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def _read_dims(path):
    with _open_json(path) as f:
        obj = json.load(f)
    return len(obj['K']), len(obj['J']), len(obj['S'])


# Benchmark solver configurations (EF with valid inequalities vs. Benders).
ALGORITHMS = [
    {
        'key': 'EF',
        'name': 'Extended Form (Gurobi)',
        'enabled': True,
        'run': lambda inst, tl, gap: gurobi_solver.solve(
            inst, time_limit=tl, gap_tol=gap, valid_inequalities=True,
            scenario_input=bool(inst.get('random_input', False))),
    },
    {
        'key': 'BD',
        'name': 'Benders Decomposition',
        'enabled': True,
        'run': lambda inst, tl, gap: benders_solver.solve(
            inst, time_limit=tl, gap_tol=gap),
    },
]

_TUPLE_KEY_FIELDS = {'D', 'L_out', 'nu_scen'}
_INT_KEY_FIELDS = {'C_cap', 'C_elast', 'M', 'N_bar', 'Theta', 'P_idle', 'P_peak',
                   'L_out_bar', 'sigma_out', 'sigma_in', 'sigma_tot', 'L_in_logsigma', 'C_rej', 'D_bar', 'p', 'gpu_tiers',
                   'L_in_mean', 'demand_cv', 'L_out_mu', 'L_out_sigma'}
_AUTO_KEY_FIELDS = {'L_in'}


# Restore integer and tuple dictionary keys after JSON deserialization.
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


# Reconstruct scenario per-slot service capacities nu_jk,s if omitted on disk.
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


# Load a .json(.gz) instance, rebuild nu_scen, and normalize probabilities.
def load_instance(path):
    with _open_json(path) as f:
        return _normalize_probabilities(_ensure_nu_scen(restore_keys(json.load(f))))


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
    if limit is not None:
        paths = paths[:limit]
    return paths


def status_str(code):
    return {GRB.OPTIMAL: "Opt", GRB.INFEASIBLE: "Inf",
            GRB.UNBOUNDED: "Unb", GRB.TIME_LIMIT: "TL"}.get(code, f"C{code}")


def enabled_algorithms():
    return [a for a in ALGORITHMS if a.get('enabled', False)]


def _instance_configs_sheet(rows):
    groups = []
    for row in rows:
        sc = str(row[0]).split('_')[0]
        if not groups or groups[-1][0] != sc:
            groups.append((sc, []))
        groups[-1][1].append(row)

    out_rows = []
    start = 1
    for sc, grp in groups:
        end = start + len(grp) - 1
        nK, nJ, nS = grp[0][1], grp[0][2], grp[0][3]
        span = f"{start}" if start == end else f"{start}--{end}"
        out_rows.append([f"{sc}-class", span, nK, nJ, nS])
        start = end + 1
    return {
        'header': ['Class', 'Instances', '|K|', '|J|', '|S|'],
        'rows': out_rows,
    }


def _algorithm_comparison_sheet(rows, algorithms):
    def time_or_tl(t, status):
        return "TL" if status == "TL" else t

    col_prefix = {'EF': 'BC', 'BD': 'BD'}
    header = ['ID', '|K|', '|J|', '|S|']
    nfmt = {}
    for idx, a in enumerate(algorithms):
        pfx = col_prefix.get(a['key'], a['key'])
        base_col = 5 + idx * 3
        header += [f'{pfx}_Obj', f'{pfx}_Gap%', f'{pfx}_Time']
        nfmt[base_col] = '#,##0'
        nfmt[base_col + 1] = '0.00'
        nfmt[base_col + 2] = '0.0'

    out_rows = []
    for i, r in enumerate(rows, start=1):
        out_row = [i, r[1], r[2], r[3]]
        for idx in range(len(algorithms)):
            off = 4 + idx * 4
            obj, gap, tm, st = r[off], r[off + 1], r[off + 2], r[off + 3]
            out_row += [obj, gap, time_or_tl(tm, st)]
        out_rows.append(out_row)

    return {
        'header': header,
        'rows': out_rows,
        'number_formats': nfmt,
    }


# Atomically upsert formatted worksheets into results.xlsx while keeping other sheets.
def _upsert_sheets(sheets, path=RESULT_FILE):
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


def save_xlsx(rows, algorithms, filename=RESULT_FILE):
    _upsert_sheets({
        'Instance_Configs': _instance_configs_sheet(rows),
        'Algorithm_Comparison': _algorithm_comparison_sheet(rows, algorithms),
    }, path=filename)


# Run the comparative benchmark across all configured scales and save results.xlsx.
def main():
    algorithms = enabled_algorithms()
    if not algorithms:
        print("[WARN] No algorithm enabled in settings; nothing to do.")
        return

    parts = [f"{'Instance':>10} | {'|K|':>3} {'|J|':>3} {'|S|':>5}"]
    for a in algorithms:
        k = a['key']
        parts.append(f"{k+'_Obj':>12} {k+'_Gap%':>8} {k+'_T':>8} {k:>4}")
    hdr = " | ".join(parts)
    print("[Experiment] Comparative Benchmark (Branch-and-Cut vs. Benders Decomposition)", flush=True)
    print('-' * len(hdr))
    print(hdr)
    print('-' * len(hdr))

    def fo(v): return f"{v:>12.2f}" if v is not None else f"{'N/A':>12}"
    def fg(v): return f"{v*100:>8.4f}" if v is not None else f"{'N/A':>8}"
    def ft(v): return f"{v:>8.1f}" if v is not None else f"{'N/A':>8}"

    rows = []
    for scale in SCALES:
        for path in list_instance_paths(scale, INSTANCES_DIR, INSTANCE_LIMIT):
            name = _instance_stem(path)

            cached = _probe_cached(name, algorithms)
            if cached is not None:
                dims = _dims_from_cache(cached)
                if dims is None:
                    dims = _read_dims(path)
                nK, nJ, nS = dims
                results = cached
                inst = None
            else:
                inst = load_instance(path)
                nK, nJ, nS = len(inst['K']), len(inst['J']), len(inst['S'])
                results = {a['key']: _run_cached(a, inst, name, TIME_LIMIT, GAP_TOL,
                                                 dims=(nK, nJ, nS))
                           for a in algorithms}

            line = f"{name:>10} | {nK:>3} {nJ:>3} {nS:>5}"
            for a in algorithms:
                res = results[a['key']]
                line += (f" | {fo(res['obj_val'])} {fg(res['gap'])} "
                         f"{ft(res['time'])} {status_str(res['status']):>4}")
            cache_status = ""
            if _CACHE_DEBUG:
                cache_status = " " + " ".join(
                    f"{a['key']}:(cache:{results[a['key']].get('_cache', 'MISS')})"
                    for a in algorithms)
            print(line + cache_status, flush=True)

            row = [name, nK, nJ, nS]
            for a in algorithms:
                res = results[a['key']]
                row += [
                    round(res['obj_val'], 2) if res['obj_val'] is not None else None,
                    round(res['gap'] * 100, 4) if res['gap'] is not None else None,
                    round(res['time'], 2) if res['time'] is not None else None,
                    status_str(res['status']),
                ]
            rows.append(row)

            save_xlsx(rows, algorithms, RESULT_FILE)
            del inst, results
            gc.collect()

    print('-' * len(hdr))
    save_xlsx(rows, algorithms, RESULT_FILE)
    print(f"[OK] Results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
