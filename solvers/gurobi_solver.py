# Extensive-form Branch-and-Cut MILP solver for the two-stage stochastic capacity planning model.
import copy
import math
import gurobipy as gp
from gurobipy import GRB


# Normalize scenario probabilities to sum to 1.
def _normalize_probs(raw_p, S):
    probs = {s: float(raw_p[s]) for s in S}
    total = math.fsum(probs.values())
    if total <= 0:
        raise ValueError("Scenario probabilities must have positive total mass")
    return {s: v / total for s, v in probs.items()}


# Compute exact weighted discrete CVaR at confidence level alpha.
def _weighted_cvar(costs, probs, alpha):
    tail = 1.0 - alpha
    if tail <= 0:
        return max(costs) if costs else 0.0
    order = sorted(zip(costs, probs), key=lambda t: t[0], reverse=True)
    acc, total = 0.0, 0.0
    for c, pr in order:
        if acc + pr < tail:
            total += c * pr
            acc += pr
        else:
            total += c * (tail - acc)
            acc = tail
            break
    return total / tail if acc > 0 else (order[0][0] if order else 0.0)


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


# Build single-scenario baseline-mean restriction for warm-starting.
def _build_mean_instance(inst):
    ev = copy.deepcopy(inst)
    J, K = inst['J'], inst['K']
    ev['S'] = [0]
    ev['p'] = {0: 1.0}
    ev['D'] = {(j, 0): inst['D_bar'][j] for j in J}
    ev['L_out'] = {(j, 0): float(inst['L_out_bar'][j]) for j in J}
    L_in = inst['L_in']
    if L_in and isinstance(next(iter(L_in)), tuple):
        ev['L_in'] = {(j, 0): float(inst['L_in_mean'][j]) for j in J}
    ev['nu_scen'] = _rebuild_nu(ev, K, J)
    return ev


def _is_scenario_input(inst, scenario_input):
    if scenario_input is not None:
        return bool(scenario_input)
    L_in = inst.get('L_in', {})
    return bool(inst.get('random_input', False)) or (
        len(L_in) > 0 and isinstance(next(iter(L_in)), tuple))


# Recover physical reservation ratio z_k = y_k / n_k (Proposition 1).
def _recover_z(n_val, y_val, Z_min, Z_max):
    if round(n_val) <= 0:
        return Z_min
    return min(Z_max, max(Z_min, y_val / n_val))


# Extract detailed cost breakdown, CVaR, and fleet metrics from a solved model.
def _ef_details(inst, n, y, x, e, r, scenario_input=None):
    K, J, S = inst['K'], inst['J'], inst['S']
    Psi = set(map(tuple, inst['Psi_incomp']))
    C_cap, C_elast, C_rej = inst['C_cap'], inst['C_elast'], inst['C_rej']
    C_energy, rho, T_op = inst['C_energy'], inst['rho'], inst['T_op']
    P_idle, P_peak, M = inst['P_idle'], inst['P_peak'], inst['M']
    L_in, L_out, mu_kv, xi = inst['L_in'], inst['L_out'], inst['mu_kv'], inst['xi']
    Z_min, Z_max = inst['Z_min'], inst['Z_max']
    p = _normalize_probs(inst['p'], S)
    D, alpha = inst['D'], inst['alpha']
    gpu_tiers = inst.get('gpu_tiers', {})

    _scen_in = _is_scenario_input(inst, scenario_input)
    def _Lin(j, s):
        return L_in[(j, s)] if _scen_in else L_in[j]

    n_opt = {k: n[k].X for k in K}
    y_opt = {k: y[k].X for k in K}
    z_opt = {k: _recover_z(n_opt[k], y_opt[k], Z_min, Z_max) for k in K}
    cap = sum(C_cap[k] * n_opt[k] for k in K)
    base_e = C_energy * rho * T_op * sum((P_idle[k] + P_peak[k]) / 2 * n_opt[k] for k in K)
    tiers = {'ultra-high': 0, 'high-end': 0, 'mid-range': 0, 'standard': 0}
    for k in K:
        t = gpu_tiers.get(k, 'unknown')
        if t in tiers:
            tiers[t] += round(n_opt[k])
    # Per-scenario demand scaling for numerical stability.
    dscale = {s: max(1.0, max(D[(j, s)] for j in J)) for s in S}
    scen_costs, tot_elastic, tot_demand = [], 0.0, 0.0
    for s in S:
        qx = dscale[s] * sum(C_elast[k] * e[j, k, s].X for j in J for k in K if (j, k) not in Psi)
        qr = dscale[s] * sum(C_rej[j] * r[j, s].X for j in J)
        de = rho * T_op * sum(
            (P_peak[k] - P_idle[k]) / M[k] * sum(
                (_Lin(j, s) + L_out[(j, s)]) * mu_kv * xi * x[j, k, s].X
                for j in J if (j, k) not in Psi)
            - (P_peak[k] - P_idle[k]) / 2 * n_opt[k] for k in K)
        scen_costs.append(qx + qr + C_energy * de)
        tot_elastic += p[s] * dscale[s] * sum(e[j, k, s].X for j in J for k in K if (j, k) not in Psi)
        tot_demand += p[s] * sum(D[(j, s)] for j in J)
    active = [k for k in K if round(n_opt[k]) > 0]
    return {
        'n_vals': {k: round(n_opt[k]) for k in K},
        'z_vals': {k: round(z_opt[k], 4) for k in K},
        'capital_cost': cap, 'energy_cost': base_e,
        'expected_recourse': sum(p[s] * scen_costs[i] for i, s in enumerate(S)),
        'scenario_recourse': scen_costs,
        'cvar_95': _weighted_cvar(scen_costs, [p[s] for s in S], alpha),
        'total_gpus': sum(round(n_opt[k]) for k in K),
        'avg_z': (sum(z_opt[k] for k in active) / len(active)) if active else 0.0,
        'cloud_burst_ratio': (tot_elastic / tot_demand) if tot_demand > 0 else 0.0,
        'tier_counts': tiers,
    }


# Solve the McCormick-linearized extensive form (optionally fixing stage 1 or adding valid inequalities).
def solve(inst, time_limit=1200, gap_tol=1e-4, fix_first_stage=None,
          valid_inequalities=False, lambda_override=None,
          return_first_stage=False, return_details=False,
          scenario_input=None):
    K, J, S = inst['K'], inst['J'], inst['S']
    Psi = set(map(tuple, inst['Psi_incomp']))
    C_cap, C_elast, M, N_bar = inst['C_cap'], inst['C_elast'], inst['M'], inst['N_bar']
    Theta, P_idle, P_peak = inst['Theta'], inst['P_idle'], inst['P_peak']
    L_in, C_rej, sigma_out = inst['L_in'], inst['C_rej'], inst['sigma_out']
    sigma_tot = inst.get('sigma_tot', sigma_out)
    mu_kv, xi, beta = inst['mu_kv'], inst['xi'], inst['beta']
    Z_min, Z_max = inst['Z_min'], inst['Z_max']
    T_slot, T_op = inst['T_slot'], inst['T_op']
    C_energy, rho, kappa = inst['C_energy'], inst['rho'], inst['kappa']
    W_bar, E_bar, alpha = inst['W_bar'], inst['E_bar'], inst['alpha']
    lam = inst['lambda_risk'] if lambda_override is None else lambda_override
    p = _normalize_probs(inst['p'], S)
    D, L_out, nu = inst['D'], inst['L_out'], inst['nu_scen']

    _scen_in = _is_scenario_input(inst, scenario_input)
    def _Lin(j, s):
        return L_in[(j, s)] if _scen_in else L_in[j]

    m = gp.Model("EF")
    m.Params.OutputFlag = 0
    m.Params.MIPGap = gap_tol
    m.Params.MIPGapAbs = 1e-6
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    m.Params.IntFeasTol = 1e-9
    m.Params.TimeLimit = time_limit
    m.Params.NumericFocus = 3

    # Per-scenario demand scaling for numerical stability.
    dscale = {s: max(1.0, max(D[(j, s)] for j in J)) for s in S}

    if fix_first_stage is not None:
        n = m.addVars(K, vtype=GRB.CONTINUOUS, name="n")
        z = m.addVars(K, vtype=GRB.CONTINUOUS, lb=Z_min, ub=Z_max, name="z")
        _fix_z = fix_first_stage.get('z')
        for k in K:
            n[k].lb = n[k].ub = fix_first_stage['n'][k]
            if _fix_z is not None:
                z[k].lb = z[k].ub = _fix_z[k]
    else:
        n = m.addVars(K, vtype=GRB.INTEGER, lb=0, ub={k: N_bar[k] for k in K}, name="n")
        z = m.addVars(K, vtype=GRB.CONTINUOUS, lb=Z_min, ub=Z_max, name="z")

    eta = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="eta")
    phi = m.addVars(S, lb=0, name="phi")
    y = m.addVars(K, lb=0, name="y")
    x = m.addVars(J, K, S, lb=0, name="x")
    e = m.addVars(J, K, S, lb=0, name="e")
    r = m.addVars(J, S, lb=0, name="r")

    if fix_first_stage is not None:
        fixed_y = fix_first_stage.get('y')
        for k in K:
            y[k].lb = y[k].ub = (fixed_y[k] if fixed_y is not None
                                 else fix_first_stage['n'][k] * fix_first_stage['z'][k])

    # Objective: capital + baseline energy + (1 - lambda) * E[Q] + lambda * CVaR.
    E_base = rho * T_op * gp.quicksum((P_idle[k] + P_peak[k]) / 2 * n[k] for k in K)
    Delta_E = {s: rho * T_op * gp.quicksum(
        (P_peak[k] - P_idle[k]) / M[k] * gp.quicksum(
            (_Lin(j, s) + L_out[(j, s)]) * mu_kv * xi * x[j, k, s]
            for j in J if (j, k) not in Psi)
        - (P_peak[k] - P_idle[k]) / 2 * n[k] for k in K) for s in S}
    Q = {s: dscale[s] * gp.quicksum(C_elast[k] * e[j, k, s] for j in J for k in K if (j, k) not in Psi)
            + dscale[s] * gp.quicksum(C_rej[j] * r[j, s] for j in J)
            + C_energy * Delta_E[s] for s in S}

    m.setObjective(
        gp.quicksum(C_cap[k] * n[k] for k in K) + C_energy * E_base
        + (1 - lam) * gp.quicksum(p[s] * Q[s] for s in S)
        + lam * (eta + (1 / (1 - alpha)) * gp.quicksum(p[s] * phi[s] for s in S)),
        GRB.MINIMIZE)

    # First-stage supply, power, carbon, CVaR shortfall, and McCormick constraints.
    for k in K:
        m.addConstr(n[k] <= N_bar[k])
    m.addConstr(rho * gp.quicksum(P_peak[k] * n[k] for k in K) <= W_bar)
    m.addConstr(kappa * E_base <= E_bar)
    for s in S:
        m.addConstr(phi[s] >= Q[s] - eta)
    for k in K:
        m.addConstr(y[k] >= Z_min * n[k])
        m.addConstr(y[k] >= Z_max * n[k] + z[k] * N_bar[k] - Z_max * N_bar[k])
        m.addConstr(y[k] <= Z_max * n[k])
        m.addConstr(y[k] <= Z_min * n[k] + z[k] * N_bar[k] - Z_min * N_bar[k])

    # Second-stage flow, memory wall, safety-capacity, throughput, and compatibility constraints.
    for s in S:
        for j in J:
            m.addConstr(gp.quicksum(nu[(j, k, s)] / dscale[s] * x[j, k, s] + e[j, k, s]
                                    for k in K if (j, k) not in Psi) + r[j, s] == D[(j, s)] / dscale[s])
        for k in K:
            m.addConstr(gp.quicksum((_Lin(j, s) + L_out[(j, s)]) * mu_kv * xi * x[j, k, s]
                                    for j in J if (j, k) not in Psi)
                        <= n[k] * M[k] - y[k] * M[k])
            m.addConstr(gp.quicksum(beta * sigma_tot[j] * mu_kv * xi * x[j, k, s]
                                    for j in J if (j, k) not in Psi)
                        <= y[k] * M[k])
            m.addConstr(gp.quicksum(L_out[(j, s)] / T_slot * x[j, k, s]
                                    for j in J if (j, k) not in Psi)
                        <= n[k] * Theta[k])
        for (j, k) in Psi:
            m.addConstr(x[j, k, s] == 0)
            m.addConstr(e[j, k, s] == 0)

    # Aggregate valid inequalities and per-scenario elastic offloading bounds.
    if valid_inequalities:
        for k in K:
            m.addConstr(gp.quicksum(p[s] * gp.quicksum(
                (_Lin(j, s) + L_out[(j, s)]) * mu_kv * xi * x[j, k, s]
                for j in J if (j, k) not in Psi) for s in S) <= n[k] * M[k] - y[k] * M[k])
            m.addConstr(gp.quicksum(p[s] * gp.quicksum(
                L_out[(j, s)] / T_slot * x[j, k, s]
                for j in J if (j, k) not in Psi) for s in S) <= n[k] * Theta[k])
            m.addConstr(gp.quicksum(p[s] * gp.quicksum(
                beta * sigma_tot[j] * mu_kv * xi * x[j, k, s]
                for j in J if (j, k) not in Psi) for s in S) <= y[k] * M[k])
        for s in S:
            for j in J:
                m.addConstr(gp.quicksum(e[j, k, s] for k in K if (j, k) not in Psi)
                            <= D[(j, s)] / dscale[s])

    # Warm-start first-stage decisions from the baseline-mean restriction.
    if valid_inequalities and fix_first_stage is None:
        warm = solve(_build_mean_instance(inst), time_limit=time_limit,
                     gap_tol=gap_tol, lambda_override=lambda_override,
                     return_first_stage=True, scenario_input=_scen_in)
        if warm.get('status') == GRB.INTERRUPTED:
            m.dispose()
            del m
            return {'status': GRB.INTERRUPTED, 'obj_val': None, 'gap': None,
                    'time': warm.get('time', 0.0)}
        if warm.get('n') is not None:
            for k in K:
                n[k].Start = warm['n'][k]
                z[k].Start = warm['z'][k]
                y[k].Start = warm['y'][k]

    try:
        m.optimize()
    except (gp.GurobiError, MemoryError) as ex:
        msg = str(ex).lower()
        is_oom = isinstance(ex, MemoryError) or 'out of memory' in msg \
            or 'memory' in msg or getattr(ex, 'errno', None) == 10001
        res = {'status': GRB.MEM_LIMIT if is_oom else GRB.NUMERIC,
               'obj_val': None, 'gap': None, 'time': m.Runtime if hasattr(m, 'Runtime') else 0.0,
               'error': str(ex)}
        m.dispose()
        del m
        return res
    res = {'status': m.Status, 'obj_val': None, 'gap': None, 'time': m.Runtime}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        res['obj_val'] = m.ObjVal
        res['gap'] = m.MIPGap if m.IsMIP else 0.0
        if return_first_stage:
            res['n'] = {k: round(n[k].X) for k in K}
            res['y'] = {k: y[k].X for k in K}
            res['z'] = {k: _recover_z(res['n'][k], res['y'][k], Z_min, Z_max) for k in K}
        if return_details:
            res.update(_ef_details(inst, n, y, x, e, r, scenario_input=_scen_in))
    m.dispose()
    del m
    return res
